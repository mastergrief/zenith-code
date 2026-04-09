#!/usr/bin/env python3
"""Zenith Code Python Harness — multi-agent terminal coding assistant powered by local Qwen 3.5."""

import atexit
import os
import sys
import json
import time
from pathlib import Path

from agents.agent import Agent, DEFAULT_MODEL, EFFORT_LEVELS, detect_backend, detect_llamacpp_model
from agents.compact import detect_context_limit
from agents.config import load_config
from agents.permissions import PermissionMode
from agents.coordinator import Coordinator
from agents.swarm import Swarm
from agents.specialist_coordinator import SpecialistCoordinator, detect_specialists
from agents.history import HistoryLog
from agents.session import save_session, load_session, list_sessions

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}  ╔═══════════════════════════════════════════════╗
  ║     ZENITH CODE — Multi-Agent Harness          ║
  ║     Powered by Qwen 3.5 via Ollama / llama.cpp ║
  ╚═══════════════════════════════════════════════╝{RESET}

  {DIM}Type a message to chat with your agent team.
  Commands:
    /help          — show this help
    /agents        — list active agents
    /switch <name> — switch to a specific agent
    /team          — use coordinator mode (delegates to agents)
    /solo          — use single agent mode (default)
    /spawn <name> <role> — create a new agent
    /reset         — clear conversation history
    /cd <path>     — change working directory
    /model <name>  — switch model (e.g. qwen3:4b)
    /backend       — show/switch backend (ollama/llamacpp)
    /history       — show session event log
    /save          — save active agent's session
    /sessions      — list saved sessions
    /load <id>     — load a saved session
    /effort [low|medium|max] — show/set reasoning effort level
    /distill status — show available specialist models
    /distill on    — enable specialist routing
    /distill off   — disable specialist routing
    /exit          — quit{RESET}
"""


class Harness:
    """Terminal REPL harness for multi-agent system."""

    def __init__(self, model: str = DEFAULT_MODEL, backend: str | None = None, ctx_size: int | None = None):
        self.model = model
        self.backend = backend or detect_backend()
        self.ctx_size = ctx_size
        self.agents: dict[str, Agent] = {}
        self.active_agent: str = "coder"
        self.team_mode: bool = False
        self.coordinator: Coordinator | None = None
        self.specialist_mode: bool = False
        self.specialist_coordinator: SpecialistCoordinator | None = None
        self.history_log = HistoryLog()
        self._streaming_text = False  # Track whether we're mid-stream
        self._streaming_thinking = False  # Track thinking block display
        self.permission_mode = PermissionMode.WORKSPACE_WRITE

        # Discover the actual loaded GGUF once, so per-model compaction limits
        # from agents.compact.MODEL_CONTEXT_LIMITS can fire. See _compute_compact_threshold.
        self._loaded_llamacpp_model: str | None = (
            detect_llamacpp_model() if self.backend == "llamacpp" else None
        )

        # Create default agents
        self._spawn("coder", "expert software engineer who writes clean, efficient code")
        self._spawn("reviewer", "code reviewer who finds bugs, security issues, and suggests improvements")
        self._spawn("planner", "software architect who breaks down problems and designs solutions")

    def _compute_compact_threshold(self) -> int | None:
        """Compute the compaction threshold for new Agents.

        Returns the smaller of:
          - the per-model validated limit from ``agents.compact.MODEL_CONTEXT_LIMITS``
            (e.g. 232960 = 227.5K for Gemma 4 E4B, 130K for Qwen 3.5 4B)
          - 89% of the server-allocated ctx_size (so there's headroom for the
            active turn's response before the hard cap is hit)

        Returns ``None`` to let Agent auto-detect if we can't determine either:
        no ctx_size set AND no llama.cpp model discoverable.

        The 0.89 multiplier was raised from 0.85 (2026-04-08) to allow
        Gemma's 232960-token entry to clear the safe_ctx floor at the default
        256K ctx_size. Headroom at default ctx is 262144 - 232960 = 29184,
        which is BELOW EFFORT_LEVELS["max"]["max_tokens"] (32768) — max-effort
        responses can soft-truncate by up to ~3.5K when the conversation sits
        right at the threshold. After compaction fires, full 32K is available.
        Smaller ctx_size still binds via safe_ctx (e.g. ZENITH_CTX=131072 →
        safe_ctx 116654 → binds below the model limit).
        """
        if self.backend != "llamacpp":
            # Ollama path — let Agent's own detect_context_limit handle it
            return None

        model_name = self._loaded_llamacpp_model or "llamacpp"
        model_limit = detect_context_limit(model_name)

        if self.ctx_size:
            # Cap at 89% of server ctx so the next request can still fit its response
            safe_ctx = int(self.ctx_size * 0.89)
            return min(model_limit, safe_ctx)
        return model_limit

    def _spawn(self, name: str, role: str) -> Agent:
        """Create a new agent."""
        agent = Agent(
            name=name,
            role=role,
            model=self.model,
            backend=self.backend,
            max_context_tokens=self._compute_compact_threshold(),
        )
        self.agents[name] = agent
        return agent

    def _rebuild_coordinator(self):
        """Rebuild coordinator with current agent list."""
        self.coordinator = Coordinator(list(self.agents.values()), model=self.model)

    def _auto_save(self):
        """Auto-save active agent session on exit."""
        agent = self.agents[self.active_agent]
        if agent.history:
            path = save_session(agent)
            print(f"\n  {DIM}Session auto-saved to {path}{RESET}")

    def _confirm(self, tool_name: str, tool_args: dict, reason: str) -> bool:
        """Prompt user to confirm a risky tool call."""
        if self._streaming_text:
            print(f"{RESET}")
            self._streaming_text = False
        if self._streaming_thinking:
            self._streaming_thinking = False
        summary = ""
        if tool_name == "bash":
            summary = tool_args.get("command", "")[:80]
        elif tool_name in ("write_file", "edit_file"):
            summary = tool_args.get("path", "")
        print(f"\n  {YELLOW}{BOLD}[confirm]{RESET} {reason}")
        print(f"    {DIM}{tool_name}: {summary}{RESET}")
        try:
            answer = input(f"    Allow? [y/N] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        return answer in ("y", "yes")

    def _ask_user(self, question: str, options: list[str] | None) -> str:
        """Prompt the user with a freeform or multiple-choice question.

        Wired onto Agent instances in `_chat()` so the AskUserQuestion tool
        in `agents.tools._run_ask_user_question` can call back into the
        harness UI. Returns the user's answer as a string. On EOF/Ctrl-C
        returns a sentinel string so the model gets a clear "no answer"
        signal instead of crashing the tool loop.
        """
        # Close any open ANSI block before prompting
        if self._streaming_text:
            print(f"{RESET}")
            self._streaming_text = False
        if self._streaming_thinking:
            self._streaming_thinking = False

        print(f"\n  {YELLOW}{BOLD}[ask]{RESET} {question}")
        if options:
            for i, opt in enumerate(options, 1):
                print(f"    {DIM}{i}.{RESET} {opt}")
            try:
                raw = input(f"    Choice [1-{len(options)} or freeform]: ").strip()
            except (KeyboardInterrupt, EOFError):
                return "(no answer — input closed)"
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            return raw
        try:
            return input(f"    Answer: ").strip()
        except (KeyboardInterrupt, EOFError):
            return "(no answer — input closed)"

    def _on_event(self, event_type: str, data: dict):
        """Handle agent events for display."""
        if event_type == "thinking_start":
            print(f"\n  {DIM}{MAGENTA}thinking...{RESET}", end="", flush=True)
            self._streaming_thinking = True
            return

        elif event_type == "thinking_token":
            # Stream thinking tokens dimmed
            if not self._streaming_thinking:
                print(f"\n  {DIM}", end="")
                self._streaming_thinking = True
            return  # suppress thinking tokens from display (too verbose)

        elif event_type == "thinking_end":
            if self._streaming_thinking:
                self._streaming_thinking = False
            return

        elif event_type == "token":
            # Streaming token — print inline
            if not self._streaming_text:
                print(f"\n  {GREEN}", end="")
                self._streaming_text = True
            print(data["text"], end="", flush=True)
            return

        elif event_type == "response":
            # Don't reset _streaming_text here — the main loop checks it to
            # decide whether to re-print the response. Resetting here causes
            # the main loop to take the "non-streamed" branch and double-print
            # everything that was already streamed.
            if self._streaming_text:
                print(f"{RESET}", end="")
            self.history_log.add("response", data.get("content", "")[:100])
            return

        elif event_type == "compact":
            removed = data.get("removed", 0)
            remaining = data.get("remaining", 0)
            print(f"\n  {MAGENTA}{DIM}(compacted {removed} messages, {remaining} remaining){RESET}")
            self.history_log.add("compact", f"removed={removed} remaining={remaining}")
            return

        elif event_type == "tool_call":
            # End streaming if mid-stream before tool output
            if self._streaming_text:
                print(f"{RESET}")
                self._streaming_text = False
            if self._streaming_thinking:
                self._streaming_thinking = False

            name = data["name"]
            args = data["args"]
            print(f"\n  {YELLOW}{BOLD}[tool]{RESET} {YELLOW}{name}{RESET}", end="")
            if name == "bash":
                print(f" {DIM}$ {args.get('command', '')}{RESET}")
            elif name == "read_file":
                print(f" {DIM}{args.get('path', '')}{RESET}")
            elif name == "write_file":
                print(f" {DIM}{args.get('path', '')}{RESET}")
            elif name == "edit_file":
                print(f" {DIM}{args.get('path', '')}{RESET}")
            elif name == "grep":
                print(f" {DIM}/{args.get('pattern', '')}/{RESET}")
            elif name == "list_files":
                print(f" {DIM}{args.get('pattern', '')}{RESET}")
            else:
                print()
            self.history_log.add("tool_call", f"{name}({json.dumps(args)[:80]})")

        elif event_type == "tool_result":
            output = data["output"]
            lines = output.split("\n")
            preview = "\n    ".join(lines[:10])
            if len(lines) > 10:
                preview += f"\n    {DIM}... ({len(lines) - 10} more lines){RESET}"
            print(f"    {DIM}{preview}{RESET}")
            self.history_log.add("tool_result", f"{data.get('name', '?')}: {output[:80]}")

        elif event_type == "model_swap":
            model = data.get("model", "?")
            print(f"\n  {MAGENTA}{DIM}swapping to {model}...{RESET}", end="", flush=True)

        elif event_type == "model_ready":
            elapsed = data.get("elapsed", 0)
            print(f" ready ({elapsed:.1f}s){RESET}")

        elif event_type == "usage":
            pass  # Token count displayed inline with timing

    def _handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if handled."""
        parts = cmd.strip().split(maxsplit=2)
        command = parts[0].lower()

        if command == "/help":
            print(BANNER)

        elif command == "/agents":
            print(f"\n  {BOLD}Active agents:{RESET}")
            for name, agent in self.agents.items():
                marker = f" {GREEN}<- active{RESET}" if name == self.active_agent else ""
                print(f"    {CYAN}{name}{RESET} — {agent.role}{marker}")
            print(f"\n  Mode: {MAGENTA}{'team' if self.team_mode else 'solo'}{RESET}")

        elif command == "/switch":
            if len(parts) < 2:
                print(f"  {RED}Usage: /switch <agent_name>{RESET}")
            elif parts[1] in self.agents:
                self.active_agent = parts[1]
                self.team_mode = False
                print(f"  Switched to {CYAN}{self.active_agent}{RESET}")
            else:
                print(f"  {RED}Unknown agent '{parts[1]}'. Use /agents to see list.{RESET}")

        elif command == "/team":
            self.team_mode = True
            self._rebuild_coordinator()
            print(f"  {MAGENTA}Team mode{RESET} — coordinator will delegate to agents.")

        elif command == "/solo":
            self.team_mode = False
            print(f"  {MAGENTA}Solo mode{RESET} — talking to {CYAN}{self.active_agent}{RESET}")

        elif command == "/spawn":
            if len(parts) < 3:
                print(f"  {RED}Usage: /spawn <name> <role description>{RESET}")
            else:
                name, role = parts[1], parts[2]
                self._spawn(name, role)
                print(f"  Spawned agent {CYAN}{name}{RESET} — {role}")

        elif command == "/reset":
            for agent in self.agents.values():
                agent.reset()
            print(f"  {GREEN}All agent histories cleared.{RESET}")

        elif command == "/cd":
            if len(parts) < 2:
                print(f"  {os.getcwd()}")
            else:
                try:
                    os.chdir(os.path.expanduser(parts[1]))
                    print(f"  {GREEN}Changed to {os.getcwd()}{RESET}")
                except OSError as e:
                    print(f"  {RED}{e}{RESET}")

        elif command == "/model":
            if len(parts) < 2:
                print(f"  Current model: {CYAN}{self.model}{RESET}")
            else:
                self.model = parts[1]
                for agent in self.agents.values():
                    agent.model = self.model
                print(f"  Model set to {CYAN}{self.model}{RESET} for all agents.")

        elif command == "/backend":
            if len(parts) < 2:
                detected = detect_backend()
                print(f"  Current backend: {CYAN}{self.backend}{RESET}")
                print(f"  Auto-detected:   {CYAN}{detected}{RESET}")
                if self.backend == "llamacpp":
                    print(f"  {DIM}llama.cpp at localhost:8080, thinking enabled{RESET}")
                else:
                    print(f"  {DIM}Ollama at localhost:11434{RESET}")
            elif parts[1] in ("ollama", "llamacpp"):
                self.backend = parts[1]
                # Re-discover loaded GGUF for the new backend and recompute the
                # compaction threshold from the per-model limits table.
                if self.backend == "llamacpp":
                    self._loaded_llamacpp_model = detect_llamacpp_model()
                new_threshold = self._compute_compact_threshold()
                for agent in self.agents.values():
                    agent.backend = self.backend
                    if new_threshold is not None:
                        agent.max_context_tokens = new_threshold
                print(f"  Backend set to {CYAN}{self.backend}{RESET} for all agents.")
                if self.backend == "llamacpp" and new_threshold:
                    print(f"  {DIM}Compaction threshold: {new_threshold:,} tokens{RESET}")
            else:
                print(f"  {RED}Usage: /backend [ollama|llamacpp]{RESET}")

        elif command == "/effort":
            if len(parts) < 2:
                current = self.agents[self.active_agent].effort
                info = EFFORT_LEVELS[current]
                print(f"  Effort: {CYAN}{current}{RESET} (max_tokens={info['max_tokens']})")
                for level, cfg in EFFORT_LEVELS.items():
                    marker = " ←" if level == current else ""
                    print(f"    {level:8s} — {cfg['max_tokens']:5d} tokens{marker}")
            elif parts[1] in EFFORT_LEVELS:
                for agent in self.agents.values():
                    agent.effort = parts[1]
                info = EFFORT_LEVELS[parts[1]]
                print(f"  Effort set to {CYAN}{parts[1]}{RESET} (max_tokens={info['max_tokens']}) for all agents.")
            else:
                print(f"  {RED}Usage: /effort [low|medium|max]{RESET}")

        elif command == "/history":
            print(f"\n{self.history_log.as_markdown()}")

        elif command == "/save":
            agent = self.agents[self.active_agent]
            path = save_session(agent)
            print(f"  {GREEN}Session saved to {path}{RESET}")

        elif command == "/sessions":
            sessions = list_sessions()
            if not sessions:
                print(f"  {DIM}No saved sessions.{RESET}")
            else:
                print(f"\n  {BOLD}Saved sessions:{RESET}")
                for s in sessions[:20]:
                    print(f"    {CYAN}{s['id']}{RESET} — {s['name']} ({s['messages']} msgs, {s['saved_at'][:19]})")

        elif command == "/load":
            if len(parts) < 2:
                print(f"  {RED}Usage: /load <session_id>{RESET}")
            else:
                try:
                    data = load_session(parts[1])
                    agent = self.agents[self.active_agent]
                    agent.history = data["history"]
                    print(f"  {GREEN}Loaded {len(data['history'])} messages into {self.active_agent}{RESET}")
                except FileNotFoundError:
                    print(f"  {RED}Session '{parts[1]}' not found. Use /sessions to list.{RESET}")
                except Exception as e:
                    print(f"  {RED}Error loading session: {e}{RESET}")

        elif command == "/resume":
            sessions = list_sessions()
            if sessions:
                data = load_session(sessions[0]["id"])
                agent = self.agents[self.active_agent]
                agent.history = data["history"]
                print(f"  {GREEN}Resumed {len(data['history'])} messages{RESET}")
            else:
                print(f"  {DIM}No saved sessions found{RESET}")

        elif command == "/swap":
            # Hot-swap the llama.cpp server to a different GGUF.
            # Accepts: full path, name stem, or substring that uniquely
            # matches one file in ~/models/*.gguf.
            from agents.model_swap import LlamaServerManager, ModelSwapError
            mgr = LlamaServerManager()
            if len(parts) < 2:
                current = mgr.current_model()
                if current:
                    print(f"  Current: {CYAN}{current.name}{RESET}")
                    print(f"  {DIM}{current}{RESET}")
                else:
                    print(f"  {DIM}No model loaded (is llama-server running?){RESET}")
                # List available
                models_dir = Path.home() / "models"
                if models_dir.exists():
                    ggufs = sorted(models_dir.glob("*.gguf"))
                    if ggufs:
                        print(f"\n  {BOLD}Available in ~/models/:{RESET}")
                        for g in ggufs:
                            marker = f" {GREEN}<- loaded{RESET}" if current and g.resolve() == current.resolve() else ""
                            size_gb = g.stat().st_size / 1e9
                            print(f"    {CYAN}{g.stem}{RESET} ({size_gb:.1f} GB){marker}")
            else:
                target_arg = parts[1]
                target = Path(target_arg).expanduser()
                if not target.exists():
                    # Try exact basename match first (with or without .gguf
                    # extension), then fall back to substring match.
                    models_dir = Path.home() / "models"
                    ggufs = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
                    exact = [
                        g for g in ggufs
                        if target_arg == g.name or target_arg == g.stem
                    ]
                    if len(exact) == 1:
                        target = exact[0]
                    else:
                        candidates = [
                            g for g in ggufs
                            if target_arg.lower() in g.stem.lower()
                        ]
                        if len(candidates) == 1:
                            target = candidates[0]
                        elif len(candidates) > 1:
                            print(f"  {RED}Ambiguous '{target_arg}':{RESET}")
                            for c in candidates:
                                print(f"    {DIM}{c.name}{RESET}")
                            return True
                        else:
                            print(f"  {RED}Model not found: {target_arg}{RESET}")
                            return True
                print(f"  {MAGENTA}Swapping to {target.name}...{RESET}")
                try:
                    elapsed = mgr.swap(
                        target,
                        on_event=lambda k, p: print(f"  {DIM}{k}...{RESET}", flush=True),
                    )
                    if elapsed == 0:
                        print(f"  {DIM}(already loaded){RESET}")
                    else:
                        print(f"  {GREEN}Ready in {elapsed:.1f}s{RESET}")
                        # Recompute compaction threshold for the newly loaded model
                        self._loaded_llamacpp_model = detect_llamacpp_model()
                        new_threshold = self._compute_compact_threshold()
                        if new_threshold is not None:
                            for agent in self.agents.values():
                                agent.max_context_tokens = new_threshold
                            print(f"  {DIM}Compaction threshold: {new_threshold:,} tokens{RESET}")
                        print(f"  {DIM}Consider /reset — agent history was built with the old model{RESET}")
                except ModelSwapError as e:
                    print(f"  {RED}Swap failed: {e}{RESET}")

        elif command == "/distill":
            subcmd = parts[1] if len(parts) > 1 else "status"
            if subcmd == "status":
                specialists = detect_specialists()
                from agents.distill.config import DOMAINS as DIST_DOMAINS
                print(f"\n  {BOLD}Specialist models:{RESET}")
                for domain, config in DIST_DOMAINS.items():
                    name = config["ollama_name"]
                    if domain in specialists:
                        print(f"    {GREEN}{name:30s} available{RESET}")
                    else:
                        print(f"    {DIM}{name:30s} not trained{RESET}")
                print(f"\n  Specialist routing: {MAGENTA}{'ON' if self.specialist_mode else 'OFF'}{RESET}")
            elif subcmd == "on":
                self.specialist_mode = True
                self.specialist_coordinator = SpecialistCoordinator(fallback_model=self.model)
                status = self.specialist_coordinator.specialist_status
                print(f"  {GREEN}Specialist routing enabled.{RESET}")
                for domain, model in status.items():
                    print(f"    {domain}: {CYAN}{model}{RESET}")
            elif subcmd == "off":
                self.specialist_mode = False
                print(f"  {YELLOW}Specialist routing disabled.{RESET}")
            else:
                print(f"  {RED}Usage: /distill [status|on|off]{RESET}")

        elif command in ("/exit", "/quit", "/q"):
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            sys.exit(0)

        else:
            print(f"  {RED}Unknown command: {command}. Type /help for commands.{RESET}")

        return True

    def _chat(self, message: str) -> str:
        """Send message to active agent, coordinator, or specialist coordinator."""
        if self.specialist_mode and self.team_mode:
            if not self.specialist_coordinator:
                self.specialist_coordinator = SpecialistCoordinator(fallback_model=self.model)
            return self.specialist_coordinator.run(message)
        elif self.team_mode:
            if not self.coordinator:
                self._rebuild_coordinator()
            return self.coordinator.run(message)
        else:
            agent = self.agents[self.active_agent]
            agent.confirm_fn = self._confirm
            agent.ask_user_fn = self._ask_user
            agent.permission_mode = self.permission_mode
            return agent.chat(message, on_event=self._on_event)

    def run(self, resume: bool = False):
        """Start the interactive REPL."""
        # Readline history
        try:
            import readline
            readline.parse_and_bind("tab: complete")
            histfile = os.path.expanduser("~/.zenith_history")
            try:
                readline.read_history_file(histfile)
            except FileNotFoundError:
                pass
            atexit.register(readline.write_history_file, histfile)
        except ImportError:
            pass

        atexit.register(self._auto_save)

        # Resume latest session if requested
        if resume:
            sessions = list_sessions()
            if sessions:
                data = load_session(sessions[0]["id"])
                agent = self.agents[self.active_agent]
                agent.history = data["history"]
                print(f"  {GREEN}Resumed {len(data['history'])} messages from latest session{RESET}")

        print(BANNER)
        print(f"  {DIM}Working directory: {os.getcwd()}{RESET}")
        print(f"  {DIM}Backend: {self.backend}{RESET}")
        if self.backend == "llamacpp":
            print(f"  {DIM}Thinking: enabled{RESET}")
        else:
            print(f"  {DIM}Model: {self.model}{RESET}")
        effort = self.agents[self.active_agent].effort
        if effort != "medium":
            print(f"  {DIM}Effort: {effort}{RESET}")
        print(f"  {DIM}Active agent: {self.active_agent}{RESET}")
        compact_threshold = self.agents[self.active_agent].max_context_tokens
        if compact_threshold is None:
            compact_threshold = self._compute_compact_threshold()
        if compact_threshold:
            print(f"  {DIM}Compact threshold: {compact_threshold:,} tokens{RESET}")
        print()

        while True:
            try:
                # Wrap ANSI codes in \001/\002 so readline calculates cursor position correctly
                _s, _e = "\001", "\002"
                if not self.team_mode:
                    prompt = f"  {_s}{BLUE}{BOLD}{_e}{self.active_agent}{_s}{RESET}{_e} {_s}{BOLD}{_e}>{_s}{RESET}{_e} "
                else:
                    prompt = f"  {_s}{MAGENTA}{BOLD}{_e}team{_s}{RESET}{_e} {_s}{BOLD}{_e}>{_s}{RESET}{_e} "
                user_input = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n  {DIM}Goodbye!{RESET}\n")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Send to agent
            print()
            start = time.time()

            try:
                response = self._chat(user_input)
                elapsed = time.time() - start

                if not self._streaming_text:
                    # Non-streamed response (coordinator/swarm) — print it
                    print(f"\n  {GREEN}{response}{RESET}")
                else:
                    # Streamed response already printed — just close out
                    print(f"{RESET}")
                    self._streaming_text = False
                print(f"\n  {DIM}({elapsed:.1f}s){RESET}\n")

            except Exception as e:
                print(f"\n  {RED}Error: {e}{RESET}\n")


def main():
    import argparse

    # Load defaults from .zenithrc/zenith.json/ZENITH_* env vars first.
    # CLI flags below override these via argparse defaults.
    config = load_config()

    # Propagate auto_compact_tokens to env var so compact.py picks it up.
    # compact.py reads ZENITH_AUTO_COMPACT_TOKENS at detect_context_limit() time.
    if config["auto_compact_tokens"] is not None:
        os.environ["ZENITH_AUTO_COMPACT_TOKENS"] = str(config["auto_compact_tokens"])

    parser = argparse.ArgumentParser(description="Zenith Code Python Harness")
    parser.add_argument(
        "--model", "-m",
        default=config["model"],
        help=f"Model to use (default: {config['model']})",
    )
    parser.add_argument(
        "--backend", "-b",
        default=config["backend"],
        choices=["ollama", "llamacpp"],
        help="Backend to use (default: auto-detect, prefers llama.cpp)",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=config["ctx_size"],
        help=f"Context size in tokens (default: {config['ctx_size']})",
    )
    parser.add_argument(
        "--cd", "-C",
        default=None,
        help="Working directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest session",
    )
    parser.add_argument(
        "--permission-mode",
        choices=["readonly", "workspace", "full"],
        default=config["permission_mode"],
        help=f"Permission mode (default: {config['permission_mode']})",
    )
    parser.add_argument(
        "--effort", "-e",
        choices=["low", "medium", "max"],
        default=config["effort"],
        help=f"Reasoning effort level (default: {config['effort']})",
    )
    args = parser.parse_args()

    if args.cd:
        os.chdir(os.path.expanduser(args.cd))

    harness = Harness(model=args.model, backend=args.backend, ctx_size=args.ctx_size)

    mode_map = {"readonly": PermissionMode.READ_ONLY, "workspace": PermissionMode.WORKSPACE_WRITE, "full": PermissionMode.FULL_ACCESS}
    harness.permission_mode = mode_map[args.permission_mode]

    if args.effort != "medium":
        for agent in harness.agents.values():
            agent.effort = args.effort

    harness.run(resume=args.resume)


if __name__ == "__main__":
    main()
