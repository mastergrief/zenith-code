#!/usr/bin/env python3
"""Claw Code Python Harness — multi-agent terminal coding assistant powered by local Qwen 3.5."""

import os
import sys
import json
import time
from pathlib import Path

from agents.agent import Agent, DEFAULT_MODEL
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
  ║     CLAW CODE — Multi-Agent Harness            ║
  ║     Powered by Qwen 3.5 via Ollama             ║
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
    /history       — show session event log
    /save          — save active agent's session
    /sessions      — list saved sessions
    /load <id>     — load a saved session
    /distill status — show available specialist models
    /distill on    — enable specialist routing
    /distill off   — disable specialist routing
    /exit          — quit{RESET}
"""


class Harness:
    """Terminal REPL harness for multi-agent system."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.agents: dict[str, Agent] = {}
        self.active_agent: str = "coder"
        self.team_mode: bool = False
        self.coordinator: Coordinator | None = None
        self.specialist_mode: bool = False
        self.specialist_coordinator: SpecialistCoordinator | None = None
        self.history_log = HistoryLog()
        self._streaming_text = False  # Track whether we're mid-stream

        # Create default agents
        self._spawn("coder", "expert software engineer who writes clean, efficient code")
        self._spawn("reviewer", "code reviewer who finds bugs, security issues, and suggests improvements")
        self._spawn("planner", "software architect who breaks down problems and designs solutions")

    def _spawn(self, name: str, role: str) -> Agent:
        """Create a new agent."""
        agent = Agent(name=name, role=role, model=self.model)
        self.agents[name] = agent
        return agent

    def _rebuild_coordinator(self):
        """Rebuild coordinator with current agent list."""
        self.coordinator = Coordinator(list(self.agents.values()), model=self.model)

    def _on_event(self, event_type: str, data: dict):
        """Handle agent events for display."""
        if event_type == "token":
            # Streaming token — print inline
            if not self._streaming_text:
                print(f"\n  {GREEN}", end="")
                self._streaming_text = True
            print(data["text"], end="", flush=True)
            return

        elif event_type == "response":
            if self._streaming_text:
                # Already streamed — just close the color and newline
                print(f"{RESET}", end="")
                self._streaming_text = False
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
            return agent.chat(message, on_event=self._on_event)

    def run(self):
        """Start the interactive REPL."""
        print(BANNER)
        print(f"  {DIM}Working directory: {os.getcwd()}{RESET}")
        print(f"  {DIM}Model: {self.model}{RESET}")
        print(f"  {DIM}Active agent: {self.active_agent}{RESET}")
        print()

        while True:
            try:
                prompt = f"{BLUE}{BOLD}{self.active_agent}{RESET}" if not self.team_mode else f"{MAGENTA}{BOLD}team{RESET}"
                user_input = input(f"  {prompt} {BOLD}>{RESET} ").strip()
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

    parser = argparse.ArgumentParser(description="Claw Code Python Harness")
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--cd", "-C",
        default=None,
        help="Working directory",
    )
    args = parser.parse_args()

    if args.cd:
        os.chdir(os.path.expanduser(args.cd))

    harness = Harness(model=args.model)
    harness.run()


if __name__ == "__main__":
    main()
