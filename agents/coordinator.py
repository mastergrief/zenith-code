"""Coordinator agent that delegates tasks across a team of agents."""

from agents.agent import Agent


class Coordinator:
    """Routes tasks to specialized agents and synthesizes results."""

    def __init__(self, agents: list[Agent], model: str = "qwen3.5:4b"):
        self.agents = {a.name: a for a in agents}
        self.leader = Agent(
            name="coordinator",
            role="task coordinator",
            model=model,
            system_prompt=self._build_system_prompt(agents),
        )

    @staticmethod
    def _build_system_prompt(agents: list[Agent]) -> str:
        agent_list = "\n".join(
            f"- {a.name}: {a.role}" for a in agents
        )
        return (
            "You are a coordinator managing a team of specialist agents.\n"
            "Your job is to break down tasks, delegate to the right agent, "
            "and synthesize their outputs into a final answer.\n\n"
            f"Available agents:\n{agent_list}\n\n"
            "When delegating, respond with JSON like:\n"
            '{"delegate": "agent_name", "task": "what to ask them"}\n\n'
            "When you have a final answer, respond with:\n"
            '{"final": "your synthesized answer"}'
        )

    def run(self, task: str, max_rounds: int = 5) -> str:
        """Execute a task using the agent team."""
        response = self.leader.chat(task, think=False)

        for _ in range(max_rounds):
            parsed = self._try_parse(response)

            if parsed and "final" in parsed:
                return parsed["final"]

            if parsed and "delegate" in parsed:
                agent_name = parsed["delegate"]
                sub_task = parsed["task"]

                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    result = agent.chat(sub_task)
                    response = self.leader.chat(
                        f"Agent '{agent_name}' responded:\n{result}",
                        think=False,
                    )
                else:
                    response = self.leader.chat(
                        f"Agent '{agent_name}' not found. "
                        f"Available: {list(self.agents.keys())}",
                        think=False,
                    )
            else:
                return response

        return response

    @staticmethod
    def _try_parse(text: str) -> dict | None:
        """Try to extract JSON from agent response."""
        import re
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return __import__("json").loads(match.group())
            except Exception:
                return None
        return None
