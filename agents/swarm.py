"""Swarm: run multiple agents in parallel on sub-tasks."""

import concurrent.futures
from agents.agent import Agent


class Swarm:
    """Execute tasks across agents concurrently."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def broadcast(self, message: str, max_workers: int = 4) -> dict[str, str]:
        """Send the same message to all agents in parallel."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(agent.chat, message): agent.name
                for agent in self.agents
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = f"Error: {e}"
        return results

    def scatter(
        self, tasks: dict[str, str], max_workers: int = 4
    ) -> dict[str, str]:
        """Send different tasks to specific agents.

        Args:
            tasks: mapping of agent_name -> task message
        """
        agent_map = {a.name: a for a in self.agents}
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for name, task in tasks.items():
                if name in agent_map:
                    futures[pool.submit(agent_map[name].chat, task)] = name
                else:
                    results[name] = f"Error: agent '{name}' not found"

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = f"Error: {e}"

        return results
