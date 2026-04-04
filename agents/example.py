"""Example: multi-agent system with Qwen 3.5 via Ollama.

Make sure Ollama is running: ollama serve
And the model is pulled: ollama pull qwen3.5:4b
"""

from agents import Agent, Coordinator, Swarm


def demo_coordinator():
    """Demo: coordinator delegates tasks to specialist agents."""
    print("=== Coordinator Demo ===\n")

    coder = Agent("coder", "expert Python developer who writes clean code")
    reviewer = Agent("reviewer", "code reviewer who finds bugs and suggests improvements")
    planner = Agent("planner", "software architect who breaks down complex problems")

    coordinator = Coordinator([coder, reviewer, planner])
    result = coordinator.run("Build a Python function that finds duplicate files by content hash.")
    print(result)


def demo_swarm():
    """Demo: broadcast a question to multiple agents in parallel."""
    print("\n=== Swarm Demo ===\n")

    agents = [
        Agent("optimist", "someone who sees the bright side of everything"),
        Agent("critic", "a skeptical analyst who finds flaws"),
        Agent("pragmatist", "a practical problem solver focused on what works"),
    ]

    swarm = Swarm(agents)
    results = swarm.broadcast("Should we rewrite our monolith into microservices?")

    for name, response in results.items():
        print(f"\n--- {name} ---")
        print(response[:500].encode("utf-8", errors="replace").decode("utf-8"))


if __name__ == "__main__":
    demo_coordinator()
    demo_swarm()
