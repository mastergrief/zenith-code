"""
CALM Docker/container knowledge backend — Dockerfile instructions, compose, best practices.

Models hallucinate Dockerfile syntax, confuse COPY/ADD, mess up multi-stage builds.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_INSTRUCTIONS = {
    "FROM": {"description": "Set base image", "required": True, "position": "first (except ARG)", "example": "FROM python:3.12-slim"},
    "RUN": {"description": "Execute command during build", "creates_layer": True, "example": "RUN apt-get update && apt-get install -y curl", "tip": "Chain with && to reduce layers"},
    "CMD": {"description": "Default command when container starts", "creates_layer": False, "forms": ["exec (preferred): CMD [\"python\", \"app.py\"]", "shell: CMD python app.py"], "overridable": True},
    "ENTRYPOINT": {"description": "Container executable (CMD becomes args)", "creates_layer": False, "forms": ["exec: ENTRYPOINT [\"python\"]"], "overridable": "only with --entrypoint flag"},
    "COPY": {"description": "Copy files from build context to image", "creates_layer": True, "example": "COPY . /app", "preferred_over": "ADD (unless you need tar extraction)"},
    "ADD": {"description": "Copy + auto-extract tar + fetch URLs", "creates_layer": True, "gotcha": "Use COPY unless you specifically need tar extraction or URL fetch"},
    "WORKDIR": {"description": "Set working directory (creates if missing)", "creates_layer": False, "example": "WORKDIR /app"},
    "ENV": {"description": "Set environment variable (persists in image)", "creates_layer": False, "example": "ENV NODE_ENV=production"},
    "ARG": {"description": "Build-time variable (NOT in final image)", "creates_layer": False, "example": "ARG VERSION=latest", "scope": "only during build"},
    "EXPOSE": {"description": "Document which port the container listens on", "creates_layer": False, "note": "Documentation only — does NOT publish the port. Use -p at runtime."},
    "VOLUME": {"description": "Create mount point for external storage", "creates_layer": False, "gotcha": "Changes after VOLUME in Dockerfile are silently lost"},
    "USER": {"description": "Set user for subsequent RUN/CMD/ENTRYPOINT", "creates_layer": False, "security": "Run as non-root in production"},
    "HEALTHCHECK": {"description": "Health check command for container orchestrators", "creates_layer": False, "example": "HEALTHCHECK CMD curl -f http://localhost/ || exit 1"},
    "LABEL": {"description": "Add metadata to image", "creates_layer": False, "example": 'LABEL version="1.0"'},
    "STOPSIGNAL": {"description": "Signal sent to container on docker stop", "default": "SIGTERM"},
    "SHELL": {"description": "Override default shell for RUN", "default": '["sh", "-c"] on Linux'},
}

_COMPOSE_KEYS = {
    "services": "Define containers (replaces 'docker run' commands)",
    "image": "Use pre-built image",
    "build": "Build from Dockerfile (build: . or build: {context: ., dockerfile: Dockerfile})",
    "ports": "Publish ports (host:container)",
    "volumes": "Mount volumes (host_path:container_path or named_volume:path)",
    "environment": "Set env vars (list or map)",
    "env_file": "Load env vars from file (.env)",
    "depends_on": "Start order (not readiness — use healthcheck for that)",
    "networks": "Connect to Docker networks",
    "restart": "Restart policy (no, always, on-failure, unless-stopped)",
    "command": "Override CMD",
    "entrypoint": "Override ENTRYPOINT",
    "healthcheck": "Container health check",
    "deploy": "Swarm/K8s deployment config (replicas, resources, etc.)",
}

_BEST_PRACTICES = {
    "multi-stage builds": "Use multiple FROM stages to keep final image small. Build stage has dev tools, production stage copies only artifacts.",
    "minimize layers": "Chain RUN commands with && to reduce image layers and size.",
    "use .dockerignore": "Exclude .git, node_modules, __pycache__, .env from build context.",
    "pin versions": "Use specific tags (python:3.12-slim), not :latest.",
    "non-root user": "Add USER directive. Don't run as root in production.",
    "cache-friendly ordering": "COPY requirements first, RUN install, then COPY source. Leverages layer cache.",
    "use slim/alpine bases": "python:3.12-slim (~50MB) vs python:3.12 (~350MB). Alpine is smaller but can cause glibc issues.",
    "one process per container": "Don't run multiple services in one container. Use compose for multi-service apps.",
    "COPY over ADD": "ADD auto-extracts and fetches URLs — use COPY for simple file copying.",
    "health checks": "Add HEALTHCHECK for production. Orchestrators (K8s, Swarm) use it for readiness.",
}


def dockerfile_instruction(name: str) -> dict:
    """Get details about a Dockerfile instruction."""
    key = str(name).upper().strip()
    entry = _INSTRUCTIONS.get(key)
    if not entry:
        return {"error": f"Unknown: {name}", "valid": sorted(_INSTRUCTIONS.keys())}
    return {"instruction": key, **entry}


def compose_key(name: str) -> str:
    """Explain a docker-compose.yml key."""
    key = str(name).lower().strip()
    entry = _COMPOSE_KEYS.get(key)
    return entry if entry else f"Unknown key: {name}. Valid: {', '.join(sorted(_COMPOSE_KEYS.keys()))}"


def docker_best_practice(topic: str) -> str:
    """Get Docker best practice by topic."""
    key = str(topic).lower().strip()
    for k, v in _BEST_PRACTICES.items():
        if key in k or k in key:
            return v
    return f"No practice found for: {topic}"


def copy_vs_add() -> dict:
    """Explain the difference between COPY and ADD."""
    return {
        "COPY": "Simple file/directory copy from build context to image. Predictable.",
        "ADD": "Like COPY but also: auto-extracts tar archives, can fetch remote URLs.",
        "recommendation": "Use COPY unless you specifically need tar extraction or URL fetching.",
        "security": "ADD with URLs can introduce supply chain risks.",
    }


def cmd_vs_entrypoint() -> dict:
    """Explain the difference between CMD and ENTRYPOINT."""
    return {
        "CMD": "Default command. Easily overridden by docker run args.",
        "ENTRYPOINT": "Container's main executable. CMD becomes arguments to ENTRYPOINT.",
        "combined": "ENTRYPOINT [\"python\"] + CMD [\"app.py\"] → python app.py. Override: docker run img script.py → python script.py",
        "override": "CMD: docker run img <new-cmd>. ENTRYPOINT: docker run --entrypoint <new> img",
    }


DOCKER_FUNCTIONS = {
    "dockerfile_instruction": dockerfile_instruction,
    "compose_key": compose_key,
    "docker_best_practice": docker_best_practice,
    "copy_vs_add": copy_vs_add,
    "cmd_vs_entrypoint": cmd_vs_entrypoint,
}

DOCKER_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(?:the\s+)?(?:Dockerfile\s+)?(FROM|RUN|CMD|ENTRYPOINT|COPY|ADD|WORKDIR|ENV|ARG|EXPOSE|VOLUME|USER|HEALTHCHECK)', 'dockerfile_instruction("{0}")'),
    (r'(?:difference between|vs)\s+(?:COPY|copy)\s+(?:and|vs)\s+(?:ADD|add)', 'copy_vs_add()'),
    (r'(?:difference between|vs)\s+(?:CMD|cmd)\s+(?:and|vs)\s+(?:ENTRYPOINT|entrypoint)', 'cmd_vs_entrypoint()'),
    (r'(?:Docker|docker)\s+best\s+practice.*?(\w+)', None),
]
