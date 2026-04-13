"""
CALM DevOps knowledge backend — CI/CD, deployment strategies, monitoring, SRE.

Models confuse deployment strategies, hallucinate pipeline stages.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_DEPLOYMENT_STRATEGIES = {
    "rolling": {"description": "Replace instances gradually, one at a time", "downtime": False, "rollback": "redeploy old version", "risk": "low", "speed": "medium", "use_when": "stateless services, standard deploys"},
    "blue-green": {"description": "Two identical environments. Switch traffic from blue (old) to green (new)", "downtime": False, "rollback": "switch back to blue (instant)", "risk": "low", "cost": "2x infrastructure during deploy", "use_when": "zero-downtime requirement, fast rollback needed"},
    "canary": {"description": "Route small % of traffic to new version, monitor, then expand", "downtime": False, "rollback": "route all traffic back to old", "risk": "very low", "use_when": "critical services, need to validate in production before full rollout"},
    "recreate": {"description": "Stop all old instances, start all new ones", "downtime": True, "rollback": "redeploy old version", "risk": "medium", "use_when": "dev/staging, stateful apps that can't run two versions"},
    "A/B testing": {"description": "Route users to different versions based on criteria (not just random)", "downtime": False, "use_when": "feature experimentation, not deployment strategy per se"},
    "shadow": {"description": "Mirror production traffic to new version without serving responses", "downtime": False, "use_when": "validate performance/correctness with real traffic, zero user impact"},
    "feature flags": {"description": "Deploy code dark, enable features per-user/group/percentage", "downtime": False, "tools": ["LaunchDarkly", "Unleash", "GrowthBook", "Flagsmith"], "use_when": "decouple deployment from release"},
}

_CI_CD_CONCEPTS = {
    "CI": {"full": "Continuous Integration", "description": "Automatically build and test every commit/PR", "goal": "catch bugs early, keep main branch healthy", "tools": ["GitHub Actions", "GitLab CI", "Jenkins", "CircleCI"]},
    "CD (delivery)": {"full": "Continuous Delivery", "description": "Code is always deployable. Deploy is manual button push.", "goal": "fast, reliable releases on demand"},
    "CD (deployment)": {"full": "Continuous Deployment", "description": "Every passing commit automatically deploys to production.", "goal": "fastest feedback loop, requires high test confidence"},
    "pipeline stages": {"typical": ["lint", "build", "unit test", "integration test", "security scan", "deploy staging", "e2e test", "deploy production"]},
    "artifact": {"description": "Build output (Docker image, binary, package) stored in registry", "registries": ["Docker Hub", "ECR", "GCR", "Artifactory", "npm", "PyPI"]},
    "infrastructure as code": {"description": "Define infrastructure in version-controlled code", "tools": {"declarative": ["Terraform", "CloudFormation", "Pulumi"], "configuration": ["Ansible", "Chef", "Puppet"]}},
    "GitOps": {"description": "Git as single source of truth for infra + app. Changes via PRs.", "tools": ["ArgoCD", "Flux", "Jenkins X"], "principle": "desired state in git, reconciliation loop applies it"},
}

_SRE_CONCEPTS = {
    "SLI": {"full": "Service Level Indicator", "description": "Quantitative measure of service level (e.g. latency p99, error rate)", "examples": ["request latency p99 < 200ms", "availability > 99.9%", "error rate < 0.1%"]},
    "SLO": {"full": "Service Level Objective", "description": "Target value for an SLI (internal goal)", "example": "p99 latency < 200ms over 30-day window"},
    "SLA": {"full": "Service Level Agreement", "description": "Contract with consequences (refunds) if SLO is not met", "note": "SLAs should be less strict than SLOs — SLO is your internal bar"},
    "error budget": {"description": "Amount of unreliability allowed (100% - SLO). Spend it on features.", "example": "99.9% SLO → 0.1% error budget → 43.2 min/month of allowed downtime"},
    "toil": {"description": "Repetitive, manual, automatable work that grows linearly with service", "goal": "keep toil < 50% of SRE time, automate the rest"},
    "blameless postmortem": {"description": "After incidents, focus on systems not people. Document what happened and what to fix.", "sections": ["timeline", "root cause", "impact", "mitigation", "action items"]},
    "chaos engineering": {"description": "Intentionally inject failures to find weaknesses", "tools": ["Chaos Monkey", "Litmus", "Gremlin"], "principle": "controlled experiments in production"},
    "observability": {"pillars": ["metrics (numbers over time)", "logs (events)", "traces (request paths)"], "tools": {"metrics": ["Prometheus", "Datadog", "Grafana"], "logs": ["ELK", "Loki", "Splunk"], "traces": ["Jaeger", "Zipkin", "OpenTelemetry"]}},
    "incident management": {"severity_levels": ["SEV1 (critical, all hands)", "SEV2 (major, team)", "SEV3 (minor, individual)", "SEV4 (cosmetic, backlog)"], "process": ["detect", "triage", "mitigate", "resolve", "postmortem"]},
}

_NINES_TABLE = {
    "99%": {"downtime_year": "3.65 days", "downtime_month": "7.31 hours", "downtime_week": "1.68 hours"},
    "99.9%": {"downtime_year": "8.77 hours", "downtime_month": "43.83 minutes", "downtime_week": "10.08 minutes"},
    "99.95%": {"downtime_year": "4.38 hours", "downtime_month": "21.92 minutes", "downtime_week": "5.04 minutes"},
    "99.99%": {"downtime_year": "52.6 minutes", "downtime_month": "4.38 minutes", "downtime_week": "1.01 minutes"},
    "99.999%": {"downtime_year": "5.26 minutes", "downtime_month": "26.3 seconds", "downtime_week": "6.05 seconds"},
}


def deployment_strategy(name: str) -> dict:
    """Get details about a deployment strategy."""
    key = str(name).lower().strip().replace("-", "-")
    for k, v in _DEPLOYMENT_STRATEGIES.items():
        if key in k.lower() or k.lower() in key:
            return {"strategy": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_DEPLOYMENT_STRATEGIES.keys())}


def cicd_concept(name: str) -> dict:
    """Get details about a CI/CD concept."""
    key = str(name).lower().strip()
    for k, v in _CI_CD_CONCEPTS.items():
        if key in k.lower():
            return {"concept": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_CI_CD_CONCEPTS.keys())}


def sre_concept(name: str) -> dict:
    """Get details about an SRE concept."""
    key = str(name).upper().strip()
    entry = _SRE_CONCEPTS.get(key)
    if not entry:
        for k, v in _SRE_CONCEPTS.items():
            if key.lower() in k.lower() or key.lower() in v.get("full", "").lower():
                return {"concept": k, **v}
        return {"error": f"Unknown: {name}", "valid": list(_SRE_CONCEPTS.keys())}
    return {"concept": key, **entry}


def nines_downtime(availability: str) -> dict:
    """Get allowed downtime for an availability level (e.g. '99.9%')."""
    key = str(availability).strip()
    if not key.endswith('%'):
        key += '%'
    entry = _NINES_TABLE.get(key)
    if not entry:
        return {"error": f"Unknown: {availability}", "valid": list(_NINES_TABLE.keys())}
    return {"availability": key, **entry}


def blue_green_vs_canary() -> dict:
    """Compare blue-green and canary deployments."""
    return {"blue_green": _DEPLOYMENT_STRATEGIES["blue-green"], "canary": _DEPLOYMENT_STRATEGIES["canary"]}


def sli_vs_slo_vs_sla() -> dict:
    """Compare SLI, SLO, SLA."""
    return {k: _SRE_CONCEPTS[k] for k in ["SLI", "SLO", "SLA"]}


def list_deployment_strategies() -> list[str]:
    """List all deployment strategies."""
    return list(_DEPLOYMENT_STRATEGIES.keys())


DEVOPS_FUNCTIONS = {
    "deployment_strategy": deployment_strategy,
    "cicd_concept": cicd_concept,
    "sre_concept": sre_concept,
    "nines_downtime": nines_downtime,
    "blue_green_vs_canary": blue_green_vs_canary,
    "sli_vs_slo_vs_sla": sli_vs_slo_vs_sla,
    "list_deployment_strategies": list_deployment_strategies,
}

DEVOPS_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(rolling|blue.green|canary|recreate|shadow|feature flag)\s+(?:deploy|deployment)', 'deployment_strategy("{0}")'),
    (r'(?:compare|difference|vs)\s+blue.green\s+(?:and|vs)\s+canary', 'blue_green_vs_canary()'),
    (r'(?:what is|explain)\s+(CI|CD|continuous integration|continuous delivery|continuous deployment|GitOps|infrastructure as code)', 'cicd_concept("{0}")'),
    (r'(?:what is|explain)\s+(SLI|SLO|SLA|error budget|toil|postmortem|chaos engineering|observability)', 'sre_concept("{0}")'),
    (r'(?:compare|difference|vs)\s+SLI\s+(?:and|vs)\s+SLO\s+(?:and|vs)\s+SLA', 'sli_vs_slo_vs_sla()'),
    (r'(?:downtime|allowed downtime)\s+(?:for|at)\s+(99(?:\.\d+)?%?)', 'nines_downtime("{0}")'),
]
