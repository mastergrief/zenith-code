"""
Auto-CALM Hypothesis Generation — ranked candidate explanations.

Given observations (symptoms, errors, behaviors), generate structured
hypotheses ranked by plausibility. The core of debugging, diagnosis,
and troubleshooting.

Usage:
    from calm.hypothesis_gen import HypothesisEngine
    he = HypothesisEngine()
    result = he.generate("API returns 500 errors intermittently under load")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Hypothesis:
    """A candidate explanation for observed behavior."""
    description: str
    plausibility: float        # 0-1
    category: str              # "resource", "logic", "config", "dependency", "concurrency", "data"
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    test: str = ""             # how to confirm/deny this hypothesis
    fix: str = ""              # what to do if confirmed


@dataclass
class HypothesisResult:
    """Ranked hypotheses for observed behavior."""
    observation: str
    hypotheses: List[Hypothesis] = field(default_factory=list)

    @property
    def ranked(self) -> List[Hypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.plausibility, reverse=True)

    def summary(self) -> str:
        if not self.hypotheses:
            return "No hypotheses generated"
        lines = [f"{len(self.hypotheses)} hypotheses for: {self.observation[:60]}"]
        for i, h in enumerate(self.ranked[:5], 1):
            lines.append(f"  {i}. [{h.plausibility:.0%}] {h.description}")
            if h.test:
                lines.append(f"     Test: {h.test}")
        return "\n".join(lines)


# Observation → hypothesis templates
# Each pattern triggers a set of candidate hypotheses
_HYPOTHESIS_TEMPLATES = {
    "intermittent_error": {
        "pattern": re.compile(r'\b(?:intermittent|sporadic|random|sometimes|occasionally|flaky|under load)\b.*\b(?:error|fail|crash|timeout|500)\b|\b(?:error|fail|crash|timeout|500)\b.*\b(?:intermittent|sporadic|random|sometimes|occasionally|flaky|under load)\b', re.IGNORECASE),
        "hypotheses": [
            ("Resource exhaustion under load (memory, connections, file descriptors)", 0.8, "resource",
             "Check: monitor memory/connections during load spike", "Add resource limits, connection pooling, or circuit breakers"),
            ("Race condition in concurrent code", 0.7, "concurrency",
             "Check: does it correlate with concurrent request count?", "Add locking, use thread-safe data structures"),
            ("External dependency timeout (database, API, DNS)", 0.7, "dependency",
             "Check: correlate error timestamps with dependency health", "Add timeouts, retries with backoff, circuit breaker"),
            ("Connection pool exhaustion", 0.6, "resource",
             "Check: connection pool metrics, max pool size vs concurrent requests", "Increase pool size, add connection timeout, fix connection leaks"),
            ("GC pauses or memory pressure", 0.4, "resource",
             "Check: GC logs, heap usage during errors", "Tune GC, reduce allocation rate, increase heap"),
        ],
    },
    "slow_query": {
        "pattern": re.compile(r'\b(?:slow|timeout|takes? (?:too )?long|performance)\b.*\b(?:query|database|db|sql)\b', re.IGNORECASE),
        "hypotheses": [
            ("Missing index on filtered/joined columns", 0.9, "config",
             "Check: EXPLAIN ANALYZE on the slow query", "Add appropriate indexes"),
            ("N+1 query pattern (ORM generating too many queries)", 0.7, "logic",
             "Check: query count per request, look for loops with DB calls", "Use eager loading, batch queries"),
            ("Full table scan on large table", 0.7, "data",
             "Check: EXPLAIN shows Seq Scan on large table", "Add index, partition table, or optimize query"),
            ("Lock contention from concurrent writes", 0.5, "concurrency",
             "Check: pg_stat_activity for waiting queries", "Optimize transaction scope, use SKIP LOCKED"),
            ("Outdated statistics causing bad query plan", 0.4, "config",
             "Check: when were stats last updated? Run ANALYZE", "Schedule regular ANALYZE, check autovacuum"),
        ],
    },
    "deployment_failure": {
        "pattern": re.compile(r'\b(?:deploy|release|push)\b.*\b(?:fail|broken|error|crash|down)\b', re.IGNORECASE),
        "hypotheses": [
            ("Missing environment variable or config", 0.8, "config",
             "Check: diff env vars between working and broken environment", "Add missing config, use config validation on startup"),
            ("Incompatible dependency version", 0.7, "dependency",
             "Check: diff package-lock/requirements between versions", "Pin versions, use lockfile"),
            ("Database migration not applied or failed", 0.6, "data",
             "Check: migration status, schema diff", "Run pending migrations, verify migration order"),
            ("Port conflict or resource already in use", 0.5, "resource",
             "Check: is another process using the port?", "Kill conflicting process, use different port"),
            ("Build artifact stale or corrupted", 0.4, "config",
             "Check: rebuild from clean state", "Clean build, verify checksums"),
        ],
    },
    "memory_issue": {
        "pattern": re.compile(r'\b(?:memory|OOM|out of memory|heap|leak|RSS|VRAM)\b.*\b(?:error|issue|problem|grow|increase|high)\b', re.IGNORECASE),
        "hypotheses": [
            ("Memory leak — objects not being freed", 0.8, "logic",
             "Check: RSS growth over time, heap dump for accumulating objects", "Find and fix the leak, add memory monitoring"),
            ("Unbounded cache or buffer", 0.7, "logic",
             "Check: cache size metrics, look for caches without eviction", "Add LRU eviction, set max cache size"),
            ("Large data loaded entirely into memory", 0.6, "data",
             "Check: are you loading entire datasets? Streaming possible?", "Use streaming, pagination, or lazy loading"),
            ("Too many concurrent operations", 0.5, "resource",
             "Check: concurrent request/thread count vs available memory", "Add concurrency limits, backpressure"),
        ],
    },
    "auth_failure": {
        "pattern": re.compile(r'\b(?:auth|login|401|403|permission|access denied|unauthorized|forbidden)\b', re.IGNORECASE),
        "hypotheses": [
            ("Expired or invalid token/session", 0.8, "logic",
             "Check: token expiry time, clock skew between services", "Implement token refresh, check clock sync"),
            ("Missing or wrong permissions/roles", 0.7, "config",
             "Check: user's assigned roles vs required permissions", "Grant correct permissions, audit role assignments"),
            ("CORS misconfiguration (browser requests)", 0.6, "config",
             "Check: browser console for CORS errors, check Access-Control headers", "Configure CORS headers correctly"),
            ("API key rotation — old key still in use", 0.5, "config",
             "Check: when was the key last rotated? Is the new key deployed?", "Update deployed key, implement graceful key rotation"),
        ],
    },
    "generic_error": {
        "pattern": re.compile(r'\b(?:error|bug|broken|doesn.t work|not working|fail|crash)\b', re.IGNORECASE),
        "hypotheses": [
            ("Input validation failure — unexpected data format", 0.6, "data",
             "Check: what input triggers the error? Try with known-good input", "Add input validation, handle edge cases"),
            ("Null/undefined reference", 0.6, "logic",
             "Check: stack trace for null pointer/undefined access", "Add null checks, use Optional types"),
            ("Configuration mismatch between environments", 0.5, "config",
             "Check: diff configs between working and broken environments", "Standardize config, use config-as-code"),
            ("Dependency not installed or wrong version", 0.5, "dependency",
             "Check: are all dependencies installed? Correct versions?", "Reinstall dependencies, check lockfile"),
        ],
    },
}


class HypothesisEngine:
    """Generates ranked hypotheses for observed problems."""

    def generate(self, observation: str) -> HypothesisResult:
        """Generate hypotheses for an observation."""
        result = HypothesisResult(observation=observation)

        # Try each template
        matched = False
        for name, template in _HYPOTHESIS_TEMPLATES.items():
            if template["pattern"].search(observation):
                matched = True
                for desc, plausibility, category, test, fix in template["hypotheses"]:
                    # Adjust plausibility based on observation keywords
                    adj_plausibility = self._adjust_plausibility(
                        plausibility, category, observation
                    )
                    result.hypotheses.append(Hypothesis(
                        description=desc,
                        plausibility=adj_plausibility,
                        category=category,
                        test=test,
                        fix=fix,
                    ))

        # If no specific template matched, use generic
        if not matched:
            for desc, plausibility, category, test, fix in _HYPOTHESIS_TEMPLATES["generic_error"]["hypotheses"]:
                result.hypotheses.append(Hypothesis(
                    description=desc,
                    plausibility=plausibility,
                    category=category,
                    test=test,
                    fix=fix,
                ))

        # Deduplicate by description
        seen = set()
        unique = []
        for h in result.hypotheses:
            if h.description not in seen:
                seen.add(h.description)
                unique.append(h)
        result.hypotheses = unique

        return result

    def _adjust_plausibility(self, base: float, category: str,
                              observation: str) -> float:
        """Adjust plausibility based on observation keywords."""
        obs_lower = observation.lower()

        # Boost if observation mentions the category explicitly
        boosts = {
            "resource": ["memory", "cpu", "disk", "connection", "pool", "limit", "oom"],
            "concurrency": ["thread", "concurrent", "parallel", "race", "lock", "deadlock"],
            "dependency": ["api", "service", "external", "third-party", "upstream", "downstream"],
            "config": ["config", "environment", "variable", "setting", "misconfigured"],
            "data": ["data", "input", "format", "schema", "migration", "corrupt"],
            "logic": ["bug", "logic", "wrong", "incorrect", "unexpected"],
        }

        for keyword in boosts.get(category, []):
            if keyword in obs_lower:
                base = min(1.0, base + 0.1)
                break

        return round(base, 2)
