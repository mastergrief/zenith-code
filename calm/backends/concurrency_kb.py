"""
CALM Concurrency knowledge backend — threads, async, locks, patterns.

Models confuse concurrency vs parallelism, mix up lock types, hallucinate guarantees.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_MODELS = {
    "threads": {"type": "shared memory", "description": "OS-level concurrent execution with shared address space", "pros": ["true parallelism on multi-core", "mature tooling"], "cons": ["race conditions", "deadlocks", "GIL in Python"], "sync": ["mutex", "semaphore", "condition variable", "barrier"]},
    "async/await": {"type": "cooperative", "description": "Single-threaded concurrency via event loop. Tasks yield at await points.", "pros": ["no locks needed", "efficient I/O", "simpler mental model"], "cons": ["no CPU parallelism", "function coloring", "can't use blocking calls"], "languages": ["Python asyncio", "JavaScript", "Rust tokio", "C# async", "Kotlin coroutines"]},
    "actors": {"type": "message passing", "description": "Isolated actors communicate via messages. No shared state.", "pros": ["no shared state", "location transparency", "fault isolation"], "cons": ["message ordering", "mailbox overflow", "debugging harder"], "implementations": ["Erlang/OTP", "Akka", "Orleans"]},
    "CSP": {"type": "channels", "description": "Communicating Sequential Processes. Goroutines communicate via channels.", "pros": ["simple model", "no shared state", "composable"], "cons": ["channel management", "can still deadlock"], "implementations": ["Go goroutines", "Clojure core.async"]},
    "STM": {"type": "transactional memory", "description": "Software Transactional Memory. Atomic blocks retry on conflict.", "pros": ["composable", "no deadlocks", "declarative"], "cons": ["retry overhead", "side effects restricted"], "implementations": ["Haskell STM", "Clojure refs"]},
    "fork-join": {"type": "divide and conquer", "description": "Split task into subtasks, execute in parallel, join results.", "pros": ["natural for recursive problems", "work stealing"], "implementations": ["Java ForkJoinPool", "Rayon (Rust)", "Intel TBB"]},
}

_PRIMITIVES = {
    "mutex": {"type": "mutual exclusion", "description": "Only one thread can hold at a time", "variants": ["recursive/reentrant", "try-lock", "timed"], "gotcha": "can deadlock if multiple mutexes acquired in different order"},
    "semaphore": {"type": "counting", "description": "Allow up to N concurrent accesses", "binary": "binary semaphore ≈ mutex (but no ownership)", "use": "rate limiting, resource pools"},
    "rwlock": {"type": "reader-writer lock", "description": "Multiple readers OR one writer", "use": "read-heavy workloads", "starvation": "readers can starve writers (or vice versa depending on policy)"},
    "condition variable": {"type": "signaling", "description": "Thread waits until condition is signaled by another", "pattern": "while (!condition) { cv.wait(lock); }", "gotcha": "always use while loop, never if (spurious wakeups)"},
    "barrier": {"type": "synchronization point", "description": "All threads wait until everyone arrives", "use": "phased computation"},
    "spinlock": {"type": "busy-wait lock", "description": "Thread loops (spins) until lock available", "use": "very short critical sections where context switch is more expensive than spinning", "warning": "wastes CPU — only for kernel/driver/real-time code"},
    "atomic": {"type": "lock-free", "description": "Hardware-level atomic operations (CAS, fetch-add)", "operations": ["compare-and-swap (CAS)", "fetch-and-add", "load/store with ordering"], "use": "counters, flags, lock-free data structures"},
    "channel": {"type": "message passing", "description": "FIFO queue between producers and consumers", "variants": ["unbuffered (sync)", "buffered (async)", "MPSC", "MPMC"], "implementations": ["Go chan", "Rust mpsc", "Python asyncio.Queue"]},
}

_PROBLEMS = {
    "race condition": {"description": "Output depends on timing/ordering of concurrent operations", "fix": "synchronization (locks, atomics, channels)"},
    "deadlock": {"description": "Two+ threads each waiting for a resource held by the other", "conditions": ["mutual exclusion", "hold and wait", "no preemption", "circular wait"], "fix": "lock ordering, try-lock with timeout, lock-free algorithms"},
    "livelock": {"description": "Threads keep changing state in response to each other without making progress", "analogy": "two people in a hallway, both step aside the same direction", "fix": "randomized backoff"},
    "starvation": {"description": "Thread never gets access to shared resource (always preempted)", "fix": "fair scheduling, priority inversion handling"},
    "priority inversion": {"description": "High-priority thread blocked by low-priority thread holding a lock", "fix": "priority inheritance protocol", "famous_bug": "Mars Pathfinder (1997)"},
    "ABA problem": {"description": "CAS sees value A, another thread changes A→B→A, CAS succeeds incorrectly", "fix": "tagged pointers (counter + value), hazard pointers"},
    "thundering herd": {"description": "Many threads wake up for one event, most go back to sleep", "fix": "wake one (EPOLLONESHOT), accept groups"},
    "false sharing": {"description": "Different threads modify variables on the same cache line, causing invalidation", "fix": "padding/alignment to separate cache lines (64 bytes typical)"},
}

_ASYNC_PATTERNS = {
    "promise/future": {"description": "Placeholder for a value that will be available later", "languages": {"JavaScript": "Promise", "Java": "CompletableFuture", "Rust": "Future", "Python": "asyncio.Future"}},
    "callback": {"description": "Function passed to be called when operation completes", "gotcha": "callback hell (nested callbacks)"},
    "observer": {"description": "Push-based: subjects notify observers of changes", "implementations": ["RxJS Observable", "EventEmitter", "pub/sub"]},
    "producer-consumer": {"description": "Producers add to queue, consumers take from queue", "sync": "bounded buffer with semaphores or blocking queue"},
    "map-reduce": {"description": "Map: apply function to each element. Reduce: combine results.", "use": "parallel data processing"},
    "pipeline": {"description": "Chain of stages, each processing and passing to next", "use": "data processing, Unix pipes"},
    "fan-out/fan-in": {"description": "Fan-out: one task spawns many. Fan-in: many results collected into one.", "use": "parallel API calls, scatter-gather"},
    "work stealing": {"description": "Idle threads steal work from busy threads' queues", "use": "dynamic load balancing", "implementations": ["Java ForkJoinPool", "Rayon", "Tokio"]},
}


def concurrency_model(name: str) -> dict:
    """Get details about a concurrency model."""
    key = str(name).lower().strip().replace("-", "/")
    for k, v in _MODELS.items():
        if key in k.lower() or k.lower() in key:
            return {"model": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_MODELS.keys())}


def sync_primitive(name: str) -> dict:
    """Get details about a synchronization primitive."""
    key = str(name).lower().strip()
    for k, v in _PRIMITIVES.items():
        if key in k or k in key:
            return {"primitive": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_PRIMITIVES.keys())}


def concurrency_problem(name: str) -> dict:
    """Get details about a concurrency problem."""
    key = str(name).lower().strip()
    for k, v in _PROBLEMS.items():
        if key in k or k in key:
            return {"problem": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_PROBLEMS.keys())}


def async_pattern(name: str) -> dict:
    """Get details about an async/concurrency pattern."""
    key = str(name).lower().strip()
    for k, v in _ASYNC_PATTERNS.items():
        if key in k or k in key:
            return {"pattern": k, **v}
    return {"error": f"Unknown: {name}", "valid": list(_ASYNC_PATTERNS.keys())}


def concurrency_vs_parallelism() -> dict:
    """Explain the difference between concurrency and parallelism."""
    return {
        "concurrency": "Dealing with multiple things at once (structure). Single core can be concurrent via interleaving.",
        "parallelism": "Doing multiple things at once (execution). Requires multiple cores/processors.",
        "analogy": "Concurrency = one cook switching between dishes. Parallelism = multiple cooks working simultaneously.",
        "both": "Most real systems use both — concurrent structure enables parallel execution.",
    }


def threads_vs_async() -> dict:
    """Compare threads vs async/await."""
    return {"threads": _MODELS["threads"], "async": _MODELS["async/await"]}


CONCURRENCY_FUNCTIONS = {
    "concurrency_model": concurrency_model,
    "sync_primitive": sync_primitive,
    "concurrency_problem": concurrency_problem,
    "async_pattern": async_pattern,
    "concurrency_vs_parallelism": concurrency_vs_parallelism,
    "threads_vs_async": threads_vs_async,
}

CONCURRENCY_NL_PATTERNS = [
    (r'(?:what is|explain)\s+(threads?|async|actors?|CSP|channels?|STM|fork.join)', 'concurrency_model("{0}")'),
    (r'(?:what is|explain)\s+(?:a\s+)?(mutex|semaphore|rwlock|condition variable|barrier|spinlock|atomic|channel)', 'sync_primitive("{0}")'),
    (r'(?:what is|explain)\s+(race condition|deadlock|livelock|starvation|priority inversion|ABA|thundering herd|false sharing)', 'concurrency_problem("{0}")'),
    (r'(?:what is|explain)\s+(promise|future|callback|observer|producer.consumer|map.reduce|pipeline|fan.out|work stealing)', 'async_pattern("{0}")'),
    (r'(?:compare|difference|vs)\s+concurrency\s+(?:and|vs)\s+parallelism', 'concurrency_vs_parallelism()'),
    (r'(?:compare|difference|vs)\s+threads?\s+(?:and|vs)\s+async', 'threads_vs_async()'),
]
