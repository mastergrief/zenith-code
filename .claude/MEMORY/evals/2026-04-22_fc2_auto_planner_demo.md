# FC2 — Auto-refactor planner demo

Substrate plans + executes a refactor session from a user's
code + tests, zero human direction beyond the baseline harness.

## Input

```python
class Inventory:
    def summary(self, items):
        names = []
        for item in items:
            names.append(item['name'])

        total_value = 0
        for item in items:
            total_value += item['qty'] * item['price']

        over_threshold = []
        for item in items:
            if item['qty'] > 10:
                over_threshold.append(item['name'])

        return {
            'names': names,
            'total_value': total_value,
            'over_threshold': over_threshold,
            'count': len(items),
        }
```

## Opportunities detected

| severity | kind | location | detail |
|---|---|---|---|
| info | loop_to_comprehension | <module> | converted for-loop accumulation into `names = [...]` |
| info | loop_to_comprehension | <module> | converted for-loop accumulation into `over_threshold = [...]` |

## Plan built

| # | primitive | kwargs |
|---|---|---|
| 1 | convert_loop_to_comprehension | {} |

## Execution outcome

**1/1 steps applied & verified, 2 opportunities detected**

- Final tests pass: True
- Applied + verified: 1
- Rolled back: 0
- Refused (no-op/error): 0

## Final code

```python
class Inventory:

    def summary(self, items):
        names = [item['name'] for item in items]
        total_value = 0
        for item in items:
            total_value += item['qty'] * item['price']
        over_threshold = [item['name'] for item in items if item['qty'] > 10]
        return {'names': names, 'total_value': total_value, 'over_threshold': over_threshold, 'count': len(items)}
```

