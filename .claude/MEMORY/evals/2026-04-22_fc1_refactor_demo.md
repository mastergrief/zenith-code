# FC1 — End-to-end refactor demo

Shipped as part of the 2026-04-22 frontier-coding pivot.
Demonstrates `VerifiedRefactorSession` chaining 6+ rename_variable
operations through sandbox-gated verification.

## Initial code

```python
class Analytics:
    def summarize(self, txs):
        t = 0
        for x in txs:
            t += x['amount']
        n = len(txs)
        a = t / n if n > 0 else 0
        c = {}
        for x in txs:
            k = x['category']
            c[k] = c.get(k, 0) + x['amount']
        high = t > 1000
        return {'total': t, 'avg': a, 'by_cat': c, 'high_spender': high}
```

## Session outcome

**OK: 6/7 steps applied & verified**

| Step | Operation | Applied | Tests pass |
|---|---|---|---|
| 1 | rename_variable({'old': 't', 'new': 'total_amount', 'scope': 'summarize'}) | ✓ | ✓ |
| 2 | rename_variable({'old': 'n', 'new': 'count', 'scope': 'summarize'}) | ✓ | ✓ |
| 3 | rename_variable({'old': 'a', 'new': 'avg_amount', 'scope': 'summarize'}) | ✓ | ✓ |
| 4 | rename_variable({'old': 'c', 'new': 'by_category', 'scope': 'summarize'}) | ✓ | ✓ |
| 5 | rename_variable({'old': 'x', 'new': 'tx', 'scope': 'summarize'}) | ✓ | ✓ |
| 6 | rename_variable({'old': 'k', 'new': 'cat_key', 'scope': 'summarize'}) | ✓ | ✓ |
| 7 | convert_loop_to_comprehension({}) | ✗ | - |

## Final code

```python
class Analytics:

    def summarize(self, txs):
        total_amount = 0
        for tx in txs:
            total_amount += tx['amount']
        count = len(txs)
        avg_amount = total_amount / count if count > 0 else 0
        by_category = {}
        for tx in txs:
            cat_key = tx['category']
            by_category[cat_key] = by_category.get(cat_key, 0) + tx['amount']
        high = total_amount > 1000
        return {'total': total_amount, 'avg': avg_amount, 'by_cat': by_category, 'high_spender': high}
```

