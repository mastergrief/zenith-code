#!/usr/bin/env python3
"""Append hand-written React/frontend and security training examples.

Targets the 4B base eval weak spots: React/frontend and security partial
passes. Each example follows the existing format in coding_reasoning_claude.jsonl:
system + user + assistant(<think>...</think> + answer).

Topics are chosen to avoid duplicating existing coverage (hydration
mismatch, ORDER BY SQLi, basic JWT, re-render perf, session-vs-JWT choice
are already present).

Run once:
    python3 scripts/generate_react_security_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET = Path("agents/distill/data/coding_reasoning_claude.jsonl")

SYSTEM = "You are a helpful assistant"


def example(user: str, think: str, answer: str) -> dict:
    """Build one training example. Normalizes whitespace in think/answer."""
    assistant = f"<think>\n{think.strip()}\n</think>\n\n{answer.strip()}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant},
        ]
    }


EXAMPLES: list[dict] = []


# ── React / frontend ───────────────────────────────────────────────

EXAMPLES.append(example(
    user="""My setInterval callback in a React component always sees the initial count value, not the current one. What's going on?

```jsx
function Timer() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setCount(count + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return <div>{count}</div>;
}
```""",
    think="""This is the classic stale closure problem. The user has `[]` as the effect dependency, so the effect only runs once on mount. The callback passed to setInterval captures `count` at that moment — which is 0. Every tick, it calls `setCount(0 + 1)` = `setCount(1)`, so it flips to 1 and stays.

Three fixes:

1. **Functional updater** — `setCount(c => c + 1)`. The updater gets the current state, so no closure over `count` is needed. Best fix for simple counters.

2. **Ref** — stash the latest count in a ref, read it inside the interval. Needed when you need to *read* current state for more than just the next value.

3. **Re-create the interval** — add `count` to the dependency array. This clears and recreates the interval every tick, which is wasteful and can drift.

The functional updater is the right answer here. I should also mention the ref pattern for when the closure needs more than just an updater.""",
    answer="""Stale closure. Your `useEffect` runs once (empty deps), so the `setInterval` callback captures `count = 0`. Every tick it calls `setCount(0 + 1)`, which is why the count jumps to 1 and freezes there.

**Fix: use the functional updater form of setState**

```jsx
useEffect(() => {
  const id = setInterval(() => setCount(c => c + 1), 1000);
  return () => clearInterval(id);
}, []);
```

`setCount(c => c + 1)` receives the current state from React — no closure needed.

**If you need to *read* current state (not just update it):**

```jsx
const countRef = useRef(count);
useEffect(() => { countRef.current = count; });

useEffect(() => {
  const id = setInterval(() => {
    console.log('current count:', countRef.current);
    setCount(c => c + 1);
  }, 1000);
  return () => clearInterval(id);
}, []);
```

**Don't** add `count` to the dependency array unless you want the interval to reset every tick — it works but burns a timer per render, and causes drift because the interval restarts each time.

Rule of thumb: if your effect callback only needs to *set* state based on prior state, use the updater form. If it needs to *read* current state for branching or logging, use a ref.""",
))

EXAMPLES.append(example(
    user="""I'm confused about when to put \"use client\" at the top of a file in Next.js App Router. Do I always need it for anything interactive?""",
    think="""The user is asking about the RSC (React Server Components) boundary in Next.js 13+ App Router. This is a common point of confusion.

Key facts:
- By default, all components in `app/` are Server Components — they render on the server, can be async, can access the filesystem/db directly, but cannot use hooks or event handlers
- `"use client"` marks a file as a Client Component — it and everything it imports becomes client-side JS
- The directive marks the *boundary*, not every file that uses hooks. Once you're inside a client component, imported components can use hooks without their own directive (unless they're explicitly imported as server components via a Server Component wrapper, which is advanced)

Common mistakes:
1. Adding "use client" to every file "just in case" — defeats the benefit of RSC
2. Thinking you need it for every file that uses any browser API — only needed at the boundary
3. Trying to pass functions from server → client (not serializable)
4. Trying to use hooks in a server component

Rule: put "use client" on the smallest component that actually needs browser APIs (useState, useEffect, onClick, window, localStorage). Everything above it stays on the server.""",
    answer="""**You only need `"use client"` at the *boundary*** — the highest component that uses state, effects, event handlers, or browser APIs. Everything below that boundary is automatically client-side.

**Server Components (default in `app/`):**
- Render on the server, send HTML + serialized props
- Can be `async`, fetch data, read the filesystem, hit a database directly
- **Cannot** use `useState`, `useEffect`, `onClick`, `window`, `localStorage`

**Client Components (`"use client"` directive):**
- Hydrate on the client, can use hooks and event handlers
- Cannot be `async` (they're regular React components)
- Their props must be serializable (no functions, no class instances)

**Practical pattern:**

```tsx
// app/dashboard/page.tsx — Server Component (no directive)
import { getUser } from '@/lib/db';
import InteractiveButton from './InteractiveButton';

export default async function Dashboard() {
  const user = await getUser();  // direct DB access
  return (
    <div>
      <h1>Welcome {user.name}</h1>
      <InteractiveButton userId={user.id} />
    </div>
  );
}
```

```tsx
// app/dashboard/InteractiveButton.tsx — Client Component
"use client";
import { useState } from 'react';

export default function InteractiveButton({ userId }: { userId: string }) {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
```

**Don't sprinkle `"use client"` everywhere** — that turns the whole tree into a traditional SPA and loses the server-render benefit. Keep the boundary as low (deep) in the tree as possible.

**Gotcha**: you can't pass a function from a Server Component to a Client Component (not serializable). If you need a callback, move the state up or use a Server Action.""",
))

EXAMPLES.append(example(
    user="""React is throwing \"Rendered more hooks than during the previous render\". My component sometimes needs to skip a hook when a prop is null.

```jsx
function Profile({ userId }) {
  if (!userId) return null;
  const user = useUser(userId);
  return <div>{user.name}</div>;
}
```""",
    think="""The user is violating the Rules of Hooks — hooks must be called in the same order every render. The early return skips `useUser` on some renders and calls it on others, which breaks React's internal hook ordering.

React identifies hooks by their position in the call order, not by name. Conditional hooks corrupt that position.

The fix: always call the hook, handle the null case inside the hook or after it.

Three approaches:
1. Call hook unconditionally, return null from the render
2. Extract conditional logic into a child component
3. Have the hook itself handle the null case (`useUser(userId)` → returns null when userId is null)

All three are valid. The third is cleanest for reusable hooks — the hook's contract includes handling the missing-input case.""",
    answer="""You're breaking the **Rules of Hooks**: hooks must be called in the same order on every render. Your early return skips `useUser` when `userId` is null, which breaks React's ability to match hooks to their slots.

**Fix 1 — reorder: call hooks first, return later**

```jsx
function Profile({ userId }) {
  const user = useUser(userId);  // always called
  if (!userId || !user) return null;
  return <div>{user.name}</div>;
}
```

Now the hook runs on every render; the null check moves after it.

**Fix 2 — make the hook itself handle null**

If `useUser` is yours, make it accept null:

```jsx
function useUser(userId) {
  const [user, setUser] = useState(null);
  useEffect(() => {
    if (!userId) { setUser(null); return; }
    fetchUser(userId).then(setUser);
  }, [userId]);
  return user;
}
```

Now callers don't need the early return.

**Fix 3 — extract a child component**

```jsx
function Profile({ userId }) {
  if (!userId) return null;
  return <LoadedProfile userId={userId} />;
}

function LoadedProfile({ userId }) {
  const user = useUser(userId);
  return <div>{user?.name}</div>;
}
```

`LoadedProfile` only mounts when `userId` is truthy, so its hook count is consistent.

**Why this rule exists**: React tracks hook state by call order, not by name. If you sometimes call 3 hooks and sometimes 4, React can't tell which slot each value belongs to. The ESLint plugin `react-hooks/rules-of-hooks` catches this at lint time — turn it on.""",
))

EXAMPLES.append(example(
    user="""When should I extract logic into a custom hook vs just leaving it in the component?""",
    think="""Good architectural question. There's a common antipattern where people extract custom hooks for everything, and another where people never extract and end up with 300-line components.

Signals for extraction:
- The same stateful logic is used in 2+ components (reuse)
- The logic has a clear lifecycle (fetch, subscribe, cleanup) that's noisy inline
- You want to test the logic in isolation
- You're describing a concept with a name ("useDebounce", "useOnlineStatus")

Signals against extraction:
- One-off logic used in one place
- Just wraps a single useState or useEffect with no transformation
- The hook name doesn't describe a concept, just "where I put the code"

Return shape matters:
- Return an array `[value, setValue]` when it mirrors useState (two items)
- Return an object `{ data, loading, error }` when there are 3+ named things
- Don't return arrays for 3+ items — order is too easy to forget

I should also mention that custom hooks inherit the Rules of Hooks and shouldn't call hooks conditionally inside themselves.""",
    answer="""Extract when the logic represents a **named, reusable concept**. Leave inline when it's one-off glue code.

**Good reasons to extract:**

1. **Reused across components** — you're copy-pasting a `useEffect` + `useState` block into 3 places.
2. **Named concept** — "debounced value", "online status", "form field", "infinite scroll". If you can name it `useX`, it probably deserves to be a hook.
3. **Noisy lifecycle** — subscription + cleanup, fetch + cancel, focus/blur listeners. Extraction removes the noise from the component.
4. **Testable in isolation** — you want to unit-test the logic without rendering a component.

**Bad reasons:**

1. **"Cleaner code"** — if it's one-off, extracting just adds indirection. The component was honest; the hook hides what's happening.
2. **Wrapping a single built-in** — `function useName() { return useState(''); }` is pure ceremony.
3. **Organizing by file** — "I put all state in a hook file" is premature abstraction.

**Return shape:**

```js
// Two items, mirrors useState — array is fine
const [value, setValue] = useCounter(0);

// Three or more — use an object, order is too fragile
const { data, loading, error } = useFetch('/api/users');
```

**Example — good extraction (debounce):**

```jsx
function useDebounce(value, delay = 300) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

// Caller:
const search = useDebounce(rawSearch, 500);
```

This names a concept, encapsulates a lifecycle, and is reusable. It's worth the extraction.

**Gotcha**: custom hooks must still follow the Rules of Hooks internally — no conditional `useState` calls inside, no hooks after an early return.""",
))

EXAMPLES.append(example(
    user="""I'm using React Context for app state and everything re-renders when any field changes. Is Context just slow?""",
    think="""The user is hitting the classic Context performance issue. Context itself isn't slow — it's that any component consuming a context re-renders when the context value reference changes, even if the specific field they care about is unchanged.

Root causes:
1. Storing the whole state object in one context — any change triggers all consumers
2. Creating a new value object on every parent render (`value={{ a, b, setA, setB }}`)
3. Not memoizing the value

Fixes in order of complexity:
1. Split into multiple contexts by access pattern (read-heavy vs write)
2. Memoize the value with useMemo
3. Use context selectors via `use-context-selector` library
4. Switch to Zustand or Jotai for fine-grained subscription

I should explain the core mechanic and recommend splitting contexts as the default fix, with selectors/Zustand for advanced cases.""",
    answer="""Context isn't inherently slow, but it has **coarse-grained updates**: every consumer re-renders when the context *value reference* changes, regardless of which field they read.

**Root cause (the subtle one):**

```jsx
// 🚫 Bad — new object every render
<AppContext.Provider value={{ user, theme, setUser, setTheme }}>
```

Each render of the provider creates a new `{}`, so every consumer re-renders even if `user` and `theme` are unchanged.

**Fix 1 — memoize the value:**

```jsx
const value = useMemo(
  () => ({ user, theme, setUser, setTheme }),
  [user, theme]
);
return <AppContext.Provider value={value}>...</AppContext.Provider>;
```

This helps, but consumers of `user` still re-render when `theme` changes.

**Fix 2 — split contexts by access pattern:**

```jsx
<UserContext.Provider value={userValue}>
  <ThemeContext.Provider value={themeValue}>
    ...
  </ThemeContext.Provider>
</UserContext.Provider>
```

Now a component that reads `useContext(ThemeContext)` doesn't re-render when `user` changes. This is usually enough and needs no libraries.

**Fix 3 — separate read and write contexts:**

```jsx
<UserStateContext.Provider value={user}>
  <UserDispatchContext.Provider value={setUser}>
    ...
  </UserDispatchContext.Provider>
</UserStateContext.Provider>
```

Components that only *dispatch* (buttons) never re-render on state changes.

**When Context isn't the right tool:**

- Frequently-changing state shared by many components (cursor position, form fields at scale)
- You need fine-grained "only re-render when `state.user.name` changes" selection

For those, reach for **Zustand** (tiny, hooks-based, selector-friendly) or **Jotai** (atomic). Both offer the `useStore(s => s.user.name)` selector pattern that Context can't do natively.

**Rule of thumb**: Context is great for rarely-changing global state (theme, auth user, locale). Bad for frequently-changing shared state. Split contexts before reaching for a library.""",
))

EXAMPLES.append(example(
    user="""My table filter feels laggy when typing in the search box — the whole list re-filters on every keystroke. What's the modern React 18 way to fix this?""",
    think="""React 18 introduced `useTransition` and `useDeferredValue` for exactly this case. Before 18, you'd debounce the search input. After 18, you can mark the expensive update as non-urgent and let React interrupt it when the user types again.

Two primitives:
- `useTransition` — wrap the expensive state update in `startTransition(() => setFiltered(...))`. The urgent update (the input value) still happens immediately.
- `useDeferredValue` — wrap an input value. React gives you a lagging copy that updates when the main thread is idle.

Debouncing still works and is simpler if you control the pipeline. Transitions are better when you can't debounce (e.g. filtering state owned elsewhere) or when you want the UI to stay responsive during an expensive render even at high type rates.

Note: transitions only help if the expensive work is the React render itself. If you're calling a slow function synchronously, you need memoization or a worker, not a transition.""",
    answer="""React 18 has two primitives for this: `useTransition` and `useDeferredValue`. They let you mark an update as **non-urgent**, so React can interrupt a slow render when the user types the next character.

**Option A — `useDeferredValue` (simplest):**

```jsx
function Table({ items }) {
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);

  const filtered = useMemo(
    () => items.filter(i => i.name.includes(deferredSearch)),
    [items, deferredSearch]
  );

  return (
    <>
      <input value={search} onChange={e => setSearch(e.target.value)} />
      <List items={filtered} />
    </>
  );
}
```

The input stays on the fresh `search` value (instant typing feedback). `deferredSearch` lags behind, so `filtered` recomputes only when React has spare time.

**Option B — `useTransition` (explicit start point):**

```jsx
const [isPending, startTransition] = useTransition();
const [search, setSearch] = useState('');
const [filtered, setFiltered] = useState(items);

const handleChange = (e) => {
  setSearch(e.target.value);           // urgent (input updates)
  startTransition(() => {              // non-urgent (filter work)
    setFiltered(items.filter(i => i.name.includes(e.target.value)));
  });
};
```

Use `isPending` to show a subtle "updating..." indicator.

**Important caveat:** transitions only help when the slow work is the **React render itself** (large lists, expensive child components). If you're calling a synchronous 500ms function inside `.filter()`, a transition won't save you — React still blocks while that function runs. For that, use `useMemo` (to avoid recomputing when inputs haven't changed) or move the work to a web worker.

**When to still use debounce:**

- You're calling a **server** API on each keystroke — debounce to throttle requests, then use `useDeferredValue` on top of the response.
- You need a specific delay (e.g. "wait 300ms after typing stops"). Transitions let React decide when to flush; debounce gives you control.

**Don't use both on the same state** — it's confusing and they fight for priority. Pick one based on where the cost is (client render vs server request).""",
))

EXAMPLES.append(example(
    user="""I want global state but don't want to set up Redux. What should I use? I've heard of Zustand, Jotai, Recoil, and the built-in useReducer + Context.""",
    think="""The user is asking for a library recommendation, which is inherently opinionated. Let me frame it by problem shape rather than declaring a winner.

Options:
- `useReducer` + Context — zero deps, fine for small app-wide state that rarely changes, suffers from the Context re-render problem
- Zustand — tiny (~1KB), hooks-based, selector-friendly, no provider needed, imperative-ish API
- Jotai — atomic state, bottom-up composition, TypeScript-friendly, slightly steeper learning curve
- Recoil — was Meta's bet, now semi-abandoned, don't recommend for new projects
- Redux Toolkit — still solid for large apps with strict patterns, overkill for most

For most apps starting today, Zustand is the sweet spot: minimal ceremony, good perf, scales to non-trivial size. Jotai is better if you think in atoms. Redux Toolkit if your team already knows Redux or you need time-travel debugging.

I'll recommend Zustand as the default and briefly note when each alternative wins.""",
    answer="""For new projects, **Zustand is the default recommendation**. Tiny, no provider, selector-based performance, minimal ceremony.

```js
import { create } from 'zustand';

const useStore = create((set) => ({
  count: 0,
  user: null,
  increment: () => set((s) => ({ count: s.count + 1 })),
  setUser: (user) => set({ user }),
}));

// In a component:
const count = useStore((s) => s.count);  // subscribes only to count
const increment = useStore((s) => s.increment);
```

The selector (`s => s.count`) means this component only re-renders when `count` changes — not when `user` does. That's the key win over Context.

**When to pick each:**

| Choice | When it fits |
|---|---|
| **useReducer + Context** | Tiny app, 1-2 rarely-changing values, no deps allowed |
| **Zustand** | Default for new projects, mid-size apps, teams that value simplicity |
| **Jotai** | You think bottom-up in atoms; heavy TypeScript use; complex derived state |
| **Redux Toolkit** | Large app, team already knows Redux, need DevTools time-travel, strict action patterns |
| **Recoil** | Don't — development stalled, ecosystem moved on |

**Don't pick Context if:**
- State changes frequently (cursor, form fields, animation values)
- Many unrelated components subscribe to different slices
- You're chasing re-render issues — Context's coarse updates are the root cause, not a symptom

**Don't pick Redux Toolkit if:**
- You're reaching for it "because it's the standard" without a concrete need
- Your team finds the reducer/action/slice pattern ceremonial
- The app is under ~20 screens — Zustand scales there fine

**Don't pick useReducer + Context if:**
- You already know it won't solve re-render perf and you're hoping otherwise
- The state shape will grow beyond a few fields

**Migration note**: all of these can coexist. You can start with Zustand and introduce a separate store for a specific subsystem (a drawing canvas, a real-time editor) without touching the rest. Don't try to unify every library decision across the whole app.""",
))

EXAMPLES.append(example(
    user="""What's the best way to do form validation in React? I've used plain useState with manual checks and it's getting messy.""",
    think="""The user is asking about form validation. Modern answer for most projects is react-hook-form + zod. It's worth explaining why rather than just saying "use this library".

Three approaches:
1. Plain `useState` per field + manual validation — fine for 1-2 simple forms, gets painful fast
2. `react-hook-form` + schema validation (zod / yup) — current standard, low re-renders, great DX
3. Formik — mature, still used, but heavier and more re-renders than react-hook-form

react-hook-form wins because:
- Uncontrolled inputs by default → inputs don't re-render the form on every keystroke
- Schema integration → one source of truth for types + validation
- Small bundle
- Works with MUI, Radix, shadcn, etc

I should show a concrete example that demonstrates the pattern, not just a feature list.""",
    answer="""For anything beyond a 2-field form, use **`react-hook-form` + `zod`**. It's low-ceremony, low-re-render, and gives you a single source of truth for both types and validation.

**Install:**

```bash
npm i react-hook-form zod @hookform/resolvers
```

**Pattern:**

```tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'At least 8 characters'),
  age: z.coerce.number().int().min(18, 'Must be 18+'),
});

type FormValues = z.infer<typeof schema>;

export function SignupForm() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  const onSubmit = async (data: FormValues) => {
    await api.signup(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('email')} />
      {errors.email && <p>{errors.email.message}</p>}

      <input type="password" {...register('password')} />
      {errors.password && <p>{errors.password.message}</p>}

      <input type="number" {...register('age')} />
      {errors.age && <p>{errors.age.message}</p>}

      <button type="submit" disabled={isSubmitting}>Sign up</button>
    </form>
  );
}
```

**Why this beats plain `useState`:**

1. **Uncontrolled inputs** — typing in a field doesn't re-render the whole form, only the field.
2. **One schema** — `z.infer<typeof schema>` gives you TypeScript types for free, so you can't drift the validation from the type.
3. **Async validation** — pass an async function to the resolver (check username availability on the server).
4. **Arrays / nested fields** — `useFieldArray` handles lists and nesting cleanly.

**Plain `useState` is fine when:**

- Form has 1-2 fields (login, search)
- No validation beyond "required"
- You're not going to grow the form

**Don't use Formik for new projects** — react-hook-form has won on ergonomics and perf. Existing Formik code is fine, don't migrate without a reason.

**Server errors**: map server-side validation errors back to fields with `setError('email', { message: '...' })`. Don't duplicate server validation in the client schema — use the same schema on both sides (export from a shared package).""",
))

EXAMPLES.append(example(
    user="""My React Error Boundary doesn't catch errors thrown inside a setTimeout or a fetch .then() callback. Why?""",
    think="""Good question. Error boundaries only catch errors that bubble up through React's rendering / lifecycle / effects. They don't catch errors in:
- Async callbacks (setTimeout, setInterval)
- Event handlers (onClick, etc)
- Promise rejections
- Server-side rendering (well, they do in newer versions, but historically no)
- The error boundary itself

React's reasoning: these errors happen outside the render tree, so React has no way to associate them with a component. The fix is to catch them manually and either set state (which triggers a render that can throw during render) or call a global error handler.

Modern approaches:
1. Try/catch in async code, setState with error → re-throws during render → error boundary catches it
2. react-error-boundary library's useErrorBoundary hook — explicitly pushes errors to the nearest boundary
3. Global unhandled promise rejection handler

Let me give them the cleanest pattern.""",
    answer="""Error boundaries catch errors in **rendering, lifecycle methods, and constructors** of components below them in the tree. They do **not** catch:

- Event handlers (onClick, onChange)
- Async code (setTimeout, promises, fetch callbacks)
- Errors thrown inside the error boundary itself
- Server-side rendering errors (in legacy React; React 18+ handles this)

React's reason: async errors happen outside React's control flow, so it has no way to match them to a component.

**Pattern 1 — re-throw during render (plain React):**

```jsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);

  if (error) throw error;  // propagates to nearest ErrorBoundary

  useEffect(() => {
    fetchUser(userId)
      .then(setUser)
      .catch(setError);
  }, [userId]);

  return user ? <Profile data={user} /> : <Spinner />;
}
```

Setting state with the error triggers a re-render; `throw error` then fires during render, which the boundary *can* catch.

**Pattern 2 — `react-error-boundary` library:**

```jsx
import { useErrorBoundary } from 'react-error-boundary';

function UserProfile({ userId }) {
  const { showBoundary } = useErrorBoundary();
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser(userId).then(setUser).catch(showBoundary);
  }, [userId]);

  return user ? <Profile data={user} /> : <Spinner />;
}
```

`showBoundary(err)` explicitly pushes the error to the nearest boundary. This is cleaner than the re-throw trick.

**Also set up a global unhandled-rejection handler:**

```js
window.addEventListener('unhandledrejection', (event) => {
  logToSentry(event.reason);
  event.preventDefault();  // prevents console noise in prod
});
```

This catches promises that nothing handled — not a substitute for per-request handling, but a safety net.

**Don't:**

- Rely on error boundaries to catch API failures — wrap fetches in try/catch or `.catch()` and convert to state.
- Use a single top-level error boundary for everything. Nest them: one around the page, one around each major section. A failed widget shouldn't blank the page.
- Forget that event handlers need their own try/catch — onClick errors don't bubble to boundaries either.""",
))

EXAMPLES.append(example(
    user="""I have a list of items that re-orders based on user drag-and-drop, and I'm using the array index as the key. React is behaving weirdly — sometimes items get the wrong state. What's happening?""",
    think="""Classic key prop misuse. Using index as key means React identifies each item by its position, not its identity. When the array reorders:
- Item A was at index 0, now at index 1
- React sees "key=0 still has item at index 0" (which is now B, was A)
- React reuses A's component/state/refs for B

So internal component state (a local useState, an input value, a ref) gets attached to the wrong item after reorder.

The fix is to use a stable unique ID from the data. If the data doesn't have one, generate one client-side on load and keep it with the item.

Index as key is OK when:
- The list is static (never reorders or removes)
- Items are stateless (pure presentation)

Otherwise, always use a real ID.""",
    answer="""You're using **`index` as the key**, which tells React "item at position N is the same thing across renders". When you reorder, the component that was at index 0 is now at index 1 — but React thinks the component at index 0 is still the same, so it reuses its internal state (and any mounted refs, form input values, animations) for the *new* item at that position.

**Fix: use a stable unique ID from the data.**

```jsx
// 🚫 Bad
{items.map((item, i) => <Row key={i} item={item} />)}

// ✅ Good
{items.map((item) => <Row key={item.id} item={item} />)}
```

Now React identifies each row by its data identity. When items reorder, React moves the existing DOM node (and its state) along with the item, and nothing gets misattributed.

**If your data doesn't have an ID:**

Generate one when the data arrives and keep it with the item:

```js
const [items, setItems] = useState(() =>
  rawItems.map(item => ({ ...item, _key: crypto.randomUUID() }))
);
```

Or use a library like `nanoid`. Don't generate the ID inside the `.map()` call — that creates a new key every render, which is worse than index.

**When index *is* OK:**

- The list is static (never reordered, never has items added/removed from the middle)
- The items are stateless (just presentation, no `useState`, no form inputs, no refs inside)

For a dynamic draggable list, index is basically always wrong.

**Symptom checklist — if you see any of these, suspect a key problem:**

- Form inputs show the wrong value after reorder
- Animations play on the wrong element
- Focus jumps to an unexpected element
- "Flashing" state when deleting from the middle of a list
- `useEffect` fires for the wrong item

All of these come from React reusing component instances across what you *thought* were different items.""",
))


# ── Security ────────────────────────────────────────────────────────

EXAMPLES.append(example(
    user="""Our app fetches a URL provided by the user (it's a \"preview this link\" feature). Is there a security risk here?""",
    think="""This is SSRF (Server-Side Request Forgery). Classic and often missed. When your server fetches a user-provided URL, the attacker can point it at:

1. Internal IP addresses (169.254.169.254 — AWS metadata, localhost, RFC1918 ranges)
2. Internal services not exposed to the internet (admin panels, databases, Redis)
3. Cloud metadata endpoints (AWS, GCP, Azure) to steal instance credentials
4. File:// URIs to read local files (depending on HTTP client)
5. Protocols like gopher://, dict://, etc (old libcurl)

Capital One's 2019 breach was SSRF against AWS metadata.

Defenses:
1. Allow-list of domains (best if feasible)
2. Deny-list of private IP ranges (fragile but required as a baseline)
3. Resolve DNS yourself and reject private IPs BEFORE the fetch (prevents DNS rebinding)
4. Use a forward proxy that enforces allow-list
5. Disable redirects or re-validate each redirect target
6. Set a short timeout

The DNS rebinding angle is critical — if you check the hostname but let the HTTP client resolve it, an attacker can serve a public IP on first resolve and internal IP on second resolve.

I should cover these clearly and give a concrete safe-fetch helper.""",
    answer="""Yes — this is **SSRF (Server-Side Request Forgery)**, one of the easiest vulnerabilities to miss. An attacker provides a URL that points at something *your server* can reach but they can't directly:

- **Cloud metadata** — `http://169.254.169.254/latest/meta-data/iam/` steals IAM credentials (this is how the Capital One breach happened).
- **Internal services** — `http://localhost:6379/` (Redis), `http://admin.internal/`, `http://10.0.0.5/` (DBs, admin panels).
- **File URIs** — `file:///etc/passwd` if the HTTP client follows them.
- **Port scanning** — `http://internal-host:22/` to enumerate your network.

**Defense — layered:**

```python
import ipaddress
import socket
from urllib.parse import urlparse
import httpx

ALLOWED_SCHEMES = {"http", "https"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB
TIMEOUT = 5.0

def _is_private(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )

def safe_fetch(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"scheme not allowed: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("missing hostname")

    # Resolve DNS ourselves and reject private IPs BEFORE fetching.
    # Prevents DNS rebinding (attacker serves public IP first, private second).
    infos = socket.getaddrinfo(parsed.hostname, None)
    ips = {info[4][0] for info in infos}
    for ip in ips:
        if _is_private(ip):
            raise ValueError(f"blocked private address: {ip}")

    # Pin the connection to the resolved IP to prevent rebinding mid-request.
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    with httpx.Client(
        timeout=TIMEOUT,
        follow_redirects=False,  # re-validate manually if needed
        transport=transport,
    ) as client:
        r = client.get(url)
        if int(r.headers.get("content-length", 0)) > MAX_SIZE:
            raise ValueError("response too large")
        return r.content[:MAX_SIZE]
```

**Key points:**

1. **Scheme allow-list** — only `http`/`https`. No `file`, no `gopher`.
2. **DNS resolve + check BEFORE fetch** — otherwise DNS rebinding bypasses a hostname-only check.
3. **Block private IP ranges** — 10/8, 172.16/12, 192.168/16, 127/8, 169.254/16, ::1, fc00::/7.
4. **Disable redirects** or re-validate each hop — attacker can redirect a public URL to an internal one.
5. **Timeout + size cap** — prevent resource exhaustion.
6. **Best fix if feasible**: allow-list specific domains (`*.trusted-cdn.com`). Deny-lists are always incomplete.

**Architectural fix**: route outbound fetches through a dedicated proxy (or a separate network namespace / VPC) that has no route to internal subnets. Then SSRF can't reach anything valuable even if you miss a check.

**Don't** rely on blocking just `localhost` or `127.0.0.1` — that misses IPv6 `::1`, `0.0.0.0`, `[::ffff:127.0.0.1]`, decimal-encoded IPs (`2130706433`), and DNS entries that resolve to private ranges.""",
))

EXAMPLES.append(example(
    user="""I need to render user-submitted markdown in my React app. I've heard about `dangerouslySetInnerHTML` but is there a safe way to do it?""",
    think="""User-submitted markdown → HTML is a common feature (comments, wiki, docs) and a classic XSS vector. The attack: user submits `[click me](javascript:alert(1))` or `<img src=x onerror=alert(1)>` or `<iframe>`.

Two safe approaches:
1. Render markdown with a library that escapes by default (e.g. `react-markdown` — renders to React elements, no HTML string, no dangerouslySetInnerHTML needed)
2. Render to HTML string, then sanitize with DOMPurify before dangerouslySetInnerHTML

Option 1 is much safer — you never have a raw HTML string, so there's no escape to miss. It also handles edge cases like embedded HTML in markdown (which is stripped by default).

If you must use HTML string (server-rendered, existing pipeline), DOMPurify is the standard. Configure its allow-list carefully.

Don't: trust the markdown parser alone (some allow HTML passthrough), write your own sanitizer, blacklist `<script>` only.""",
    answer="""**Best answer: use `react-markdown`. Don't build an HTML string at all.**

```jsx
import ReactMarkdown from 'react-markdown';

function Comment({ body }) {
  return <ReactMarkdown>{body}</ReactMarkdown>;
}
```

`react-markdown` parses markdown directly into React elements. There's no intermediate HTML string, so there's nothing for XSS to escape from. Raw HTML embedded in the markdown is ignored by default.

**If you want to allow SOME HTML (e.g. `<kbd>`, `<sub>`):**

```jsx
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';

<ReactMarkdown rehypePlugins={[rehypeSanitize]}>
  {body}
</ReactMarkdown>
```

`rehype-sanitize` uses a conservative allow-list (GitHub's schema by default). Customize carefully.

**If you already have an HTML string (legacy pipeline, server-rendered markdown):**

Use **DOMPurify** before `dangerouslySetInnerHTML`:

```jsx
import DOMPurify from 'dompurify';

function Comment({ html }) {
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['p', 'a', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'blockquote'],
    ALLOWED_ATTR: ['href', 'title'],
    ALLOWED_URI_REGEXP: /^(?:https?|mailto):/i,  // no javascript:
  });
  return <div dangerouslySetInnerHTML={{ __html: clean }} />;
}
```

**Key points:**

1. **Allow-list, never deny-list.** New attack vectors appear constantly. Your allow-list should be "what I explicitly need", not "everything except what I've heard of".
2. **Restrict `href` schemes.** `javascript:`, `data:text/html`, `vbscript:` are all XSS vectors. The `ALLOWED_URI_REGEXP` above blocks them.
3. **Strip `on*` attributes.** DOMPurify does this by default; don't disable it.
4. **Sanitize on render, not on input.** If you sanitize on input, a DOMPurify bug fix doesn't retroactively clean old entries. Store the raw input, sanitize on display.

**Don't:**

- Use a regex to "strip `<script>` tags". There are dozens of bypasses (`<ScRiPt>`, `<script >`, embedded in `<img onerror>`, SVG with `<script>`, etc).
- Trust the markdown parser alone unless it explicitly documents XSS safety with its default config (some parsers pass HTML through by default).
- Write your own sanitizer.
- Store HTML and sanitize on read *and* write — one of them is load-bearing; keep it in one place.

**Bonus: Content Security Policy**

Set a CSP header that forbids inline scripts: `Content-Security-Policy: script-src 'self'`. This turns stored XSS into a non-issue even if a sanitizer bug slips through — inline `<script>` won't execute.""",
))

EXAMPLES.append(example(
    user="""Our API has endpoints like `GET /api/orders/:id`. Any logged-in user can fetch any order by guessing the ID. Is this a real problem and how do I fix it?""",
    think="""This is IDOR (Insecure Direct Object Reference), which is the poster child for authorization bugs. The user is correctly identifying the issue: authentication (are you logged in?) is not authorization (can you access THIS resource?).

The fix is to always scope queries by the authenticated user. Three approaches:
1. Add WHERE clauses to every query (scoped repository pattern)
2. Row-level security in the database (Postgres RLS)
3. Middleware that resolves and checks ownership before the handler runs

I'll show the scoped-query approach as the primary fix because it's framework-agnostic, with a note about RLS for Postgres.

Key principles:
- Never use IDs from the URL directly without an authorization check
- "Is the user logged in" is not enough — must check "does this user own this resource"
- UUID instead of sequential IDs is defense in depth, not a fix — IDs are meant to be public

I should also mention that this is OWASP Broken Access Control #1 for a reason.""",
    answer="""Yes, this is a real problem — it's called **IDOR** (Insecure Direct Object Reference) and it's the #1 issue on the OWASP Top 10 (Broken Access Control).

**The bug**: authentication ("logged in") is not authorization ("allowed to access this specific thing"). Any logged-in user can iterate order IDs and dump everyone's data.

**Fix: scope every query by the authenticated user.**

```js
// 🚫 Bad
app.get('/api/orders/:id', requireAuth, async (req, res) => {
  const order = await db.orders.findById(req.params.id);
  if (!order) return res.status(404).end();
  res.json(order);
});

// ✅ Good — WHERE user_id = current user
app.get('/api/orders/:id', requireAuth, async (req, res) => {
  const order = await db.orders.findOne({
    where: { id: req.params.id, userId: req.user.id },
  });
  if (!order) return res.status(404).end();  // 404, not 403 — don't leak existence
  res.json(order);
});
```

The `userId: req.user.id` clause turns "find this order" into "find this order **that belongs to me**". If the order exists but belongs to someone else, the query returns nothing and you 404.

**Why return 404 and not 403?** 403 leaks the fact that the ID exists. An attacker can enumerate IDs and map out which are real. 404 says "no such order, for you" regardless of whether it exists.

**For admin endpoints**, add a role check:

```js
if (!req.user.isAdmin && order.userId !== req.user.id) {
  return res.status(404).end();
}
```

**Better: use a "scoped repository" pattern** so you can't accidentally forget:

```js
class OrderRepository {
  constructor(userId) { this.userId = userId; }
  findById(id) {
    return db.orders.findOne({ where: { id, userId: this.userId } });
  }
  list() {
    return db.orders.findAll({ where: { userId: this.userId } });
  }
}

// In the route:
const orders = new OrderRepository(req.user.id);
const order = await orders.findById(req.params.id);
```

Now every query through this repo is scoped. The ownership check can't be skipped because there's no unscoped query.

**Better still: Postgres Row-Level Security** (if you control the DB):

```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY orders_user_isolation ON orders
  USING (user_id = current_setting('app.current_user_id')::uuid);
```

Set `app.current_user_id` per request. The database now enforces ownership regardless of what the application code does. This is the strongest defense — an ORM bug or a missed WHERE can't bypass it.

**Don't:**

- Use sequential IDs as your only defense. UUIDs make enumeration slower but don't fix authorization. The fix is always the ownership check.
- Do the check in the frontend. The frontend is untrusted; the server must verify.
- Skip checks for "internal" endpoints. Anything with a URL is reachable if you know or guess it.
- Return 403 when you mean "you don't own this" — 404 is safer.

**Audit**: grep your routes for `findById`, `getById`, etc and check each one scopes by user. This is tedious but catches real bugs.""",
))

EXAMPLES.append(example(
    user="""How should I rate limit an API? I've seen IP-based, user-based, and per-endpoint. Which do I need and how do I implement it?""",
    think="""Rate limiting is multi-layered. The naive answer (per-IP limit) is insufficient in modern networks where many users share an IP (corporate NAT, mobile carriers) and attackers can rotate IPs cheaply.

Layers:
1. Per-IP — blunt instrument, catches naive abuse, but unfair to NATed users
2. Per-user (authenticated) — fair, specific, but only works post-login
3. Per-endpoint global — protects expensive endpoints from Thundering herd
4. Per-user per-endpoint — most granular

Algorithms:
- Fixed window — simple, suffers from burst at boundaries
- Sliding window — smoother, more complex
- Token bucket — allows bursts up to N, refills at rate R, great for APIs
- Leaky bucket — strict smoothing

Implementation: Redis is the usual backend. Libraries like `express-rate-limit` + `rate-limit-redis` handle most cases. For Python, `slowapi` (FastAPI) or `django-ratelimit`.

Key considerations:
- Trust proxy headers only if you trust the proxy (X-Forwarded-For)
- Return 429 with Retry-After header
- Don't rate limit behind a login form with the same bucket as authenticated traffic — attacker can lock out a user
- Differentiate between "abusive" (block) and "heavy but legitimate" (throttle)

Let me give them a practical layered approach with code.""",
    answer="""You need **multiple layers**, because each one catches a different attack shape.

**Layer 1 — Global per-IP (blunt, first line)**

Blocks naive attacks and broken clients. Use Redis-backed token buckets:

```js
// Express + express-rate-limit + rate-limit-redis
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';

const globalLimit = rateLimit({
  store: new RedisStore({ sendCommand: (...args) => redisClient.call(...args) }),
  windowMs: 60_000,       // 1 minute
  max: 300,               // 300 requests per IP
  standardHeaders: true,  // RateLimit-* headers
  legacyHeaders: false,
});

app.use(globalLimit);
```

**Layer 2 — Per-user (fair, for authenticated endpoints)**

```js
const userLimit = rateLimit({
  keyGenerator: (req) => req.user?.id || req.ip,
  windowMs: 60_000,
  max: 1000,  // more generous because it's per-user
});

app.use('/api', authenticate, userLimit);
```

**Layer 3 — Expensive endpoints (strict, surgical)**

```js
const expensiveLimit = rateLimit({
  keyGenerator: (req) => req.user?.id || req.ip,
  windowMs: 60_000,
  max: 10,
});

app.post('/api/export', authenticate, expensiveLimit, exportHandler);
app.post('/api/search', authenticate, expensiveLimit, searchHandler);
```

**Layer 4 — Login endpoint (prevents brute force + prevents lockout abuse)**

This is tricky. Rate limit by username *or* IP, whichever is lower, but fall back to IP-only after a threshold so an attacker can't lock out a specific user by spamming their username:

```js
const loginLimit = rateLimit({
  keyGenerator: (req) => `${req.body.username}:${req.ip}`,
  windowMs: 15 * 60_000,
  max: 5,
});
```

Also: add a **per-account** counter in your own code that locks an account after N failures across all IPs (with a self-service unlock). That's defense in depth.

**Algorithms — pick based on traffic shape:**

| Algorithm | Best for | Downside |
|---|---|---|
| Fixed window | Simple, cheap | Boundary bursts (double limit at the boundary) |
| Sliding window | Smooth, fair | More Redis ops |
| **Token bucket** | APIs with bursty clients | — |
| Leaky bucket | Strict smoothing | Can feel sluggish |

Token bucket is the default for APIs. It allows short bursts up to the bucket size, refilling at a steady rate.

**Trusting X-Forwarded-For:**

```js
// Only if you're behind a trusted proxy
app.set('trust proxy', 1);  // trust first hop
```

Trusting `X-Forwarded-For` when your app is directly internet-exposed lets attackers spoof their IP and bypass all limits. Set the number of trusted hops precisely.

**Response headers:**

Always return on 429:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
RateLimit-Limit: 300
RateLimit-Remaining: 0
RateLimit-Reset: 1712505600
```

Clients can back off intelligently.

**Don't:**

- Rate limit in application memory. It resets on restart and doesn't share across instances.
- Use only per-IP. Corporate NAT puts thousands of users behind one IP; attackers rotate IPs cheaply.
- Use only per-user. Pre-login traffic (signup, login, password reset) needs IP-based limits.
- Forget about failed login lockout. Rate limiting slows brute force; lockout with alerts stops it.
- Return 200 OK on limited requests ("silently drop"). Clients can't back off if they don't know.""",
))

EXAMPLES.append(example(
    user="""How should I implement password reset via email? I want to send a link with a token.""",
    think="""Password reset is deceptively hard. Common mistakes:
1. Predictable tokens (sequential, short, derived from user ID)
2. No expiration
3. Tokens that can be reused
4. Leaking account existence ("no user with that email" vs "email sent")
5. Reset tokens in URL query strings that end up in logs/referrers
6. Session not invalidated on password change
7. Old sessions survive reset

The right flow:
1. User submits email → server always returns "if this email is registered, we sent a link" (no enumeration)
2. If user exists: generate cryptographically random token (256 bits), hash it, store hash + user_id + expiration (15-60 min)
3. Email the plain token in the URL
4. On click: look up token hash, check expiration, single-use (delete on use), verify not already consumed
5. On successful reset: update password hash, delete all sessions for that user, delete all other reset tokens

Security:
- Store hash of token, not plain token (same reason as passwords — limit DB dump damage)
- Rate limit reset requests by email AND IP
- Send notification email to the old address after reset ("if this wasn't you, click here")
- Require re-login after reset (invalidate all sessions)

I'll write the flow as pseudocode + a sample implementation.""",
    answer="""Password reset is one of the most-attacked flows. Get every step right.

**The flow:**

```
1. User submits email on "forgot password" form
2. Server ALWAYS responds "If that email is registered, we sent a link."
   (Don't confirm whether the email exists — that's account enumeration.)
3. If the user exists:
   a. Generate 256-bit random token: token = crypto.randomBytes(32).toString('hex')
   b. Store: { user_id, token_hash: sha256(token), expires_at: now + 30min, used: false }
   c. Email: https://app.com/reset?token=<plain_token>
4. User clicks link → POST /reset with token + new password
5. Server:
   a. Look up by token_hash = sha256(request.token)
   b. Check: not used, not expired, matches a user
   c. Update user's password_hash
   d. Mark token as used
   e. Delete ALL other reset tokens for this user
   f. INVALIDATE ALL SESSIONS for this user (log them out everywhere)
   g. Send notification email to the old address: "Your password was changed"
6. User logs in with new password
```

**Implementation (Node + Postgres, adapt as needed):**

```js
import crypto from 'node:crypto';
import argon2 from 'argon2';

// Step 1-3: request reset
app.post('/api/auth/reset-request', rateLimit, async (req, res) => {
  const { email } = req.body;
  const user = await db.users.findByEmail(email);

  // Always respond the same, whether user exists or not
  res.json({ message: 'If that email is registered, we sent a link.' });

  if (!user) return;  // intentionally silent

  const token = crypto.randomBytes(32).toString('hex');
  const tokenHash = crypto.createHash('sha256').update(token).digest('hex');

  await db.passwordResets.create({
    user_id: user.id,
    token_hash: tokenHash,
    expires_at: new Date(Date.now() + 30 * 60 * 1000),
    used: false,
  });

  await sendEmail(user.email, `https://app.com/reset?token=${token}`);
});

// Step 4-6: perform reset
app.post('/api/auth/reset', async (req, res) => {
  const { token, newPassword } = req.body;
  const tokenHash = crypto.createHash('sha256').update(token).digest('hex');

  const record = await db.passwordResets.findOne({
    where: { token_hash: tokenHash, used: false },
  });

  if (!record || record.expires_at < new Date()) {
    return res.status(400).json({ error: 'Invalid or expired token' });
  }

  // Update password (use a real KDF — argon2 or bcrypt)
  const newHash = await argon2.hash(newPassword);
  await db.users.update(record.user_id, { password_hash: newHash });

  // Consume this token + all other pending resets for the user
  await db.passwordResets.update(record.id, { used: true });
  await db.passwordResets.deleteWhere({ user_id: record.user_id, used: false });

  // Invalidate all sessions — force re-login everywhere
  await db.sessions.deleteWhere({ user_id: record.user_id });

  // Notify the user of the change
  await sendEmail(record.user_email, 'Your password was changed. If this wasn\\'t you, contact support.');

  res.json({ ok: true });
});
```

**The critical details most implementations get wrong:**

1. **No enumeration** — the reset-request endpoint gives the same response whether the email exists or not. Same status, same timing (consider adding artificial delay on the "not found" branch so timing doesn't leak).

2. **Store the hash of the token, not the token itself** — if your DB leaks, attackers can't use the stolen tokens to reset passwords. Same reason you don't store plaintext passwords.

3. **Single-use** — mark `used: true` on successful reset. A reused token is an attack signal (forwarded email, attacker intercepted it).

4. **Invalidate all sessions on reset** — if the reset was triggered because the user's account was compromised, existing attacker sessions must die. This is non-negotiable.

5. **Rate limit the reset-request endpoint** — by email and by IP. Attackers use it to spam users or probe for valid emails.

6. **Notification email to the old address after reset** — "your password was changed". If the attacker took over, the real user gets alerted.

7. **Short expiration** — 15-60 minutes. Not 24 hours.

8. **Tokens in URL, not GET params that hit logs** — actually GET is fine if you take care (don't log full URLs, use HTTPS, and remember Referer can leak to third-party scripts on the reset page — use `<meta name="referrer" content="no-referrer">` on that page).

**Don't:**

- Derive the token from `user_id + timestamp` — guessable.
- Use JWTs for reset tokens — JWTs are bearer tokens; if you can't invalidate them before expiration, you've lost single-use. Random opaque tokens stored in a DB are better here.
- Let the user pick the same password they had. Require a different one (lightweight check — not true rotation enforcement, but prevents obvious laziness).
- Skip sending an email to the old address — that's how users find out they've been hacked.""",
))

EXAMPLES.append(example(
    user="""What's wrong with storing passwords as SHA-256 hashes? I've heard it's bad but SHA-256 is cryptographically secure, right?""",
    think="""Common misconception. SHA-256 is cryptographically secure in the sense that it's collision-resistant and pre-image resistant. But that's not what password hashing needs.

Password hashing needs:
1. Slow — intentionally so, to make brute force expensive
2. Memory-hard — so GPU/ASIC attacks are less effective
3. Per-password salt — to defeat rainbow tables and parallel attacks on users with the same password

SHA-256 is fast. A single modern GPU can compute billions of SHA-256 hashes per second. A password dataset protected by SHA-256 (even with salt) can be brute-forced by dictionary in hours.

The right answer is a password-specific KDF: argon2id (current best), bcrypt (widely supported, older), scrypt (memory-hard alternative), PBKDF2 (FIPS compliant but weaker).

argon2id is the winner of the Password Hashing Competition (2015) and is what you should use for new projects.

I should explain the "why" clearly and give concrete migration advice.""",
    answer="""SHA-256 is secure **for its intended use** (data integrity, digital signatures, etc) but **wrong for passwords**. The core issue: SHA-256 is *fast*. That's a feature for most uses and a disaster for password storage.

**The attack math:**

- A single NVIDIA RTX 4090 can compute ~20 billion SHA-256 hashes per second.
- A typical password database has thousands to millions of users.
- An 8-character lowercase password has 208 billion combinations.
- Brute-forcing a single SHA-256 hash: ~10 seconds. The whole database: hours.
- Salting helps against rainbow tables, but doesn't slow per-password brute force.

**What password hashing actually needs:**

1. **Slow** — tune so a single hash takes ~100-500 ms. Users accept this at login; attackers can't iterate 20 billion of them per second.
2. **Memory-hard** — require significant RAM per hash, defeating GPU/ASIC parallelism.
3. **Per-password salt** — prevents attacking identical passwords across users in parallel.
4. **Tunable cost** — as hardware gets faster, you bump the cost parameter.

**Use argon2id:**

```python
# Python — pip install argon2-cffi
from argon2 import PasswordHasher

ph = PasswordHasher()  # defaults are currently safe; tune for your hardware

# Hashing on signup:
hash = ph.hash("user_password_here")
# → "$argon2id$v=19$m=65536,t=3,p=4$..." (salt is embedded)

# Verifying on login:
try:
    ph.verify(hash, "user_password_here")
    if ph.check_needs_rehash(hash):
        new_hash = ph.hash("user_password_here")
        db.update_user_password_hash(user_id, new_hash)
except VerifyMismatchError:
    raise InvalidCredentials()
```

```js
// Node — npm install argon2
import argon2 from 'argon2';

const hash = await argon2.hash(password, { type: argon2.argon2id });
const valid = await argon2.verify(hash, password);
```

**Acceptable alternatives:**

- **bcrypt** — older, widely supported, still fine. Not memory-hard but still slow. Use `bcrypt` library with cost factor 12+. Downside: 72-byte password limit.
- **scrypt** — memory-hard, older than argon2, reasonable choice.
- **PBKDF2** — use only if compliance requires FIPS 140-2. Use SHA-256 underneath with 600,000+ iterations (OWASP 2023 recommendation).

**Don't:**

- SHA-256, SHA-512, MD5 alone — all too fast.
- "SHA-256 + salt" — salt defeats rainbow tables, not brute force.
- "SHA-256 iterated N times" — you've reinvented PBKDF2 badly; use the real one.
- Custom schemes. Cryptographers have been designing password hashing for decades; yours is almost certainly worse.
- Store the salt separately from the hash. Argon2/bcrypt encode salt + parameters + hash in a single string — keep them together.

**Migration from SHA-256 to argon2id (for existing apps):**

You can't re-hash passwords without the plaintext. Two options:

1. **Lazy migration** — when a user logs in, they provide the plaintext. Verify against the SHA-256 hash, then re-hash with argon2id and store. After some months, force a password reset for any user who hasn't logged in.

2. **Wrap** — hash passwords as `argon2id(sha256(password))`. On login, do the same wrap and compare. This immediately protects all existing hashes. Users can be gradually migrated to pure argon2id on their next login. This is the approach Dropbox used.

**How to tune cost**: pick parameters so hashing takes ~250-500ms on your server. Re-benchmark every year or two and bump the cost factor. argon2id's `memory_cost` defaults are a good starting point.""",
))

EXAMPLES.append(example(
    user="""I'm verifying JWTs in my API. I'm checking the signature with my secret. Is that enough?""",
    think="""No, signature verification is necessary but far from sufficient. Common JWT validation failures:

1. Not checking `exp` (token never expires in your code's view)
2. Not checking `nbf` (not-before, rare but worth checking)
3. Not checking `aud` (audience) — tokens intended for one service reused for another
4. Not checking `iss` (issuer) — accepting tokens from any issuer
5. Using `alg: none` vulnerability — old libraries honored "alg: none" in the token header; rejecting unsigned tokens requires explicit check
6. HS256 vs RS256 confusion — attacker sends token with alg=HS256 signed with your public key (which they can read)
7. Not checking `jti` for revoked tokens (no revocation list)
8. Not validating custom claims (user_id corresponds to a real, non-banned user)

Also critical: using a strong enough secret (256+ bits for HS256, or asymmetric keys).

I should cover these as a checklist with code.""",
    answer="""**No, signature verification alone is not enough.** A valid signature just means "the token wasn't tampered with". It doesn't tell you the token is current, meant for you, or still authorized.

**The full validation checklist:**

1. **Signature** — verify with your key/secret. Obvious.
2. **`alg`** — explicitly require the algorithm you expect; don't trust the header's `alg` field.
3. **`exp`** (expiration) — reject if past.
4. **`nbf`** (not-before) — reject if in the future.
5. **`iss`** (issuer) — must match your expected issuer.
6. **`aud`** (audience) — must match your service identifier.
7. **`jti` / revocation** — check against a denylist for logged-out tokens.
8. **User still exists and is active** — the token says `sub: user_id_123`, but the user may have been banned/deleted after the token was issued.

**Example (Node + `jsonwebtoken`):**

```js
import jwt from 'jsonwebtoken';

function verifyToken(token) {
  let payload;
  try {
    payload = jwt.verify(token, SECRET, {
      algorithms: ['HS256'],          // lock algorithm — prevent alg confusion
      issuer: 'https://auth.myapp.com',
      audience: 'https://api.myapp.com',
      // exp and nbf are checked automatically by jwt.verify
    });
  } catch (e) {
    throw new UnauthorizedError(`token: ${e.message}`);
  }

  // Revocation check
  if (revokedJtis.has(payload.jti)) {
    throw new UnauthorizedError('token revoked');
  }

  // User still valid?
  const user = db.users.findById(payload.sub);
  if (!user || user.status !== 'active') {
    throw new UnauthorizedError('user inactive');
  }

  return { user, payload };
}
```

**The classic attacks you're defending against:**

**1. `alg: none` attack**

Old JWT libraries honored `{"alg": "none"}` in the header, meaning "no signature needed". Attacker forges a token with `alg: none` and an empty signature. If your code just calls `jwt.decode()` without a strict algorithm list, it's accepted.

**Fix**: pass `algorithms: ['HS256']` explicitly. Never trust the header's `alg`.

**2. Algorithm confusion (RS256 → HS256)**

Your service uses RS256 (public/private key). The public key is, well, public. Attacker changes the token header to `alg: HS256` and signs it **using your public key as the HMAC secret**. If your verification code accepts either algorithm, it uses the public key as an HMAC key and verifies the attacker's forgery.

**Fix**: lock the algorithm list to exactly what you accept.

**3. Missing `aud` check**

Your auth service issues tokens for both `api.myapp.com` and `internal.myapp.com`. A user token for `api.myapp.com` is captured and sent to `internal.myapp.com`. If internal doesn't check `aud`, it accepts the token.

**Fix**: always check `aud` matches your service.

**4. No revocation**

User clicks "log out everywhere". Their old tokens are still valid until they expire (potentially hours). For high-security apps, maintain a revocation list (Redis set of `jti`s) and check on every request. For lower security, accept the expiration lag and use short-lived tokens (~5 min) + refresh tokens.

**5. User deleted/banned after token issue**

Token is valid, signed, not expired. But the user is banned. If you don't re-check user status on each request, they're still in. For sensitive operations, always re-check.

**Also:**

- **Use a strong secret** — 256 bits minimum for HS256. `openssl rand -hex 32`. Not "secret123".
- **Rotate keys** — support a `kid` header and multiple active keys so you can rotate without downtime.
- **Don't store sensitive data in the payload** — JWTs are base64, not encrypted. Anyone can read `user_id`, `email`, `roles`. Put nothing in there you wouldn't publish.
- **Short expiration + refresh tokens** — 5-15 min access tokens, 30-day refresh tokens stored server-side with revocation support. This limits the blast radius of a token leak.""",
))

EXAMPLES.append(example(
    user="""Our app lets users upload files (profile pictures, documents). What security issues should I worry about and how do I handle them?""",
    think="""File uploads are a classic minefield. Issues:

1. Path traversal — filename contains "../../../" and overwrites arbitrary files
2. Executable content — user uploads .php or .exe, accessed via URL = RCE
3. MIME type spoofing — says "image/png" but is actually PHP
4. Content sniffing — browser detects script content in "image" and runs it
5. XSS via SVG — SVGs can contain JavaScript
6. Unlimited size / disk exhaustion
7. Metadata leaks (EXIF GPS coordinates)
8. Filename XSS when displayed back to users
9. Zip bombs (for archive uploads)
10. Image parser bugs (ImageMagick had many — ImageTragick)
11. Storing in web-accessible directory
12. Not scanning for malware (for business apps)

Defenses:
1. Generate new filenames server-side (UUID)
2. Store outside web root, serve through an app handler that sets Content-Disposition
3. Validate MIME by parsing (not trusting extension/header)
4. Allow-list content types
5. Size limits (request-level and per-file)
6. X-Content-Type-Options: nosniff
7. Store in object storage (S3, GCS) with strict permissions
8. Strip EXIF from images
9. Rate limit uploads

I'll give the concrete pattern.""",
    answer="""File uploads have a dozen attack vectors. Treat every uploaded byte as hostile.

**Core threats:**

1. **RCE via executable upload** — attacker uploads `shell.php`, server serves it as PHP = code execution.
2. **Path traversal** — filename `../../../etc/passwd` overwrites arbitrary files.
3. **MIME spoofing** — header says `image/png`, content is JavaScript or PHP.
4. **XSS via SVG** — SVGs are XML and can contain `<script>`.
5. **Content sniffing** — browsers may detect and execute script content in files labeled as images.
6. **Disk exhaustion** — huge files or many small files.
7. **EXIF leaks** — phone photos contain GPS coordinates.
8. **XSS via filename** — `<script>.jpg` displayed unescaped in the UI.
9. **Zip bombs** — 1 KB zip expands to 10 GB.
10. **Image parser exploits** — ImageMagick's ImageTragick was devastating.

**Safe pattern — checklist:**

```js
import crypto from 'node:crypto';
import path from 'node:path';
import { fileTypeFromBuffer } from 'file-type';
import sharp from 'sharp';
import { S3 } from '@aws-sdk/client-s3';

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const MAX_SIZE = 5 * 1024 * 1024;  // 5 MB

async function handleUpload(req, res) {
  const file = req.file;  // from multer or similar
  if (!file) return res.status(400).json({ error: 'no file' });

  // 1. Size check (multer should already enforce, double-check)
  if (file.size > MAX_SIZE) return res.status(413).json({ error: 'too large' });

  // 2. Detect actual type by reading magic bytes — don't trust the client
  const detected = await fileTypeFromBuffer(file.buffer);
  if (!detected || !ALLOWED_TYPES.has(detected.mime)) {
    return res.status(400).json({ error: 'unsupported type' });
  }

  // 3. Re-encode images to strip payloads and EXIF
  //    This also catches hidden non-image content inside "image" files
  let clean;
  try {
    clean = await sharp(file.buffer)
      .rotate()                  // respect EXIF rotation, then strip
      .resize(2000, 2000, { fit: 'inside', withoutEnlargement: true })
      .toFormat(detected.ext)
      .toBuffer();
  } catch (e) {
    return res.status(400).json({ error: 'image decode failed' });
  }

  // 4. Generate server-side filename — never use user-supplied name
  const id = crypto.randomUUID();
  const key = `uploads/${req.user.id}/${id}.${detected.ext}`;

  // 5. Store in object storage, not the web root
  await s3.putObject({
    Bucket: 'my-uploads',
    Key: key,
    Body: clean,
    ContentType: detected.mime,
    CacheControl: 'private, max-age=31536000',
    // Block execution in case it's served wrong:
    Metadata: { 'x-content-type-options': 'nosniff' },
  });

  // 6. Record in DB with server-generated ID, not the filename
  await db.files.create({
    id, user_id: req.user.id, key,
    original_name: sanitizeForDisplay(file.originalname),  // for UI only
    mime: detected.mime, size: clean.length,
  });

  res.json({ id });
}

function sanitizeForDisplay(name) {
  // For display only — never use for filesystem or URLs
  return name.replace(/[^\\w.\\- ]/g, '').slice(0, 200);
}
```

**Key points:**

1. **Detect type by magic bytes**, not the extension or the Content-Type header. Use `file-type` or similar. The client lies.
2. **Re-encode images** with `sharp` or `ImageMagick`. This strips EXIF, strips any embedded payloads, and catches fake "images" that don't parse. It also size-limits the output.
3. **Generate the filename server-side** (UUID). Never trust `file.originalname` for anything except display. Path traversal becomes impossible because you never use the user's name.
4. **Store outside the web root** — use S3/GCS or a directory the web server can't serve directly. Serve downloads through an app endpoint that checks authorization and sets `Content-Disposition: attachment`.
5. **Block execution by MIME** — set `X-Content-Type-Options: nosniff` on downloads so browsers don't sniff an image as script.
6. **Allow-list types**, not deny-list. `['image/jpeg', 'image/png']` not "anything except .php".
7. **No SVG** unless you absolutely need it. If you do, parse with a library that strips scripts, or render it as a `<img src="...">` (doesn't execute scripts) instead of inline `<svg>`.
8. **Rate limit** uploads per user and per IP.
9. **Authorize downloads**. `GET /files/:id` should check the user owns (or has permission to view) the file.

**For business apps handling arbitrary documents** (PDFs, Word, etc):

- Scan with ClamAV or a commercial malware scanner.
- For PDFs, consider rendering to PNG server-side before showing a preview — PDF parsers have a terrible security history.
- Sandbox anything that parses user files (separate process, seccomp, or a dedicated worker container).

**Don't:**

- Trust the extension. `shell.php.jpg` is a classic bypass if you check only the last extension.
- Trust the Content-Type header. It's user-controlled.
- Store files with user-supplied names in a web-served directory. Ever.
- Return detailed error messages (`"file shell.php rejected: .php not allowed"`) — they help attackers enumerate your filters.
- Assume your image library is safe. Keep sharp/ImageMagick/libvips patched; subscribe to their security advisories.""",
))

EXAMPLES.append(example(
    user="""What's Content Security Policy (CSP) and should I bother setting one up? It seems like a lot of work for marginal benefit.""",
    think="""CSP is defense in depth against XSS. The user is asking if it's worth it. Short answer: yes, for any app that handles user content or sensitive data. It dramatically reduces XSS impact even when you have a sanitization bug.

Key points:
1. CSP is a header telling the browser which sources are allowed for scripts, styles, images, etc
2. A good CSP prevents inline scripts and unknown script sources
3. Even if XSS gets injected, it can't execute because the browser refuses
4. Biggest wins: `script-src 'self'` with nonces/hashes, `object-src 'none'`, `frame-ancestors` against clickjacking

The work is:
- Initial setup (the hard part)
- Deal with inline scripts (move to files, use nonces)
- Report-only mode first to catch false positives
- Maintain as the app evolves

I'll explain what it protects, give a starting config, and mention the report-only workflow.""",
    answer="""**Yes, it's worth it** — CSP is one of the highest-ROI security headers. It's defense in depth: even if a sanitization bug lets XSS slip through, CSP prevents the injected script from executing. It's also the only practical defense against some classes of attack (clickjacking, data exfiltration via third-party scripts).

**What CSP does:**

A `Content-Security-Policy` header tells the browser: "only load scripts from these sources, only connect to these APIs, etc". Anything outside the policy is blocked.

**Starter policy for a typical app:**

```
Content-Security-Policy:
  default-src 'self';
  script-src 'self' 'nonce-{random_per_request}';
  style-src 'self' 'nonce-{random_per_request}';
  img-src 'self' data: https:;
  font-src 'self' data:;
  connect-src 'self' https://api.myapp.com;
  frame-ancestors 'none';
  form-action 'self';
  base-uri 'self';
  object-src 'none';
  upgrade-insecure-requests;
```

**What each line does:**

- **`default-src 'self'`** — fallback: only same-origin for anything not specified.
- **`script-src 'self' 'nonce-...'`** — scripts only from your domain + nonce-tagged inline scripts. Kills `<script>` injection without the nonce.
- **`style-src`** — same for stylesheets.
- **`img-src 'self' data: https:`** — images from your domain, data URIs, or HTTPS (broad; tighten if possible).
- **`connect-src`** — where `fetch`, `XMLHttpRequest`, WebSocket can connect. Prevents exfiltration to attacker domains.
- **`frame-ancestors 'none'`** — nobody can `<iframe>` your site. Defeats clickjacking (replaces `X-Frame-Options: DENY`).
- **`form-action 'self'`** — forms can only submit to your domain. Prevents form hijacking.
- **`base-uri 'self'`** — prevents `<base href>` injection attacks.
- **`object-src 'none'`** — no Flash, Java applets, etc. Nobody should be using these in 2026.
- **`upgrade-insecure-requests`** — auto-upgrade `http://` URLs to HTTPS.

**The inline-script problem:**

If your pages have `<script>console.log('hi')</script>` or `<button onclick="doThing()">`, a strict `script-src 'self'` breaks them. Three fixes:

1. **Move inline scripts to files** — best long-term.
2. **Use a nonce per response** — generate a random string, include it in the header and on every inline script tag:

    ```html
    Content-Security-Policy: script-src 'self' 'nonce-r4nd0m';
    <script nonce="r4nd0m">console.log('allowed');</script>
    ```

3. **Use hashes** — `script-src 'sha256-...'`. Good for static inline scripts, painful to maintain.

**Don't use `'unsafe-inline'`** — it lets any inline script run, which defeats the XSS protection. It's a common shortcut and a trap.

**The rollout:**

CSP breaks things when you turn it on. The migration path:

1. **Deploy with `Content-Security-Policy-Report-Only`** first. This sends violation reports to a `report-uri` endpoint without blocking anything. Collect reports for 1-2 weeks.
2. **Fix the violations** — inline scripts, external resources, etc. Each fix either moves code to a file, adds a nonce, or adds a trusted source to the policy.
3. **Switch to enforce mode** (`Content-Security-Policy`) when the report noise is near zero.
4. **Keep Report-Only in parallel** with a tighter policy you're testing for the next iteration.

**Integration (Node/Express):**

```js
import crypto from 'node:crypto';

app.use((req, res, next) => {
  res.locals.nonce = crypto.randomBytes(16).toString('base64');
  res.setHeader('Content-Security-Policy',
    `default-src 'self'; ` +
    `script-src 'self' 'nonce-${res.locals.nonce}'; ` +
    `style-src 'self' 'nonce-${res.locals.nonce}'; ` +
    `object-src 'none'; ` +
    `frame-ancestors 'none'; ` +
    `base-uri 'self'; ` +
    `upgrade-insecure-requests;`
  );
  next();
});
```

Then in your templates, use the nonce:

```html
<script nonce="<%= nonce %>">...</script>
```

**Don't:**

- Use `'unsafe-inline'` or `'unsafe-eval'`. These undo most of CSP's protection.
- Use wildcard sources (`*.google.com`) unless you really need them. Attackers rent subdomains.
- Forget `frame-ancestors` — it replaces `X-Frame-Options` and prevents clickjacking.
- Copy a policy from someone else's site without review. Your sources are different.

**Diminishing returns**: CSP doesn't protect against server-side issues (SQL injection, SSRF, IDOR). It's specifically a client-side defense layer. Use it on top of sanitization, not instead.""",
))


def main() -> None:
    print(f"Appending {len(EXAMPLES)} new examples to {TARGET}...")
    with open(TARGET, "a", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Done. Run: PYTHONPATH=. python3 -m agents.distill.filter_reasoning --merge")
    print("       to rebuild claude_reasoning.jsonl (merged training file).")


if __name__ == "__main__":
    main()
