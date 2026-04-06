# DISCOVER — Quick Single-Agent Investigation

**Input**: [$ARGUMENTS] - Feature/bug description, query, or area to investigate

**Single-agent discovery**: One explorer performs focused code investigation and reports findings. No team creation, no cross-challenge — fast and lightweight.

**When to use:**
- **/DISCOVER**: Quick investigation, narrow scope, single concern area
- **/DISCOVER-DEEP**: Complex investigation, cross-module, harness + training + runtime concerns

---

## Step 0: Conceptual Analysis (before spawning)

Think about the request:
- What exactly is being investigated?
- Identify search terms: function names, module names, class names, config keys
- Determine relevant domains: harness (`agents/`), distillation (`agents/distill/`), rust (`rust/`), models (`models/`)
- This is conceptual only — no codebase searching

---

## Investigation

Spawn a single explorer agent:

`subagent_type: explorer`, `model: opus`

> Perform THOROUGH code exploration for: $ARGUMENTS
>
> **Context from orchestrator**: [inject Step 0 output — search terms, directories, domain objects]
>
> ### Serena-Assisted Search
> Use Serena tools for structural analysis:
> - `get_symbols_overview` for file structure
> - `find_symbol` for locating specific symbols
> - `trace_dependencies` for import/dependency graphs
> - `search_for_pattern` for flexible pattern matching
> - `find_referencing_symbols` for consumer tracking
>
> Fallback to Grep/Glob/Read for non-structural searches.
>
> ### 4-Phase Search: DISCOVER → LOCATE → UNDERSTAND → VALIDATE
>
> **DISCOVER**: Broad scoping — find all files related to the area
> **LOCATE**: Get outlines and symbol maps for target files
> **UNDERSTAND**: Read symbol bodies, trace dependencies, map data flow
> **VALIDATE**: `think_about_collected_information` — verify findings are consistent
>
> ### Report Format
> ```
> ## Investigation: [$ARGUMENTS]
>
> ### Files Involved
> [File list with line counts and key symbols]
>
> ### Architecture
> [How the area fits into the broader system]
>
> ### Dependencies
> [ASCII trees from trace_dependencies]
>
> ### Current Behavior
> [What the code currently does]
>
> ### Issues Found
> [Bugs, gaps, inconsistencies]
>
> ### Recommendations
> [Concrete suggestions with file:line references]
> ```

---

## Output

Orchestrator reads explorer's findings and presents to user. No code changes — analysis and recommendations only.

---

## Rules

- Single agent, no team creation
- Explorer runs in foreground (orchestrator waits for result)
- No code changes — investigation only
- Serena tools preferred for structural analysis
- `think_about_collected_information` mandatory before reporting
- Step 0 context injected into explorer prompt
