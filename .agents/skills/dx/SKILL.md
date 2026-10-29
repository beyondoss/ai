---
name: dx
description: Developer experience design tool. Uses Jobs-to-be-Done, Jakob's Law, and DOET (signifiers, affordances, feedback, conceptual model) to design or evaluate APIs, SDKs, CLIs, and frameworks as interfaces. Works in two modes — Design (greenfield, shape what to build) and Audit (evaluate what exists). Surfaces gaps between what developers need to accomplish, what they expect, and what the surface actually communicates. Use when starting a new API, SDK, CLI, or framework, or when asked to review DX, API ergonomics, CLI usability, or interface quality. Trigger with "dx", "dx audit", "dx review", "design this api", "design an sdk", "design a cli", "developer experience", "api ergonomics", "sdk design", "cli design", "framework design", "affordances", "signifiers", "jakob's law", "interface design".
allowed-tools: "Read,Glob,Grep,Bash(git log:*),Bash(git diff:*),Bash(cargo:*),Bash(mise:*),Bash(cat:*)"
version: "1.1.0"
---

# Developer Experience Design

Design or evaluate APIs, SDKs, CLIs, and frameworks as interfaces. Derives what developers need to accomplish, checks whether the surface matches ecosystem expectations, and diagnoses where the interface fails to signal its own capabilities and constraints.

Works in two modes:

- **Design mode** — greenfield. Shape the surface before writing code. Produce a concrete interface design: endpoints, method signatures, error shapes, config schema, composable primitives.
- **Audit mode** — evaluate what exists. Find gaps between the jobs developers need to accomplish and how well the current surface serves them.

Determine the mode from context. If there's no existing implementation to evaluate, use Design mode. If there's a codebase, use Audit mode. If the user asks to both evaluate and improve, run Audit then produce Design-mode output for the gaps.

---

## The Three Lenses

A developer-facing interface — API, SDK, CLI, framework — is a product. It has users. Those users bring needs, expectations, and the ability to be confused or delighted. The goal of DX design is to collapse the gap between what a developer needs to do, what they expect the surface to look like, and what the surface actually communicates.

Three frameworks, applied together, form a complete lens:

---

### Lens 1: Jobs to Be Done (JTBD)

**Core idea**: Developers don't adopt tools because of features. They hire tools to make progress on something they're trying to accomplish. The job is the unit of analysis — not the feature, not the user story.

A well-formed job: _"When [situation], I want to [motivation], so I can [outcome]."_

Jobs operate at multiple levels:

| Dimension      | Question                              | Example                                                         |
| -------------- | ------------------------------------- | --------------------------------------------------------------- |
| **Functional** | What are they literally trying to do? | Send a message to a queue without blocking the request          |
| **Emotional**  | How do they want to feel?             | Confident it won't silently drop messages                       |
| **Social**     | How do they want to be perceived?     | Someone who ships resilient systems, not someone who loses data |

The job is **stable and solution-agnostic** — it exists whether or not your service exists. Features are proposed solutions. Design that starts from features works backwards from answers. Start from jobs.

**JTBD tells us**: What progress does the developer need to make? Is the surface shaped around those jobs?

---

### Lens 2: Jakob's Law

**Core idea**: Developers spend most of their time using other tools, not yours. They arrive with a mental model built from every SDK, CLI, and API they've already learned. If your surface matches that model, they're immediately productive. If it deviates, you've created a tax — a thing they must learn before they can work.

Jakob's Law: _"Users spend most of their time on other sites. This means that users prefer your site to work the same way as all the other sites they already know."_

Applied to developer tooling:

- **Match ecosystem conventions** — naming, casing, error shapes, pagination patterns, config idioms. If every major SDK uses `client.resource.verb()`, yours should too.
- **Deviate intentionally** — breaking a convention is sometimes the right call. But it must earn the right by being demonstrably better, and it must be communicated clearly, not discovered through confusion.
- **Familiarity lowers the cost of adoption** — a developer who recognizes your surface picks it up in minutes. A developer who must re-learn everything may reach for a competitor instead.

**Jakob's Law tells us**: Does this feel familiar? Where are we asking developers to unlearn things? Is that trade-off worth it?

---

### Lens 3: The Design of Everyday Things (DOET)

Don Norman's framework for physical interfaces applies directly to developer surfaces. Every API, SDK, and CLI is a set of controls. Those controls either guide users toward correct use or leave them to guess.

**Five properties — all apply to developer interfaces:**

**Signifiers**: What signals what the interface can do?

- Method names, parameter names, types, return types — these are signifiers.
- A method named `send` signals less than `send_with_retry`. A parameter typed `Duration` signals more than one typed `u64`.
- Weak signifiers force developers to read docs before they can act. Strong signifiers let them proceed on instinct.

**Affordances**: What does the interface actually allow?

- The real capabilities of the surface. What can be done?
- The gap between affordances and signifiers is where bugs and confusion live. When an interface signals one thing and does another, trust breaks.

**Feedback**: What does the interface tell you about what happened?

- Errors should name what went wrong, not just that something went wrong.
- Return types should make success and failure explicit — not buried in a nullable field or a status code you have to remember to check.
- Silent failure (returns `nil`, drops messages, ignores unknown config) is the worst feedback: it signals success while hiding failure.

**Conceptual model**: Does the interface's model match the developer's model?

- If developers think of a queue as a FIFO list and the API exposes it as a weighted graph, they'll use it wrong no matter how well the methods are named.
- Leaky abstractions expose the wrong model — implementation details that shouldn't be visible. Incomplete abstractions leave jobs half-served.

**Constraints**: What does the interface make impossible or hard?

- Good constraints guide developers toward correct use without requiring them to understand every invariant.
- Bad constraints make the right thing hard and the wrong thing easy.

**DOET tells us**: Does the surface signal its own capabilities correctly? Are the conceptual models aligned? Does the interface guide toward correct use or leave users to guess?

---

## The Integrating Principle

The three lenses converge on one question: **Does the interface keep its promise?**

- **JTBD** defines the promise: what the developer came here to accomplish.
- **Jakob's Law** defines the expectation: what the developer assumes the promise looks like.
- **DOET** defines the signal: whether the interface communicates that it can keep the promise.

A gap at any layer is a broken promise:

- JTBD gap: the job isn't served — the developer can't accomplish what they came to do.
- Jakob's Law gap: the convention is violated — the developer can't proceed on instinct and must pay a learning tax.
- DOET gap: the signal is wrong — the developer can't tell what the interface does, or the interface does something other than what it signals.

---

## API-First Principle

**Everything starts with the API. The implementation is just your detail.**

The API — the interface contract — is what developers depend on. The implementation may be rewritten, optimized, replaced. The API persists. This means:

- Design the interface as if the implementation doesn't exist yet. What would a developer need to call?
- Treat implementation constraints as invisible to the caller. If you can't serve a job cleanly at the API layer, fix the implementation — don't expose the limitation as an API wart.
- **Composability is a first-class property.** A surface where small, focused pieces combine naturally is better than a surface where large, opaque operations hide the seams. Composable parts let developers build exactly what they need; monolithic operations lock them into what you anticipated.
- **Progressive disclosure**: simple things should be simple, complex things should be possible. The common case should require the least ceremony.

---

## Mode: Design

Use when shaping a new surface before implementation, or when redesigning after an audit identifies structural gaps.

### Step 1: Establish Context

Ask or infer:

- What domain is this serving? (queues, auth, payments, storage, etc.)
- Who are the developers that will use it? (backend engineers, operators, coding agents, framework authors)
- What stack or ecosystem? (Rust, Go, TypeScript, Python — this determines naming and idiom conventions)
- What adjacent tools do these developers already know? (this informs Jakob's Law baselines)
- What layers are in scope? (REST API only? SDK? CLI? All three?)

If context is incomplete, state assumptions explicitly before proceeding.

### Step 2: Derive the Jobs

Write 8–15 jobs the surface must serve. Job statements are the specification. Every design decision should trace back to a job.

Format: _"When [situation], I want to [motivation], so I can [outcome]."_

Group by functional area. Mark high-frequency jobs — these drive the progressive disclosure hierarchy. The common path must be the short path.

### Step 3: Identify Ecosystem Conventions

For each layer in scope, identify what the target ecosystem expects:

**REST API**: What do developers already know from Stripe, Twilio, GitHub, AWS? Error shape, pagination cursor patterns, resource naming, HTTP status semantics.

**SDK**: What does the standard library and dominant packages in the language do? Method naming, builder vs options pattern, error as value vs exception, async vs sync defaults.

**CLI**: What do tools like `git`, `kubectl`, `cargo`, `gh` do? Subcommand structure, flag naming, output formats (`--json`, `--quiet`), config file location and format.

**Config**: Twelve-factor app conventions (env vars for secrets, files for structure), naming conventions (snake_case, SCREAMING_SNAKE for env).

These are the baselines. Deviations need explicit justification.

### Step 4: Design the Surface

For each layer in scope, produce a concrete interface design. Be specific — not "add a method for X" but the actual signature, shape, and behavior.

**REST API design**:

- Resource table: `METHOD /path` → description, request body, response body
- Error shape (pick one and commit to it across the whole surface)
- Pagination strategy (cursor, offset, keyset)
- Auth mechanism
- Versioning approach

**SDK design**:

- Client initialization: what's required, what's optional, what has sensible defaults
- Method signatures: show actual code in the target language
- Return types: how are errors surfaced? (Result<T, E>, exceptions, callbacks)
- Fluent patterns vs builder vs options struct — pick based on language idiom
- What does the 5-line happy-path look like? That's the signifier test.

**CLI design**:

- Command tree: `tool <resource> <verb> [flags]`
- Output: human-readable default, `--json` for scripting
- Exit codes: 0 success, 1 usage error, 2 runtime error
- Config: where does it live, how is it scoped (global vs project)

**Config design**:

- Complete list of knobs with types, defaults, and valid ranges
- Which are required vs optional
- Env var names and config file equivalents

### Step 5: Apply the Three Lenses

For each design decision, verify:

**JTBD check**: Which job does this serve? If you can't name the job, cut the feature.

**Jakob's Law check**: Does this match what the target ecosystem does? If it deviates, is the deviation documented and justified?

**DOET check**:

- Signifiers: does the name/type telegraph intent without documentation?
- Affordances: does the interface do what it appears to do?
- Feedback: what does success look like? What does failure look like? Is failure visible?
- Conceptual model: does the abstraction match the domain model the developer already holds?
- Constraints: does the interface make the wrong thing hard to do?

### Step 6: Deliver the Design

```
## DX Design: [Surface Name]

### Context & Assumptions
[Domain, audience, ecosystem, layers in scope]

### Jobs to Be Done
[Grouped list, job statement format — these are the spec]

### Ecosystem Baselines
[What the target ecosystem expects, per layer]

### Interface Design
[Concrete, specific — actual endpoints, actual method signatures, actual config keys]

### Design Decisions
[Each non-obvious choice: what convention was followed or broken and why]

### Composability Model
[The primitives. What combines with what. What the caller controls vs what the library controls.]

### The Happy Path
[The 5-line example. What does the most common job look like end-to-end?]
```

---

## Mode: Audit

Use when evaluating an existing developer-facing surface.

### Step 1: Orient to the Project

Read enough to build a confident picture of:

- What domain this service operates in
- Who calls it (operators, developers building on top of it, agents, machines)
- What the full developer-facing surface looks like — REST API, SDK, CLI, config, error messages, webhooks, docs

Start with:

- `CLAUDE.md` or `README.md` — service model, constraints, philosophy
- `ARCHITECTURE.md` — if present
- Routing layer — the REST surface (`src/routes/`, `src/handlers/`, or equivalent)
- SDK or client library — if present (`clients/`, `sdk/`, or sibling repos)
- CLI — if present
- Config schema — env vars, config files, `config.rs` or equivalent
- Error types — `errors.rs`, error variants, HTTP error responses

Do not read everything. Read enough to understand the domain and surface.

### Step 2: Map the Full Developer-Facing Surface

Enumerate everything a developer or operator touches:

**REST API**: Method, path, brief description. Compact table.

**SDK / client libraries**: What languages? Abstraction level — thin HTTP wrapper or higher-level? Typed models? Auth, retry, pagination handled?

**CLI**: What commands exist? What jobs do they serve?

**Configuration**: Env vars, config files, secrets. Are names clear? Are defaults sensible? Is everything documented in one place?

**Error surface**: What do errors look like? Are they actionable? Do they include what went wrong and what to try?

**Webhooks / events**: If emitted, what's the payload shape?

**Docs / onboarding**: What does a new integrator read first?

Note absences. A missing SDK, no CLI, undocumented config — these are surface gaps.

### Step 3: Derive the Jobs to Be Done

Write 8–15 jobs, grounded in evidence from the codebase (routes, migrations, test descriptions, error types, docs). Every job must have a signal in the code. No invented jobs.

Format: _"When [situation], I want to [motivation], so I can [outcome]."_

Group by functional area. Note functional, emotional, and social dimensions where relevant.

### Step 4: Evaluate Each Lens

#### JTBD Evaluation

For each job: is it served? At which layer? How well?

| Job                 | Served?            | Layer                     | Quality                   |
| ------------------- | ------------------ | ------------------------- | ------------------------- |
| When X, I want to Y | Yes / Partial / No | REST / SDK / CLI / Config | Clean / Friction / Broken |

#### Jakob's Law Evaluation

Identify ecosystem conventions for this type of surface. Compare:

| Convention        | Ecosystem standard                             | This surface               | Gap?                            |
| ----------------- | ---------------------------------------------- | -------------------------- | ------------------------------- |
| Error shape       | `{"error": {"code": "...", "message": "..."}}` | `{"error": "..."}`         | Minor — missing structured code |
| SDK method naming | `client.resource.verb()`                       | `client.verbResource()`    | Major — inverted convention     |
| Config            | ENV or config file, snake_case keys            | Mixed camelCase/snake_case | Minor — inconsistent casing     |

Note intentional deviations separately. Is the deviation documented? Is the alternative demonstrably better?

#### DOET Evaluation

For each major surface area:

| Property                                              | Finding | Severity |
| ----------------------------------------------------- | ------- | -------- |
| Signifiers — do names/types telegraph intent?         |         |          |
| Affordances — gap between signal and capability?      |         |          |
| Feedback — are errors actionable? Is failure visible? |         |          |
| Conceptual model — matches the developer's model?     |         |          |
| Constraints — guides toward correct use?              |         |          |

### Step 5: Evaluate Composability

- **Granularity**: Are operations at the right level? Too coarse forces workarounds; too fine requires assembling too many pieces.
- **Orthogonality**: Do parts combine cleanly, without unexpected interactions?
- **Extension points**: Can developers add behavior without forking? Middleware, hooks, interceptors, extension traits.
- **Surface area**: A smaller surface that composes is better than a larger surface that doesn't. Every method, endpoint, or flag that can be derived from simpler primitives is a liability.

### Step 6: Propose Changes — Only for Real Gaps

For each gap found, propose a change at the right layer. Not every gap requires an API change. A conceptual model gap might be fixed with docs. A signifier gap might be fixed by renaming a method. A missing job might require a new endpoint.

Each proposal must include:

- Which lens identified it (JTBD / Jakob's / DOET)
- What gap it closes
- Which layer the change lives at (REST, SDK, CLI, config, docs, error messages)
- The specific change — concrete, not gestural
- Why the existing approach is insufficient
- Trade-offs

If a gap is cosmetic or the existing approach is defensible, say so and omit the proposal.

### Step 7: Deliver the Report

```
## DX Audit: [Service Name]

### Developer-Facing Surface (current)
[Compact summary of REST, SDK, CLI, config, errors, docs]

### Jobs to Be Done
[Grouped list, job statement format]

### JTBD Analysis
[Job → layer → quality table]

### Jakob's Law Analysis
[Convention comparison table — gaps and intentional deviations]

### DOET Analysis
[Signifier, affordance, feedback, conceptual model, constraint findings]

### Composability Assessment
[Granularity, orthogonality, extension points, surface area]

### Scores
| Lens | Score | Rationale |
|------|-------|-----------|
| JTBD | x.x / 10 | [one line — what pulls it up or down] |
| Jakob's Law | x.x / 10 | [one line] |
| DOET | x.x / 10 | [one line] |
| **Overall** | **x.x / 10** | [one line summary] |

Scores use figure-skating precision (e.g. 7.4, not just 7). Overall is a weighted read across all three — not an average. A surface that fails JTBD (jobs unserved) should score lower overall than one with minor Jakob's Law friction, even if the arithmetic says otherwise.

### Proposed Changes
[Only for real gaps — each with lens, layer, change, rationale, trade-offs]

### Verdict
[Honest summary across all three lenses — let the scores speak first, then explain]
```

End with a verdict. If the surface is well-designed across all three lenses, say so plainly.

---

## Error Handling

1. **Design mode, domain unclear**: Ask the user before proceeding. A design built on wrong assumptions is worse than no design.

2. **Audit mode, routing not found**: Grep for framework-specific patterns (`axum::Router`, `actix_web::web`, `router.get`, `app.use`, etc.).

3. **No ecosystem conventions are obvious**: Default to widely-used conventions for the language and protocol (REST conventions from Stripe/Twilio for HTTP APIs, idiomatic patterns from the standard library for the language).

4. **All gaps are minor**: Report them as such. A surface where all gaps are minor is well-designed. Don't inflate findings.

5. **Everything is well-designed**: Report that and stop. No manufactured findings.

6. **User asks for both design and audit**: Run Audit first, then produce Design-mode output only for the gaps — not a full redesign.

---

## Examples

### Design Example: SDK for a queue service

**Job**: "When I want to send a message from my app server, I want a single call that handles retries and confirms delivery, so I can move on."

**Ecosystem baseline** (Rust): `Result<T, E>` returns, builder pattern for config, `async fn` by default.

**Design**:

```rust
// Happy path — common case is short
let client = QueueClient::new("http://localhost:4000");
client.send("my-queue", "hello world").await?;

// Full control — advanced case adds ceremony proportionally
client.send_with_options("my-queue", payload)
    .delay(Duration::from_secs(30))
    .retry(RetryPolicy::exponential(3))
    .await?;
```

**DOET check**: `send` signals a single-shot fire. `send_with_options` signals configuration is available. The return type `Result<MessageId, QueueError>` makes failure visible and the message ID available for tracking. No silent failure.

**Jakob's Law check**: Builder pattern with `await?` is idiomatic Rust async. `RetryPolicy::exponential(3)` matches what `reqwest` and `tokio-retry` users already know.

---

### Audit Example: Signifier gap (DOET)

**Job**: "When I want to consume messages one at a time, I want to call a simple receive method."

**Surface**: `receive(queue, count, visibility_timeout, wait_seconds, condition)` — five required parameters, no defaults.

**Finding**: Weak signifiers. The method signals complexity before the developer understands what they need. The simple job requires understanding all five parameters first.

**Proposal**: Ergonomic defaults and a simpler overload. `receive(queue)` for the common case; full signature for advanced use. Progressive disclosure.

---

### Audit Example: Jakob's Law violation

**Surface**: Error responses are `{"message": "..."}`. Every major API in the ecosystem (Stripe, Twilio, AWS) uses a structured `code` field alongside `message`.

**Impact**: Developers handling specific error conditions must parse the human-readable string — brittle, breaks on rephrasing.

**Proposal**: Add a machine-readable `code` field. `message` stays for humans. Additive, non-breaking.

---

### Audit Example: JTBD gap — job not served

**Job**: "When a message fails repeatedly, I want to know about it so I can investigate, not lose it silently."

**Gap**: No dead-letter mechanism. No event on exhausted retries. No way to list failed messages.

**Proposal**: Dead-letter queue per queue. Messages that exhaust retries move to `{queue_name}_dlq`. `GET /v1/queues/{name}/dead-letters` lists them. Major gap — the job has no path to completion at any layer.

---

### Audit Example: Honest no-op

**Situation**: A small, focused API where every job maps cleanly to a surface method with idiomatic naming, structured errors, good defaults, and a conceptual model that matches the domain.

**Output**: All three lenses show green. Verdict: "The developer-facing surface keeps its promises across all three lenses. No changes recommended."

---

## Version History

- **v1.1.0** (2026-05-04): Added Design mode for greenfield API/SDK/CLI design; refactored Audit mode to be explicit; added progressive disclosure principle; added design examples
- **v1.0.0** (2026-05-04): Initial release — audit mode only
