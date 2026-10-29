---
name: jtbd
description: Jobs-to-be-Done developer experience audit. Derives the real jobs users hire this service to do, maps them to the full developer-facing surface (REST API, SDKs, client libraries, CLI, config, error messages, docs), identifies genuine gaps, and proposes concrete improvements — only where gaps exist. Use when asked to review API ergonomics, SDK design, developer experience, missing features, or what users actually need from this service. Trigger with "jtbd", "jobs to be done", "api audit", "sdk audit", "dx review", "what are we missing", "ergonomics review".
allowed-tools: "Read,Glob,Grep,Bash(git log:*),Bash(git diff:*),Bash(cargo:*),Bash(mise:*)"
version: "1.1.0"
---

# Jobs-to-be-Done Developer Experience Audit

Derives what users actually hire this service to accomplish, maps those jobs to the full developer-facing surface, and surfaces only the gaps worth addressing.

## The Jobs-to-be-Done Framework

**Core idea**: People don't adopt tools because of features. They hire tools to make progress on something they're trying to accomplish — and they fire them when something else does the job better. A "job" is the progress a person is trying to make in a particular situation.

This reframes the question from "what does our API offer?" to "what are developers trying to accomplish, and does our surface help them get there?"

### Anatomy of a job

A well-formed job has three parts:

- **Situation** — the trigger or context that causes someone to need this
- **Motivation** — what they're trying to accomplish (the job itself)
- **Outcome** — the progress they're trying to make; how they'll know the job is done

Format: _"When [situation], I want to [motivation], so I can [outcome]."_

Jobs operate at multiple levels:

| Dimension      | Question                              | Example                                                                  |
| -------------- | ------------------------------------- | ------------------------------------------------------------------------ |
| **Functional** | What are they literally trying to do? | Verify that a request came from an authenticated user                    |
| **Emotional**  | How do they want to feel?             | Confident the auth check won't silently fail                             |
| **Social**     | How do they want to be perceived?     | Like someone who ships secure systems, not someone who gets paged at 3am |

### Jobs vs features vs user stories

A **job** is stable and solution-agnostic. It exists whether or not your service exists.
A **feature** is a proposed solution to a job. It belongs in the proposal, not the job statement.
A **user story** ("As a developer, I want X") often smuggles in a solution. A job statement stays at the level of intent.

Bad (feature-shaped): "When I call the API, I want a `200 OK` with the user object."
Good (job-shaped): "When a request arrives at my server, I want to know who made it and whether they're allowed, so I can gate access to protected resources."

### What to look for

Jobs are usually **stable** — they change slowly. A job that existed three years ago probably still exists. Look for evidence of jobs in:

- Route names and what actions they expose
- Migration history (tables added reveal features added, which reveal jobs served)
- Error types (what can go wrong reveals what the system is trying to do)
- Test descriptions ("when user is unauthenticated", "when MFA is required")
- Documentation and comments

### The job executor vs the job beneficiary

In a developer-facing service, these are often different:

- **Job executor**: the developer or operator who integrates the service
- **Job beneficiary**: the end user whose authentication experience the service enables

Frame primary jobs from the **executor's** perspective. Executor jobs are where the API surface lives. Beneficiary jobs provide context for why the executor cares.

## Overview

This skill applies the Jobs-to-be-Done framework as a lens on developer experience. "API" means the entire surface a developer touches: REST endpoints, SDKs and client libraries, CLI tools, configuration schema, error messages, webhook payloads, and integration patterns. It audits whether the jobs the service was built to serve are actually well-served across all of these touchpoints. Suggestions only emerge from real gaps between stated jobs and what the current surface offers.

The output is honest: if everything serves the identified jobs well, say so and stop. Only surface changes that close a real gap.

## Instructions

### Step 1: Orient to the Project

Read the following files to understand what this service is and what it is meant to do:

- `CLAUDE.md` — service model, constraints, design philosophy
- `ARCHITECTURE.md` — if present, for domain context
- `README.md` or `README` — if present
- `src/routes/` or equivalent routing directory — the API surface
- `src/lib.rs` or main entry point — to understand the domain model
- `migrations/` — to understand the data model and history
- `Cargo.toml` or equivalent — to understand dependencies and tech choices

Do not read every file. Read enough to build a confident picture of:

- What domain this service operates in
- Who calls it (operators, end users, machines)
- What actions it exposes
- What data it manages
- What SDKs, client libraries, or CLI tools exist (look for `clients/`, `sdk/`, `cli/`, or sibling repos referenced in docs)

### Step 2: Enumerate the Full Developer-Facing Surface

Map everything a developer touches when integrating this service:

**REST API**: Method + path + brief description. Compact table.

**SDKs / client libraries**: What languages? What level of abstraction — thin HTTP wrapper or higher-level? Are there typed request/response models? Do they handle auth, retries, pagination?

**CLI**: If present, what commands exist? What jobs does it serve?

**Configuration**: What does an operator configure to deploy this? Is configuration ergonomic — env vars, config files, secrets injection?

**Error surface**: What do error responses look like? Are they actionable? Do they include enough context to debug?

**Webhooks / events**: If the service emits events, what's the payload shape? What jobs do they serve?

**Documentation / onboarding**: What does a new integrator read first? Is there an ARCHITECTURE.md, OpenAPI spec, or examples directory?

Note what is absent. Missing SDK, no CLI, undocumented config — these are surface gaps, not just REST gaps.

### Step 3: Derive the Jobs to Be Done

Using the framework defined above, derive the **primary jobs** this service exists to perform. Write each as:

> "When [situation], I want to [motivation], so I can [outcome]."

Group them by functional area (e.g., Identity, Sessions, Authorization, Tokens, Administration). For each job, note its functional dimension and, where relevant, the emotional or social dimension — these reveal where friction hurts most.

Aim for 8–15 jobs. Every job must be grounded in evidence from the codebase — routes, migrations, test descriptions, error types, or documentation. Do not invent jobs that have no signal in the code. Do not write feature requests dressed up as jobs.

### Step 4: Map Jobs to the Full Surface

For each job, assess how well the current developer-facing surface serves it — not just the REST API:

| Job                 | Served by                              | Surface layer                    | Gap?                 |
| ------------------- | -------------------------------------- | -------------------------------- | -------------------- |
| When X, I want to Y | `POST /v1/...` or SDK method or config | REST / SDK / CLI / Config / Docs | None / Minor / Major |

A gap is:

- **None** — the job is served cleanly at an appropriate layer
- **Minor** — the job is served but with friction (extra round-trips, missing SDK abstraction, raw HTTP required where a helper would be natural, cryptic errors)
- **Major** — the job is not served at any layer, or requires a workaround that degrades reliability/safety/correctness

Consider layer-appropriate expectations: a developer should not need to hand-roll JWT parsing if an SDK exists; an operator should not need to read source code to understand config options; an integrator should not get a 500 with no body when they make a mistake.

### Step 5: Propose Changes — Only for Real Gaps

For each Minor or Major gap, propose a concrete improvement at the right layer. A gap in the REST API might be fixed there. A gap in ergonomics might call for an SDK helper, a better error message, or a config default — not a new endpoint.

Follow the service's design constraints:

- REST layer: resource orientation (nouns + HTTP verbs, not RPC), the service model (internal, single-project, not public SaaS)
- SDK layer: idiomatic for the target language, thin over the REST API unless abstraction pays for itself
- Config layer: sensible defaults, clear names, documented in one place
- Error layer: actionable, include what went wrong and what to try

Each proposal must include:

- What job it closes
- Which layer the change lives at (REST, SDK, CLI, config, docs, error messages)
- The specific change (new endpoint, new method, new field, changed response shape, new config key)
- Why the existing approach is insufficient
- Any trade-offs

If a gap is cosmetic or the existing approach is defensible, say so and omit it from proposals.

### Step 6: Deliver the Report

Structure the output as follows:

```
## Jobs-to-be-Done Audit: [Service Name]

### Developer-Facing Surface (current)
[REST endpoints — compact table]
[SDKs / client libraries — what exists, what layer of abstraction]
[CLI — if present]
[Config — key options and how they're set]
[Error surface — shape and actionability]
[Docs / onboarding — what exists]

### Jobs to Be Done
[Grouped list with situation/motivation/outcome framing]

### Gap Analysis
[Job → coverage table, with surface layer column]

### Proposed Changes
[Only if gaps exist — concrete, actionable proposals with layer called out]

### Verdict
[Honest summary: well-served, partially served, or specific areas needing work]
```

End with a verdict. If the API is well-designed for its jobs, say so plainly.

## Output

- A structured audit report delivered as a response (not written to a file unless asked)
- Gap analysis mapping jobs to API coverage
- Concrete proposals, scoped only to genuine gaps
- A verdict

## Error Handling

1. **Routing files not found**: Grep for `axum::Router`, `actix_web::web`, `router.route`, or equivalent to locate the routing layer. Adapt to the framework in use.

2. **Domain is unclear**: Read more of the source — entity types, error variants, and migration history are strong signals. The data model reveals the domain.

3. **Too many endpoints to enumerate**: Focus on functional groupings rather than listing every route. The job-level analysis does not require exhaustive enumeration.

4. **No ARCHITECTURE.md**: Derive the architecture from the routing layer, data model, and Cargo.toml dependencies. Do not manufacture a document — work from what exists.

5. **All jobs are well-served**: Report that and stop. Do not invent gaps to fill the report.

## Examples

### Example 1: SDK gap, not a REST gap

**Job identified**: "When I verify a token in my app server, I want to validate it without implementing JWT logic myself."

**REST surface**: `POST /v1/tokens/introspect` exists and works.

**SDK surface**: No SDK. Developers hand-roll the HTTP call, parse the response, handle errors manually.

**Gap**: Minor at REST layer (endpoint exists), Major at SDK layer (no wrapper). The job requires HTTP fluency that shouldn't be a prerequisite.

**Proposal**: Thin SDK client in the most common server language(s) used by integrators. `client.tokens.introspect(token)` returning a typed result. Not a REST change.

### Example 2: Error surface gap

**Job identified**: "When my integration is misconfigured, I want to know what's wrong so I can fix it fast."

**Gap**: Service returns `{"error": "unauthorized"}` with no further context when a signing key is wrong, expired, or missing. Integrator must diff their config against docs to find the issue.

**Proposal**: Structured error responses with a `code` field (e.g., `signing_key_not_found`, `token_expired`) and a `detail` string. No REST surface change required — just enrich the error body. Discuss: avoid leaking sensitive internals in detail strings.

### Example 3: Missing REST capability

**Job identified**: "When an admin needs to audit recent authentication events for a user, I want to retrieve a session history."

**Gap**: No session history or audit log endpoint exists. Sessions table tracks active sessions only.

**Proposal**: `GET /v1/users/{id}/sessions?include_revoked=true` or a separate `GET /v1/audit/sessions` resource with filtering. Discuss trade-offs (storage cost, sensitivity of data).

### Example 4: Honest no-op

**Situation**: Small, focused service where every job maps cleanly to an existing endpoint or SDK method with good ergonomics.

**Output**: Jobs table shows all green across REST, SDK, config, and errors. Verdict: "The developer-facing surface matches the jobs well. No changes recommended."

## Version History

- **v1.1.0** (2026-04-28): Broadened surface coverage to include SDKs, CLI, config, error messages, and docs — not just REST endpoints
- **v1.0.0** (2026-04-28): Initial release
