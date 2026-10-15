---
description: Load full project context (PLAN.md, PROGRESS.md, PROPOSAL.md) and brief me on where the project stands
argument-hint: "[optional first request, e.g. what's next?]"
---

Load full context for this research project before doing anything else.

Using the Read tool, read these repo-root files in this order, in full:

1. **PLAN.md** — the live plan. Authoritative for what remains; its Status table is verdicts only.
2. **PROGRESS.md** — the module and its measured behaviour; authoritative for every number and its
   provenance. Its header notes that the long-form record it was distilled from (full protocol,
   corrections register) is parked at git tag `docs-full-20260930`.
3. **PROPOSAL.md** — the direction and the original predictions. Background only: where PLAN.md /
   PROGRESS.md disagree with it, they win (§2.1's geometric conditioning and §2.4's aux heads are
   measured inert).

CLAUDE.md is loaded automatically. If this session will run anything (training, eval, data
generation), also read NOTES.md's **Runbook** and **Traps** sections first.

Also run `git log --oneline -8` and flag any commits more recent than PROGRESS.md's *Last updated*
date — results that may not be written up yet.

Then give a compact briefing — at most ~10 lines, no re-dumping: the headline results, the most
recent experiment's outcome, and what's next per PLAN.md's *Future work*.

If the command was invoked with extra arguments, they are the first request to address once the
briefing is done: $ARGUMENTS
