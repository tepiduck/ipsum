# Verdict JSON, schema version 1

`faultline check` writes `verdict.json`, `evidence.json`, and `report.html` in its output
directory. [verdict.schema.json](verdict.schema.json) is a JSON Schema 2020-12 contract.
JSON numbers are finite; indeterminate/infinite advisory values use explicit status fields.

| Field | Contract |
|---|---|
| `schema_version`, `policy_version` | Both 1 for this release; reject unsupported versions in strict consumers. |
| `verdict`, `exit_code` | SAFE/0, REGRESSION/1, NEEDS_REVIEW/2, INCONCLUSIVE/3. |
| `advisory_only` | True; existing primary outcomes and command exit codes are unchanged. |
| `execution_state` | `running` for a provisional checkpoint, `complete` when the command has finished its planned work (including inconclusive outcomes). |
| `scope` | Mode, level, planned/completed primary batch counts, gradient coverage, limitation statement. |
| `reasons`, `notes` | Arrays of `code`, `message`, `batch` (nullable), optionally `side`. Use codes for automation. |
| `accuracy_shift_replicated` | Batch counts, distribution, policy, sign counts and flag, or an unavailable/not-applicable status. Additive evidence fields are allowed. |
| `injected_fault` | True if any underlying execution used an explicitly controlled fault. |
| `artifacts` | Paths relative to the check directory: report, evidence, and exported reproducer scripts. |
| `metrics` | Optional invocation/command wall seconds and maximum supervised worker-tree RSS in MiB. Resume invocation time excludes earlier invocations. |

`evidence.json` links the summary to its configuration and per-batch enriched observations.
Original `primary/outcome.json` files remain unchanged. Diagnostic observations, limits,
stderr and stdout live beside them. A commit manifest hashes the completed JSON evidence;
resume rejects altered evidence, settings, inputs, source, or interpreter identity.

Reason codes include `candidate_only_failure`, `accuracy_shift_replicated`,
`contrary_evidence`, `unexplained_failure`, `replication_unavailable`, `missing_batches`,
`setup_error`, and `baseline_<primary_status>`/`candidate_<primary_status>`.
Consumers should display unknown reason codes instead of treating them as success.
`completed_batches` counts conclusive primary pairs, not completed fp64 diagnostics.

The report is written after each committed batch. A provisional verdict can change as more
batches arrive. Wait for `execution_state: complete` and the process exit before a CI decision;
completion does not imply sufficient coverage or a conclusive verdict.
