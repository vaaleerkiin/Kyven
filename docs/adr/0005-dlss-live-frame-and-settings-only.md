# ADR 0005: DLSS Live runs only for frame and own-setting changes

- Status: Accepted; supersedes the upstream-watching Live policy in ADR 0002
- Date: 2026-11-15

## Decision

After its initial preview, DLSS Live may request inference only when the current frame changes
or a processing setting inside the DLSS node changes. The owner explicitly requested this narrower
policy after repeated self-triggered cancellations/restarts on an unchanged frame.

Remove upstream hash evaluation, forced validation, connection identities and runtime/file metadata
from the Live signature. Ignore inputChange notifications, rewires, status, cache Read updates and
model refresh for scheduling. Do not change the history epoch on rewires. Keep the attempted intent
after failures/internal cancellation so those events cannot retry an unchanged frame/settings pair.
Only cancellation caused by a genuine newer frame/settings intent permits automatic replacement.

Keep debouncing, one active request per node, cooperative cancellation, stale-publication rejection
and reentry guards. Own export callbacks must not observe Nuke's temporary last_frame + 1 context.
Explicit Process Current Frame/Frame Range remain available for manual refresh. This policy changes
host scheduling only; backend content hashes and deterministic output/history keys remain in force.

## Consequence

Editing or reconnecting an upstream node on the same frame requires Process Current Frame, a new
timeline frame or a DLSS processing-setting edit. Output display and debounce controls do not invoke
inference. Mask Alpha Input always uses the connected mask's alpha channel A, with no channel selector.
