---
description: Trajectory-level certificate — bound how many solvable tasks compression may cost
allowed-tools: Bash(distil *), Bash(uvx *)
---

Run distil's trajectory-level risk certificate for the user.

1. Ask which outcomes file to certify if not obvious (a JSONL of matched runs,
   one object per task: `{"task_id": ..., "full_success": bool, "compressed_success": bool}`,
   produced by running their eval suite twice — full context vs compressed).
2. Run it:
   ```bash
   distil certify-trajectories <outcomes.jsonl> --alpha 0.05 --delta 0.05
   ```
3. Report the certificate verbatim — including whether it is CERTIFIED, the
   observed degradation, the bound, and the assumptions statement. Never soften
   a NOT CERTIFIED result; explain what it means (collect more matched
   trajectories, or reduce compression aggressiveness) and that refusing to
   certify on thin evidence is the feature.

If the user has no outcomes file yet, explain the two-run recipe briefly and
point them at the docs: https://dshakes.github.io/distil/concepts.html#trajectory-certificate
