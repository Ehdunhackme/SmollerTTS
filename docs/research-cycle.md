# First bounded research cycle

This records the continuation policy before the first candidate's development
result is available. It is an experiment horizon, not an instance billing limit.

The incumbent is the finished width-128 baseline, seed 17, 50,000 steps. Each
candidate starts from random initialization, uses the same five-language batches
and step budget, and changes one setting. Training time is measured separately.
The frozen development panel and promotion tolerances stay unchanged.

1. Width 144, 948,164 parameters: test whether extra shared acoustic capacity helps.
2. If it is not nominated, learning rate 0.0005 at baseline width: test whether a
   smaller update size improves acoustic fitting at the same step budget.
3. If neither is nominated, waveform-loss weight 0.30 at baseline width and
   learning rate: test stronger acoustic reconstruction supervision. Reject it if
   improved reconstruction does not improve the frozen text-input scores.

Each unsuccessful candidate is retained with its results and rejection reasons.
Stop this initial search after a provisional nomination or after these three
candidates. A failed run is diagnosed, never counted as a completed comparison.

For a nominee, retrain both its recipe and the baseline at seeds 41 and 73, with
the same 50,000 steps. Report all three seeds, including seed 17, per language.
Do not pick the best random seed. Require the existing promotion rule on the
per-language mean errors and no individual paired seed/language regression above
0.02 or invalid fraction above 5%. Otherwise retain the baseline. If no recipe
was nominated, repeat the baseline at seeds 41 and 73 to characterize variability.
This small seed count is exploratory evidence, not a statistical superiority claim.

Prepare a blinded development listening comparison for the resulting finalists
and the released external voice reference. Report the user's ratings as author
feedback. Keep subjective feedback separate from the frozen automatic scores;
do not change either to manufacture a win. Any severe quality problem is reported
and can motivate a separately declared future research cycle.

Freeze the chosen recipe, evidence and source/data identities. Train its final
50,000-step model from scratch at the predeclared seed 101. Use its last checkpoint,
not a checkpoint chosen by final-test performance. Evaluate all reserved test
sentences exactly once with the same recognition settings, normalization and
failure accounting, using a separate final-test entry point. No test text may be
used to augment pronunciation dictionaries or modify the model after selection.

The final delivery includes that checkpoint, export, matching browser assets,
complete data/experiment provenance, all language results and failures, audio,
listening feedback, measured CPU behavior, supporting asset sizes and limitations.
The existing browser demo and development comparisons are intermediate evidence.
