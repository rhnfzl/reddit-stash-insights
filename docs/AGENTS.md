# Evaluation publication boundary

Publish reusable evaluation code, metric definitions, tests and fictional examples.
Keep user archives and every derived artifact local, including questions, answers,
quotes, labels, snapshots, indexes, traces, aggregate results, corpus fingerprints,
hardware measurements, plans and case studies.

Removing raw text does not authorize publication of a user-derived report.
Do not put personal evaluation material in this documentation tree, CI artifacts,
release assets or package distributions. Do not force-add ignored evaluation files.

Use an external private directory or the ignored `data/eval/` directory.
Keep private human HTML reports under `data/eval/local-review/docs/human-html/`.
This privacy boundary overrides the usual public `docs/human-html/` location.

Local file storage and permission to use hosted models are separate decisions.
Do not upload private data to a judge or training service without user authorization.

Before staging documentation, run:

```sh
python scripts/check_eval_privacy.py --working-tree
```

The checker catches recognized evaluation formats and paths. It does not replace
checking the origin of new examples. Only the packaged fictional fixture is public data.
