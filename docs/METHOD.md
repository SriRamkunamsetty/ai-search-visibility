# Method: AI-Search Visibility

## Measures
- Each prompt is asked `n_samples` times and cached by (title, model, prompt, sample).
- Release date: correct (exact date present), partial (month-year only), wrong, unknown. Cast recall = ground-truth names found. "Possible fabricated names" = capitalised name-like strings not in the ground truth (heuristic, false positives expected).
- Category lists: mention rate (Wilson 95% CI), average rank among tracked titles, share of voice = title mentions / all tracked-title mentions.
- Ground truth must be entered by you in `title_facts` (release_date, cast, director, alias).

## Limits you should state in any write-up
- Answers depend on the model, the date and the phrasing. Use at least 5 samples per prompt for anything you report; the intervals are wide on purpose when n is small.
- "Possible fabricated names" is a capitalisation heuristic, not entity recognition: expect false positives (and some misses).
- Ground truth is whatever you enter in `import-facts`; the tool cannot verify it.
- The demo's "model" is a seeded simulator. Real runs use `aivisibility run` with your own key and cost API calls (it prints the count and requires `--yes`).
