# Data notes

- Set 1 has 24 gaps > 15 min totalling 486 h (59% of the calendar span). First 43 files are 5-min
  intervals, the rest 10-min. Set 2 has none. Set 3 has 4 gaps (21 h).
- Set 1 RMS steps up ~20% at idx 156, coinciding with a 148 h gap (likely restart). Treated as a
  separate regime; excluded via burn_in=0.12.
- Onset rule: smoothed RMS / baseline > 1.25 for 20 consecutive snapshots (loose variant: 1.10).
- RUL cap: 50 h (median onset-to-end duration is 48.75 h in both calendar and operating time).
- Calendar vs operating onset-to-end hours: B3 11.7/4.9, B4 97.5/72.3, Set 2 56.5/56.5, Set 3 41.0/41.0.
- Labels use calendar hours; operating hours are used as an ablation.