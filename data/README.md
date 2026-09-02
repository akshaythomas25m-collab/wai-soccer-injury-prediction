# Data

The four source files are **not committed**. `player_profiles.csv` alone is 26 MB, over GitHub's 25 MB web-upload limit, and committing datasets you did not create is poor practice in any case.

## What you need

Place these four files in a Google Drive folder named `wai_data` (or in a local `data/` folder if running outside Colab). The loader globs on the filename, so the primary file's exact spelling does not matter — the other three must match exactly.

| File | Size | Rows | Role |
|---|---|---|---|
| `full_dataset_thesis*.csv` | 1.6 MB | 15,603 | **Primary.** European football injuries, 2020–2025. One row per injury: season, injury name, days out, games missed, start/end dates, player name, age, position, club, league. |
| `player_injuries.csv` | 8.0 MB | 143,195 | Transfermarkt career injury history, 1973–2025, keyed on `player_id`. Source of the injury-history features. |
| `player_profiles.csv` | 26 MB | 92,152 | Transfermarkt player profiles. Supplies `player_id`, height, preferred foot and main position, and is the bridge between player names and `player_id`. |
| `player_injuries_impact.csv` | 0.2 MB | — | Premier League injuries and team performance. Context only; not used in the model. |

## Linkage

The primary dataset and Transfermarkt are joined on player name. Transfermarkt names carry a trailing id in parentheses — `Alexander Nübel (10)` — which must be stripped:

```python
df['player_name_clean'] = (df['player_name']
    .str.replace(r'\s*\(\d+\)$', '', regex=True)
    .str.strip()
    .str.lower())
```

Without the escaped parentheses in that pattern the match rate is **0%**. With them it is **97.2%**.

## A caution about the history features

`player_injuries.csv` spans 1973–2025 and therefore overlaps the 2020–25 window the model predicts. Aggregating it per player without a date cut-off leaks the target — see the leak section in the top-level README and Section 9 of the assumptions report. Phase 5b rebuilds these features with `pd.merge_asof(..., allow_exact_matches=False)` so that only strictly earlier injuries contribute.
