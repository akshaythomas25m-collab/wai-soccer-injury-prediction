# ============================================
# PHASE 9: Save the Model + Live Web Simulator
# ============================================
# Two jobs:
#   1. Persist the corrected XGBoost model to Drive so it survives a runtime
#      recycle — otherwise the demo needs a 15-minute retrain every time.
#   2. Launch a Gradio web app on a public URL, for screen recording.
#
# Tab 1  Simulate an injury — set the attributes, get a severity prediction
#                             plus the factors that drove that prediction.
# Tab 2  Look up a player   — search any of the 4,081 players, see their
#                             record, and score a hypothetical new injury.
#
# Run AFTER Phase 5b — needs xgb_h, Xh_train, df_merged.

!pip install -q gradio joblib

import os, joblib
import numpy as np
import pandas as pd
import gradio as gr

# ------------------------------------------------------------ 1. persist
MODEL_DIR = '/content/drive/MyDrive/wai_models'
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump({'model': xgb_h,
             'columns': list(Xh_train.columns),
             'template': Xh_train.median(),
             'metrics': {'auc': 0.7472, 'recall': 0.7039, 'precision': 0.5257}},
            f'{MODEL_DIR}/xgb_severe_injury.joblib')
print(f"💾 Model saved to {MODEL_DIR}/xgb_severe_injury.joblib")

try:
    import shap
    EXPLAINER = shap.TreeExplainer(xgb_h)
    print("🧠 SHAP explainer ready — per-prediction factors enabled")
except Exception as e:
    EXPLAINER = None
    print(f"⚠️  SHAP unavailable ({type(e).__name__}) — app runs without the factor panel")

# ------------------------------------------- 2. read the levels off the data
# Everything the dropdowns offer is derived from the fitted design matrix, so
# the app can never present a level the model was not trained on.
TEMPLATE = Xh_train.median()
COLS     = list(Xh_train.columns)

def levels(col):
    """Every level of a categorical, including the reference level that
    drop_first removed from the design matrix (it encodes as all-zeros)."""
    return sorted(df_merged[col].dropna().astype(str).unique().tolist())

CATEGORIES = levels('injury_category')
LEAGUES    = levels('league')
PHASES     = [p for p in ['early', 'mid', 'late', 'off_season'] if p in levels('season_phase')]

# 14 position labels, one of which ("Midfielder") appears a single time —
# a data-quality artefact. Offer only positions with a real sample behind them.
_pc = df_merged['player_position'].value_counts()
POSITIONS = sorted(_pc[_pc >= 30].index.astype(str).tolist())

def pretty(v):
    return str(v).replace('_', ' ').replace('/', ' / ').capitalize()

CAT_LABEL = {pretty(c): c for c in CATEGORIES}
POS_LABEL = {pretty(p): p for p in POSITIONS}
PHZ_LABEL = {pretty(p): p for p in PHASES}

PRETTY = {'is_knee': 'knee injury', 'is_illness': 'illness',
          'is_muscular': 'muscular injury', 'player_age': 'player age',
          'is_over_30': 'aged over 30', 'height': 'height',
          'injuries_in_dataset': 'injuries on record', 'season_num': 'season',
          'total_injuries_prior': 'prior injuries',
          'avg_days_missed_prior': 'average past layoff',
          'max_days_missed_prior': 'longest past layoff',
          'total_games_missed_prior': 'games missed previously',
          'distinct_injury_types_prior': 'variety of past injuries'}

DUMMY_PREFIXES = ('injury_category_', 'league_', 'player_position_',
                  'season_phase_', 'age_group_', 'foot_')
FLAGS = ('is_knee', 'is_illness', 'is_muscular', 'is_over_30')

def label(col):
    if col in PRETTY:
        return PRETTY[col]
    for p, w in [('injury_category_', 'injury type: '), ('league_', 'league: '),
                 ('player_position_', 'position: '), ('season_phase_', 'season phase: '),
                 ('age_group_', 'age band: '), ('foot_', 'preferred foot: ')]:
        if col.startswith(p):
            return w + pretty(col[len(p):]).lower()
    return col.replace('_', ' ')

def factor_label(col, value):
    """A one-hot sitting at zero still carries SHAP weight, and it is real:
    'this is not a fracture' genuinely lowers the risk. But printing it as
    'injury type: fracture' next to a knee injury reads as a contradiction,
    so a switched-off feature is named for what it is — an absence."""
    base = label(col)
    if value == 0 and (col.startswith(DUMMY_PREFIXES) or col in FLAGS):
        return 'not ' + base.split(': ')[-1]
    return base

def age_band(age):
    # matches the pd.cut bins in Phase 2: (0,21] (21,25] (25,30] (30,35] (35,50]
    if age <= 21: return 'U21'
    if age <= 25: return '21-25'
    if age <= 30: return '26-30'
    if age <= 35: return '31-35'
    return '35+'

# ------------------------------------------------------------ 3. build a row
def build_row(category, age, league, position, phase, prior_injuries):
    """Start from the median training row, then set what the user chose.

    Two things matter for this to be honest:
      • each one-hot group is zeroed before one level is set, so exactly one
        level is hot — and the reference level that drop_first removed is
        correctly represented as all-zeros;
      • the history aggregates are kept CONSISTENT with the prior-injury
        count. Leaving 'average past layoff' at its median while the user
        sets prior injuries to zero would feed the model a row that cannot
        exist, and the prediction would be meaningless.
    """
    row = TEMPLATE.copy()
    age, n = int(age), int(prior_injuries)

    scalars = {'player_age': age,
               'is_over_30': int(age >= 30),
               'is_muscular': int(category == 'muscular'),
               'is_knee': int(category == 'knee'),
               'is_illness': int(category == 'illness'),
               'total_injuries_prior': n,
               # a player with n priors has at least n+1 injuries in the window
               'injuries_in_dataset': n + 1}

    if n == 0:
        # no history at all — every history aggregate must be zero
        for c in ['avg_days_missed_prior', 'max_days_missed_prior',
                  'total_games_missed_prior', 'distinct_injury_types_prior']:
            scalars[c] = 0.0
    else:
        base = max(float(TEMPLATE.get('total_injuries_prior', 1)), 1.0)
        per  = float(TEMPLATE.get('total_games_missed_prior', 0)) / base
        scalars['total_games_missed_prior'] = per * n
        scalars['distinct_injury_types_prior'] = min(
            float(TEMPLATE.get('distinct_injury_types_prior', 1)) or 1.0, float(n))

    for col, val in scalars.items():
        if col in row.index:
            row[col] = val

    for prefix, chosen in [('injury_category_', category), ('league_', league),
                           ('player_position_', position), ('season_phase_', phase),
                           ('age_group_', age_band(age))]:
        for c in [c for c in row.index if c.startswith(prefix)]:
            row[c] = 0.0
        if f'{prefix}{chosen}' in row.index:
            row[f'{prefix}{chosen}'] = 1.0

    return pd.DataFrame([row])[COLS].astype(float)

def top_factors(X, k=4):
    """The features that moved this one prediction the most."""
    if EXPLAINER is None:
        return ''
    try:
        sv = EXPLAINER.shap_values(X)
        if isinstance(sv, list): sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3: sv = sv[:, :, -1]
        s = pd.Series(sv[0], index=X.columns)
        s = s.reindex(s.abs().sort_values(ascending=False).index)
        s = s[s.abs() > 0.01][:k]
        if s.empty:
            return ''
        vals = X.iloc[0]
        rows = ''.join(
            f'<div class="frow"><span>{factor_label(i, vals[i])}</span>'
            f'<b class="{"up" if v > 0 else "down"}" style="white-space:nowrap">'
            f'{"↑ raises" if v > 0 else "↓ lowers"} risk</b></div>'
            for i, v in s.items())
        return ('<div style="margin-top:16px">'
                '<div class="dim" style="margin-bottom:4px">'
                f'WHAT DROVE THIS PREDICTION</div>{rows}</div>')
    except Exception:
        return ''

# ------------------------------------------------------------ 4. presentation
# The card is a light panel that has to stay readable inside Gradio's DARK
# theme as well. Gradio's dark CSS recolours plain text inside injected HTML,
# so every text element is given an explicit colour with !important rather
# than inheriting one from the wrapper — otherwise half the card goes invisible.
CARD_STYLE = """<style>
.wai-card{font-family:ui-sans-serif,system-ui;padding:20px;border:1px solid #ddd;
 border-radius:8px;background:#fff}
.wai-card,.wai-card div,.wai-card span,.wai-card b{color:#141A17 !important}
.wai-card .dim{color:#666 !important;font-size:11.5px;letter-spacing:.14em}
.wai-card .faint{color:#999 !important;font-size:11px}
.wai-card .muted,.wai-card .muted b{color:#555 !important;font-size:12.5px;line-height:1.55}
.wai-card .band{color:var(--band) !important}
.wai-card .up{color:#BE4A10 !important}
.wai-card .down{color:#0C8F80 !important}
.wai-card .pct{font-size:56px;font-weight:600;line-height:1.1}
.wai-card .bandname{font-size:15px;font-weight:600;letter-spacing:.04em}
.wai-card .head{margin-top:12px;font-size:15px;line-height:1.6}
.wai-card .prof{margin-top:12px;font-size:14px;line-height:1.75;padding:12px 14px;
 background:#F7F9F8;border-radius:6px}
.wai-card .frow{display:flex;justify-content:space-between;gap:12px;padding:5px 0;
 border-bottom:1px solid #f0f0f0;font-size:13.5px}
.wai-card hr{border:none;border-top:1px solid #eee;margin:16px 0 12px}
</style>"""

CAVEAT = """<hr><div class="muted">
<b>This predicts severity given an injury has occurred — not whether an injury will happen.</b><br>
On held-out data (1,021 players never seen in training) the model catches <b>70%</b> of
severe cases, but <b>nearly half of what it flags is a false alarm</b> — AUC 0.747,
precision 0.526. A triage aid, not a clinical decision, and an association in historical
data rather than a causal claim.</div>"""

def card(p, headline, extra=''):
    if   p >= 0.60: band, colour = 'HIGH RISK',     '#BE4A10'
    elif p >= 0.49: band, colour = 'ELEVATED RISK', '#8A6512'
    else:           band, colour = 'LOWER RISK',    '#0C8F80'
    return f"""{CARD_STYLE}<div class="wai-card" style="--band:{colour}">
<div class="dim">PROBABILITY THIS INJURY LASTS 28+ DAYS</div>
<div class="pct band">{p*100:.0f}%</div>
<div class="bandname band">{band}</div>
<div style="height:8px;background:#eee;border-radius:4px;overflow:hidden;margin:10px 0 4px">
<div style="height:100%;width:{p*100:.0f}%;background:{colour}"></div></div>
<div style="display:flex;justify-content:space-between"><span class="faint">0%</span>
<span class="faint">49% — decision threshold</span><span class="faint">100%</span></div>
<div class="head">{headline}</div>
{extra}{CAVEAT}</div>"""

# ------------------------------------------------------------ 5. tab 1
def simulate(cat_label, age, league, pos_label, phase_label, prior_injuries):
    category = CAT_LABEL.get(cat_label, CATEGORIES[0])
    position = POS_LABEL.get(pos_label, POSITIONS[0])
    phase    = PHZ_LABEL.get(phase_label, PHASES[0])
    X = build_row(category, age, league, position, phase, prior_injuries)
    p = float(xgb_h.predict_proba(X)[0, 1])
    verdict = ('likely to run past 28 days' if p >= 0.49
               else 'likely to resolve inside 28 days')
    head = (f'A <b>{pretty(category).lower()}</b> injury to a {int(age)}-year-old '
            f'{pos_label.lower()} in the {league}, picked up in the '
            f'{phase_label.lower()} part of the season, is <b>{verdict}</b>.')
    return card(p, head, top_factors(X))

# ------------------------------------------------------------ 6. tab 2
PLAYERS = sorted(df_merged['player_name_clean'].dropna().astype(str).unique().tolist())

def lookup(name):
    blank = ('<div style="font-family:ui-sans-serif,system-ui;padding:20px;color:#666">'
             'Type a name to search {:,} players.</div>'.format(len(PLAYERS)))
    if not name:
        return blank
    h = df_merged[df_merged['player_name_clean'].astype(str) == str(name)]
    if h.empty:
        return blank

    usual    = h['injury_category'].mode().iloc[0]
    age      = int(h['player_age'].max())
    league   = h['league'].mode().iloc[0]
    position = h['player_position'].mode().iloc[0]
    n        = len(h)
    med      = h['days_numeric'].median()
    worst    = h['days_numeric'].max()
    sev      = h['severe_injury'].mean()

    # fall back to an offered level if this player's own is not in the dropdowns
    if usual    not in CATEGORIES: usual    = CATEGORIES[0]
    if league   not in LEAGUES:    league   = LEAGUES[0]
    if position not in POSITIONS:  position = POSITIONS[0]

    X = build_row(usual, age, league, position, 'mid', n)
    p = float(xgb_h.predict_proba(X)[0, 1])

    profile = f"""<div class="prof">
<b style="font-size:15px">{str(name).title()}</b> — {position}, {league}<br>
<b>{n}</b> {'injuries' if n != 1 else 'injury'} on record ·
most often <b>{pretty(usual).lower()}</b><br>
median <b>{med:.0f}</b> days out · worst <b>{worst:.0f}</b> days ·
<b>{sev*100:.0f}%</b> of them were severe</div>"""

    head = (f'If {str(name).title()} picks up a new <b>{pretty(usual).lower()}</b> '
            f'injury mid-season, the model scores it as shown.')
    return card(p, head + profile, top_factors(X))

# ------------------------------------------------------------ 7. the app
CSS = '.gradio-container{max-width:1150px!important}'
STYLE = {'css': CSS, 'theme': gr.themes.Soft(primary_hue='teal')}
# Gradio 6 moved theme/css from Blocks() to launch(); this runs on either.
_G6 = int(gr.__version__.split('.')[0]) >= 6
BLOCK_KW, LAUNCH_KW = ({}, STYLE) if _G6 else (STYLE, {})

def first(options, *preferred):
    """Open the demo on a recognisable default rather than whatever sorts first."""
    for p in preferred:
        if p in options:
            return p
    return list(options)[0]

with gr.Blocks(title='Severe Injury Risk Simulator', **BLOCK_KW) as demo:
    gr.Markdown(
        '# ⚽ Severe Injury Risk Simulator\n'
        'Will a football injury keep a player out for **28 days or more**?  \n'
        '<sub>XGBoost · 15,603 injuries · 4,081 players · 5 European leagues · '
        '2020–2025 · leakage-corrected</sub>')

    with gr.Tab('Simulate an injury'):
        with gr.Row():
            with gr.Column(scale=1):
                c   = gr.Dropdown(list(CAT_LABEL), value=pretty('knee'),
                                  label='Injury category')
                a   = gr.Slider(16, 40, value=27, step=1, label='Player age')
                l   = gr.Dropdown(LEAGUES, label='League',
                                  value=first(LEAGUES, 'Premier League'))
                pos = gr.Dropdown(list(POS_LABEL), label='Position',
                                  value=first(POS_LABEL, 'Centre-forward',
                                              'Centre-back', 'Central midfield'))
                s   = gr.Dropdown(list(PHZ_LABEL), label='Point in the season',
                                  value=first(PHZ_LABEL, 'Mid', 'Early'))
                n_  = gr.Slider(0, 20, value=2, step=1,
                                label='Injuries already on this player\'s record')
                btn = gr.Button('Predict', variant='primary')
            with gr.Column(scale=1):
                o1 = gr.HTML()
        ins = [c, a, l, pos, s, n_]
        btn.click(simulate, ins, o1)
        for w in ins:
            w.change(simulate, ins, o1)      # live — updates as you move a slider
        demo.load(simulate, ins, o1)

    with gr.Tab('Look up a player'):
        with gr.Row():
            with gr.Column(scale=1):
                pl = gr.Dropdown(PLAYERS, value=PLAYERS[0], label='Player',
                                 filterable=True,
                                 info=f'Type to search {len(PLAYERS):,} players')
                gr.Markdown(
                    '<sub>Shows what this player has actually been through, then '
                    'asks the model what a new injury of their usual type would '
                    'look like.</sub>')
            with gr.Column(scale=1):
                o2 = gr.HTML()
        pl.change(lookup, pl, o2)
        demo.load(lookup, pl, o2)

demo.launch(share=True, height=820, **LAUNCH_KW)
print("\n🎬 The public link above is live for 72 hours — record that tab in your browser.")
