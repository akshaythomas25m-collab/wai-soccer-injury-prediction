"""Severe Injury Risk Simulator — Streamlit app.

Runs the leakage-corrected XGBoost model live: every prediction is computed on
request, with SHAP explaining that individual prediction.

Files expected beside this one:
    model.json    XGBoost booster, portable format (not a pickle — a pickle
                  only loads under a matching xgboost version)
    schema.json   design-matrix column order + median row + test metrics
    levels.json   dropdown levels, read off the training data
    players.csv   one summary row per player, for the look-up tab

Deployed on Streamlit Community Cloud from the project's GitHub repo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

HERE = Path(__file__).parent

st.set_page_config(page_title='Severe Injury Risk Simulator',
                   page_icon='⚽', layout='wide')

# ------------------------------------------------------------ load artifacts
@st.cache_resource(show_spinner='Loading the model…')
def load_model():
    m = XGBClassifier()
    m.load_model(str(HERE / 'model.json'))
    try:
        import shap
        return m, shap.TreeExplainer(m)
    except Exception:
        return m, None

@st.cache_data
def load_reference():
    schema = json.loads((HERE / 'schema.json').read_text())
    levels = json.loads((HERE / 'levels.json').read_text())
    players = pd.read_csv(HERE / 'players.csv')
    players['player'] = players['player'].astype(str)
    return schema, levels, players

MODEL, EXPLAINER = load_model()
_schema, _lv, PLAYERS_DF = load_reference()

COLS     = _schema['columns']
TEMPLATE = pd.Series(_schema['template']).reindex(COLS)
METRICS  = _schema['metrics']

CATEGORIES = _lv['categories']
LEAGUES    = _lv['leagues']
POSITIONS  = _lv['positions']
PHASES     = _lv['phases']

PLAYERS = sorted(PLAYERS_DF['player'].tolist())
PROFILE = PLAYERS_DF.set_index('player')

# ------------------------------------------------------------------- labels
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

# ------------------------------------------------------------ build a row
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
               'injuries_in_dataset': n + 1}

    if n == 0:
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
            f'<b class="{"up" if v > 0 else "down"}">'
            f'{"↑ raises" if v > 0 else "↓ lowers"} risk</b></div>'
            for i, v in s.items())
        return ('<div style="margin-top:16px">'
                '<div class="dim" style="margin-bottom:4px">'
                f'WHAT DROVE THIS PREDICTION</div>{rows}</div>')
    except Exception:
        return ''

# ------------------------------------------------------------ presentation
# The card is a light panel that has to stay readable if the viewer's Streamlit
# theme is dark. A dark theme recolours text inside injected HTML, so every text
# element declares its colour with !important rather than inheriting one —
# otherwise half the card renders white-on-white and disappears.
CARD_STYLE = """<style>
.wai-card{font-family:ui-sans-serif,system-ui;padding:20px;border:1px solid #ddd;
 border-radius:8px;background:#fff}
.wai-card,.wai-card div,.wai-card span,.wai-card b{color:#141A17 !important}
.wai-card .dim{color:#666 !important;font-size:11.5px;letter-spacing:.14em}
.wai-card .faint{color:#999 !important;font-size:11px}
.wai-card .muted,.wai-card .muted b{color:#555 !important;font-size:12.5px;line-height:1.55}
.wai-card .band{color:var(--band) !important}
.wai-card .up{color:#BE4A10 !important;white-space:nowrap}
.wai-card .down{color:#0C8F80 !important;white-space:nowrap}
.wai-card .pct{font-size:56px;font-weight:600;line-height:1.1}
.wai-card .bandname{font-size:15px;font-weight:600;letter-spacing:.04em}
.wai-card .head{margin-top:12px;font-size:15px;line-height:1.6}
.wai-card .prof{margin-top:12px;font-size:14px;line-height:1.75;padding:12px 14px;
 background:#F7F9F8;border-radius:6px}
.wai-card .frow{display:flex;justify-content:space-between;gap:12px;padding:5px 0;
 border-bottom:1px solid #f0f0f0;font-size:13.5px}
.wai-card hr{border:none;border-top:1px solid #eee;margin:16px 0 12px}
</style>"""

CAVEAT = f"""<hr><div class="muted">
<b>This predicts severity given an injury has occurred — not whether an injury will happen.</b><br>
On held-out data (1,021 players never seen in training) the model catches
<b>{METRICS['recall']*100:.0f}%</b> of severe cases, but <b>nearly half of what it flags is a
false alarm</b> — AUC {METRICS['auc']:.3f}, precision {METRICS['precision']:.3f}. A triage aid,
not a clinical decision, and an association in historical data rather than a causal claim.<br>
<br>The number above is a <b>class-weighted risk score, not a calibrated probability</b>. The model
is trained with <code>scale_pos_weight</code> so that it errs towards catching severe cases, which
pushes every score upward; compare scores against the 49% decision threshold and against each other,
not against the 35.5% rate of severe injuries in the data.</div>"""

def card(p, headline, extra=''):
    if   p >= 0.60: band, colour = 'HIGH RISK',     '#BE4A10'
    elif p >= 0.49: band, colour = 'ELEVATED RISK', '#8A6512'
    else:           band, colour = 'LOWER RISK',    '#0C8F80'
    return f"""{CARD_STYLE}<div class="wai-card" style="--band:{colour}">
<div class="dim">SEVERITY RISK SCORE — WILL IT PASS 28 DAYS?</div>
<div class="pct band">{p*100:.0f}%</div>
<div class="bandname band">{band}</div>
<div style="height:8px;background:#eee;border-radius:4px;overflow:hidden;margin:10px 0 4px">
<div style="height:100%;width:{p*100:.0f}%;background:{colour}"></div></div>
<div style="display:flex;justify-content:space-between"><span class="faint">0%</span>
<span class="faint">49% — decision threshold</span><span class="faint">100%</span></div>
<div class="head">{headline}</div>
{extra}{CAVEAT}</div>"""

def first(options, *preferred):
    """Open on a recognisable default rather than whatever sorts first."""
    opts = list(options)
    for p in preferred:
        if p in opts:
            return opts.index(p)
    return 0

# ------------------------------------------------------------------ the app
st.title('⚽ Severe Injury Risk Simulator')
st.markdown(
    'Will a football injury keep a player out for **28 days or more**?  \n'
    ':grey[XGBoost · 15,603 injuries · 4,081 players · 5 European leagues · '
    '2020–2025 · leakage-corrected]')

tab1, tab2 = st.tabs(['Simulate an injury', 'Look up a player'])

with tab1:
    left, right = st.columns(2, gap='large')
    with left:
        cat_label = st.selectbox('Injury category', list(CAT_LABEL),
                                 index=first(CAT_LABEL, pretty('knee')))
        age = st.slider('Player age', 16, 40, 27)
        league = st.selectbox('League', LEAGUES,
                              index=first(LEAGUES, 'Premier League'))
        pos_label = st.selectbox('Position', list(POS_LABEL),
                                 index=first(POS_LABEL, 'Centre-forward',
                                             'Centre-back', 'Central midfield'))
        phase_label = st.selectbox('Point in the season', list(PHZ_LABEL),
                                   index=first(PHZ_LABEL, 'Mid', 'Early'))
        priors = st.slider("Injuries already on this player's record", 0, 20, 2)
    with right:
        X = build_row(CAT_LABEL[cat_label], age, league, POS_LABEL[pos_label],
                      PHZ_LABEL[phase_label], priors)
        p = float(MODEL.predict_proba(X)[0, 1])
        verdict = ('likely to run past 28 days' if p >= 0.49
                   else 'likely to resolve inside 28 days')
        head = (f'A <b>{pretty(CAT_LABEL[cat_label]).lower()}</b> injury to a '
                f'{age}-year-old {pos_label.lower()} in the {league}, picked up in '
                f'the {phase_label.lower()} part of the season, is <b>{verdict}</b>.')
        st.markdown(card(p, head, top_factors(X)), unsafe_allow_html=True)

with tab2:
    left, right = st.columns(2, gap='large')
    with left:
        name = st.selectbox(f'Player — type to search {len(PLAYERS):,}',
                            PLAYERS, index=0)
        st.caption('Shows what this player has actually been through, then asks '
                   'the model what a new injury of their usual type would look like.')
    with right:
        r = PROFILE.loc[name]
        usual    = r['usual_category'] if r['usual_category'] in CATEGORIES else CATEGORIES[0]
        league_p = r['league']         if r['league']         in LEAGUES    else LEAGUES[0]
        pos_p    = r['position']       if r['position']       in POSITIONS  else POSITIONS[0]
        n        = int(r['n_injuries'])

        X = build_row(usual, int(r['age']), league_p, pos_p, 'mid', n)
        p = float(MODEL.predict_proba(X)[0, 1])

        profile = f"""<div class="prof">
<b style="font-size:15px">{name.title()}</b> — {r['position']}, {r['league']}<br>
<b>{n}</b> {'injuries' if n != 1 else 'injury'} on record ·
most often <b>{pretty(r['usual_category']).lower()}</b><br>
median <b>{r['median_days']:.0f}</b> days out · worst <b>{r['worst_days']:.0f}</b> days ·
<b>{r['severe_rate']*100:.0f}%</b> of them were severe</div>"""

        head = (f'If {name.title()} picks up a new '
                f'<b>{pretty(usual).lower()}</b> injury mid-season, '
                f'the model scores it as shown.')
        st.markdown(card(p, head + profile, top_factors(X)), unsafe_allow_html=True)

st.divider()
st.caption('Akshay Thomas · IIM Ranchi · WAI Sports Analytics project. '
           'Trained on recorded injuries only, with no exposure data '
           '(minutes played, training load), which caps what it can achieve.')
