# ============================================================================
# WAI PROJECT — PHASE 5b: LEAKAGE REPAIR
# Paste-ready copy of the two code cells added to the notebook (cells 36-37).
# A markdown heading sits above them at index 35:
#   ## Phase 5b - Leakage Repair: As-Of Injury History
#
# WHY: the five Transfermarkt history features (total_injuries, avg_days_missed,
# max_days_missed, total_games_missed, distinct_injury_types) were built in Phase 2
# with a groupby over each player's ENTIRE Transfermarkt career (1973-2025). That
# window CONTAINS the 2020-25 injuries being predicted, so each feature carried part
# of its own answer. These cells rebuild them using only injuries that started
# STRICTLY BEFORE the row being predicted, then re-run the same three models on the
# same split so the leaked-vs-honest gap can be measured.
# ============================================================================


# ############################################################################
# CELL 1 — Rebuild injury history as-of the injury date
# ############################################################################

# ============================================
# PHASE 5b: Leakage Repair — As-Of Injury History
# ============================================
# Phase 4 audited the PRIMARY dataset for leakage and cleared it. It did not
# audit the five features that arrived through the Transfermarkt merge:
#
#     total_injuries, avg_days_missed, max_days_missed,
#     total_games_missed, distinct_injury_types
#
# Those were built in Phase 2 with a plain groupby over a player's ENTIRE
# Transfermarkt career (1973-2025). That window CONTAINS the 2020-25 injuries
# we are predicting, so each feature carries part of its own answer.
# This cell rebuilds them "as-of" each injury date: only injuries that started
# STRICTLY BEFORE the current one are allowed to contribute.

print("=" * 70)
print("🩹 PHASE 5b — LEAKAGE REPAIR: AS-OF INJURY HISTORY")
print("=" * 70)

HIST_FEATURES = ['total_injuries', 'avg_days_missed', 'max_days_missed',
                 'total_games_missed', 'distinct_injury_types']

# ---------------------------------------------------------------- 1. evidence
tm = df_injuries_tm.copy()
tm['from_date'] = pd.to_datetime(tm['from_date'], errors='coerce')
tm['days_missed'] = pd.to_numeric(tm['days_missed'], errors='coerce').fillna(0)
tm['games_missed'] = pd.to_numeric(tm['games_missed'], errors='coerce').fillna(0)
tm = tm.dropna(subset=['from_date', 'player_id'])

WINDOW = ['20/21', '21/22', '22/23', '23/24', '24/25']
in_win = tm['season_name'].astype(str).isin(WINDOW)
print(f"\n📋 The evidence:")
print(f"   Transfermarkt injury records          : {len(tm):,}")
print(f"   ...of which sit INSIDE 20/21-24/25    : {in_win.sum():,} ({in_win.mean()*100:.1f}%)")
_g = tm.groupby('player_id')['season_name'].agg(
    n='size', n_win=lambda s: s.astype(str).isin(WINDOW).sum())
_g = _g[_g.n_win > 0]
_share = _g.n_win / _g.n
print(f"   Players with >=1 injury in the window : {len(_g):,}")
print(f"   Median share of their 'career' that IS the window: {_share.median()*100:.0f}%")
print(f"   Players where the window is 100% of history      : {(_share == 1).mean()*100:.0f}%")
print(f"   → the old features average in the very row being predicted.")

# ------------------------------------------------- 2. running prior aggregates
# Sort by date so 'prior' means chronologically earlier. Aggregates are
# inclusive-to-date here; merge_asof then picks the last row STRICTLY before
# the injury being predicted, which makes them exclusive of that injury.
tm = tm.sort_values(['player_id', 'from_date'], kind='mergesort')
grp = tm.groupby('player_id', sort=False)

tm['h_n']     = grp.cumcount() + 1
tm['h_days']  = grp['days_missed'].cumsum()
tm['h_max']   = grp['days_missed'].cummax()
tm['h_games'] = grp['games_missed'].cumsum()
# running count of DISTINCT injury_reason: a reason counts once, at first sight
_first_time = ~tm.duplicated(subset=['player_id', 'injury_reason'])
tm['h_types'] = _first_time.groupby(tm['player_id']).cumsum()

# ------------------------------------------------------- 3. name -> player_id
name_id = df_profiles_clean[['player_name_clean', 'player_id']].copy()
name_id['player_name_clean'] = name_id['player_name_clean'].str.lower().str.strip()
name_id = name_id.dropna(subset=['player_id']).drop_duplicates('player_name_clean', keep='first')

left = df_merged[['player_name_clean']].copy()
left['_row'] = np.arange(len(left))
left['inj_date'] = pd.to_datetime(df_merged['injury_from_parsed'], errors='coerce').values
left = left.merge(name_id, on='player_name_clean', how='left')

usable = left.dropna(subset=['inj_date', 'player_id']).copy()
usable['player_id'] = usable['player_id'].astype('int64')
tm['player_id'] = tm['player_id'].astype('int64')
print(f"\n🔗 Rows linkable to a Transfermarkt id with a usable date: "
      f"{len(usable):,} / {len(left):,} ({len(usable)/len(left)*100:.1f}%)")

# ------------------------------------------------------------- 4. as-of merge
HIST_COLS = ['h_n', 'h_days', 'h_max', 'h_games', 'h_types']
asof = pd.merge_asof(
    usable.sort_values('inj_date'),
    tm[['player_id', 'from_date'] + HIST_COLS].sort_values('from_date'),
    left_on='inj_date', right_on='from_date', by='player_id',
    direction='backward',
    allow_exact_matches=False,      # STRICTLY earlier — this is the whole point
)

prior = pd.DataFrame(0.0, index=np.arange(len(df_merged)), columns=HIST_COLS)
prior.loc[asof['_row'].values, HIST_COLS] = asof[HIST_COLS].fillna(0).values

# ------------------------------------------------- 5. write the _prior columns
df_merged['total_injuries_prior']        = prior['h_n'].values
df_merged['total_days_missed_prior']     = prior['h_days'].values
df_merged['max_days_missed_prior']       = prior['h_max'].values
df_merged['total_games_missed_prior']    = prior['h_games'].values
df_merged['distinct_injury_types_prior'] = prior['h_types'].values
with np.errstate(divide='ignore', invalid='ignore'):
    df_merged['avg_days_missed_prior'] = np.where(
        prior['h_n'].values > 0, prior['h_days'].values / prior['h_n'].values, 0.0)

PRIOR_FEATURES = ['total_injuries_prior', 'avg_days_missed_prior', 'max_days_missed_prior',
                  'total_games_missed_prior', 'distinct_injury_types_prior']

no_hist = (df_merged['total_injuries_prior'] == 0).mean()
print(f"   Rows with NO prior injury on record: {no_hist*100:.1f}% "
      f"(these get 0 — a genuine 'first recorded injury' signal)")

# --------------------------------------------------- 6. how much did it move?
print(f"\n📉 Correlation with the target — leaked vs as-of:")
rows = []
for leaked, honest in zip(HIST_FEATURES, PRIOR_FEATURES):
    if leaked not in df_merged.columns:
        continue
    c_old = df_merged[leaked].corr(df_merged[TARGET])
    c_new = df_merged[honest].corr(df_merged[TARGET])
    rows.append({'feature': leaked, 'corr (leaked)': c_old, 'corr (as-of)': c_new,
                 'drop': abs(c_old) - abs(c_new)})
leak_tbl = pd.DataFrame(rows).set_index('feature')
display(leak_tbl.style.format('{:.4f}'))
print("   A large drop = that much of the old signal was the answer leaking back in.")

print(f"\n{'=' * 70}")
print("✅ PHASE 5b PREP COMPLETE — honest history features built")
print("=" * 70)


# ############################################################################
# CELL 2 — Re-run all three models, leakage-free
# ############################################################################

# ============================================
# PHASE 5b: Re-run the Three Models, Leakage-Free
# ============================================
# Same split, same grids, same seed as Phase 5 — the ONLY change is that the
# five history features are now as-of. Any drop in score is the leak coming out.

print("=" * 70)
print("🔁 PHASE 5b — RE-RUNNING ALL THREE MODELS (HONEST FEATURES)")
print("=" * 70)

# --- Swap the leaked history features for their as-of twins ---
HONEST_FEATURES = [f for f in MODEL_FEATURES if f not in HIST_FEATURES] + PRIOR_FEATURES
HONEST_FEATURES = [f for f in HONEST_FEATURES if f in df_merged.columns]

print(f"\n   Removed (leaked) : {[f for f in HIST_FEATURES if f in MODEL_FEATURES]}")
print(f"   Added   (as-of)  : {PRIOR_FEATURES}")
print(f"   Numeric features : {len(MODEL_FEATURES)} → {len(HONEST_FEATURES)}")

Xh_num = df_merged[HONEST_FEATURES].reset_index(drop=True).astype(float)
Xh = pd.concat([Xh_num, X_cat], axis=1)          # same one-hot block as Phase 5
print(f"   Design matrix    : {Xh.shape[0]:,} rows × {Xh.shape[1]} columns")

# --- Identical split: reuse the Phase 5 indices so the comparison is clean ---
Xh_train, Xh_test = Xh.iloc[train_idx], Xh.iloc[test_idx]
assert len(set(g_train) & set(g_test)) == 0, "Player leakage across the split!"
print(f"   Reusing the Phase 5 split — {len(Xh_train):,} train / {len(Xh_test):,} test, 0 shared players")

results_honest = []

# --- 1. Logistic Regression ---
print(f"\n{'-' * 70}\n1️⃣  Logistic Regression")
gs = GridSearchCV(ImbPipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                   solver='liblinear', random_state=RANDOM_STATE))]),
    {'clf__C': [0.01, 0.1, 1.0, 10.0]}, cv=cv, scoring='roc_auc', n_jobs=-1)
gs.fit(Xh_train, y_train, groups=g_train)
lr_h = gs.best_estimator_
print(f"   Best params: {gs.best_params_}   CV ROC-AUC: {gs.best_score_:.4f}")
results_honest.append(evaluate_model(lr_h, Xh_train, Xh_test, y_train, y_test, 'Logistic Regression'))

# --- 2. Random Forest ---
print(f"\n{'-' * 70}\n2️⃣  Random Forest")
gs = GridSearchCV(
    RandomForestClassifier(n_estimators=300, class_weight='balanced',
                           random_state=RANDOM_STATE, n_jobs=-1),
    {'max_depth': [6, 12, None], 'min_samples_leaf': [1, 5]},
    cv=cv, scoring='roc_auc', n_jobs=-1)
gs.fit(Xh_train, y_train, groups=g_train)
rf_h = gs.best_estimator_
print(f"   Best params: {gs.best_params_}   CV ROC-AUC: {gs.best_score_:.4f}")
results_honest.append(evaluate_model(rf_h, Xh_train, Xh_test, y_train, y_test, 'Random Forest'))

# --- 3. XGBoost ---
print(f"\n{'-' * 70}\n3️⃣  XGBoost")
gs = GridSearchCV(
    XGBClassifier(n_estimators=300, scale_pos_weight=SCALE_POS_WEIGHT,
                  eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    {'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]},
    cv=cv, scoring='roc_auc', n_jobs=-1)
gs.fit(Xh_train, y_train, groups=g_train)
xgb_h = gs.best_estimator_
print(f"   Best params: {gs.best_params_}   CV ROC-AUC: {gs.best_score_:.4f}")
results_honest.append(evaluate_model(xgb_h, Xh_train, Xh_test, y_train, y_test, 'XGBoost'))

# ------------------------------------------------------ leaked vs honest table
print(f"\n\n{'=' * 70}")
print("⚖️  LEAKED vs HONEST — what the leak was worth")
print("=" * 70)

honest = pd.DataFrame(results_honest).set_index('Model')
leaked = pd.DataFrame(results).set_index('Model')

side = pd.concat({'Leaked (Phase 5)': leaked[['Recall', 'F1', 'AUC']],
                  'As-of (Phase 5b)': honest[['Recall', 'F1', 'AUC']]}, axis=1)
side[('Δ', 'AUC')] = honest['AUC'] - leaked['AUC']
side = side.sort_values(('As-of (Phase 5b)', 'AUC'), ascending=False)
display(side.style.format('{:+.4f}', subset=[('Δ', 'AUC')]).format('{:.4f}', subset=side.columns[:-1]))

best_h = honest['AUC'].idxmax()
print(f"\n🏆 Best honest model: {best_h} — AUC {honest.loc[best_h, 'AUC']:.4f}, "
      f"recall {honest.loc[best_h, 'Recall']:.4f}")
print(f"   Mean AUC lost to the repair: {(honest['AUC'] - leaked['AUC']).mean():+.4f}")
print("   That gap was never real predictive skill — it was the target leaking back in.")

# --- Comparison chart ---
fig, ax = plt.subplots(figsize=(10, 5))
idx = np.arange(len(side))
ax.bar(idx - 0.2, side[('Leaked (Phase 5)', 'AUC')], 0.4, label='Leaked (Phase 5)', color='#e74c3c')
ax.bar(idx + 0.2, side[('As-of (Phase 5b)', 'AUC')], 0.4, label='As-of (Phase 5b)', color='#2ecc71')
ax.set_xticks(idx); ax.set_xticklabels(side.index)
ax.set_ylabel('Test ROC-AUC')
ax.set_title('Phase 5b — AUC before and after removing the history leak',
             fontsize=13, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6)
ax.legend()
for i, (l, h) in enumerate(zip(side[('Leaked (Phase 5)', 'AUC')], side[('As-of (Phase 5b)', 'AUC')])):
    ax.text(i - 0.2, l + 0.005, f'{l:.3f}', ha='center', fontsize=9)
    ax.text(i + 0.2, h + 0.005, f'{h:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('/content/phase5b_leak_repair.png', dpi=150, bbox_inches='tight')
plt.show()

# --- SHAP on the honest best tree model ---
try:
    tree_h = {'Random Forest': rf_h, 'XGBoost': xgb_h}
    nm = max(tree_h, key=lambda m: honest.loc[m, 'AUC'])
    smp = Xh_test.sample(min(1000, len(Xh_test)), random_state=RANDOM_STATE)
    sv = shap.TreeExplainer(tree_h[nm]).shap_values(smp)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, smp, plot_type='bar', show=False, max_display=15)
    plt.title(f'{nm} (as-of features) — mean |SHAP|', fontweight='bold')
    plt.tight_layout()
    plt.savefig('/content/phase5b_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   ✅ Saved: phase5b_shap_bar.png  (explaining {nm})")
except Exception as e:
    print(f"   ⚠️ SHAP failed: {type(e).__name__}: {e}")

honest.to_csv('/content/phase5b_honest_results.csv')
print(f"\n💾 Saved: phase5b_honest_results.csv, phase5b_leak_repair.png")

print("\n\n" + "=" * 70)
print("✅ PHASE 5b COMPLETE — leak found, repaired, and quantified")
print("=" * 70)
print("🎯 Report the AS-OF numbers as your result; the leaked ones as the finding.")

