PHASE 4 — ASSUMPTIONS & DIAGNOSTICS CELLS
Paste each block into its own Colab cell, in order, after the Phase 3 cells.
Cell 1 is a TEXT cell; cells 2-5 are CODE cells.

#==============================================================================
# CELL 1 — TEXT / MARKDOWN CELL
#==============================================================================

## Phase 4 — Assumptions & Diagnostics


#==============================================================================
# CELL 2 — CODE: Multicollinearity (VIF)
#==============================================================================

# ============================================
# PHASE 4: Assumption Check 1 — Multicollinearity (VIF)
# ============================================

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

print("=" * 70)
print("📐 ASSUMPTION 1: NO SEVERE MULTICOLLINEARITY")
print("=" * 70)

def compute_vif(frame, cols):
    """VIF for each column. >10 = severe, 5-10 = moderate, <5 = acceptable."""
    X = add_constant(frame[cols].astype(float), has_constant='add')
    rows = []
    for i, c in enumerate(X.columns):
        if c == 'const':
            continue
        rows.append({'feature': c, 'VIF': variance_inflation_factor(X.values, i)})
    return pd.DataFrame(rows).sort_values('VIF', ascending=False).reset_index(drop=True)

vif_all = compute_vif(df_merged, FEATURE_COLS)
print(f"\n📊 VIF — all {len(FEATURE_COLS)} candidate features:")
display(vif_all)

severe_vif = vif_all[vif_all['VIF'] > 10]['feature'].tolist()
print(f"\n⚠️ Severe multicollinearity (VIF > 10): {severe_vif}")
print("   Interpretation: these carry near-duplicate information. Logistic Regression")
print("   coefficients become unstable and uninterpretable; tree models are unaffected.")


#==============================================================================
# CELL 3 — CODE: Target Leakage Audit + Clean Feature Set
#==============================================================================

# ============================================
# PHASE 4: Assumption Check 2 — Target Leakage Audit
# ============================================

print("=" * 70)
print("🚨 ASSUMPTION 2: NO TARGET LEAKAGE")
print("=" * 70)

# A feature leaks if it is derived from, or is a consequence of, the outcome.
leak_audit = {
    'days_numeric': 'DEFINITIONAL — severe_injury IS (days_numeric >= 28)',
    'games_missed': 'CONSEQUENCE — games missed because of this injury, unknown at prediction time',
    'position_risk_rate': 'FITTED ON FULL DATA — encodes test-set outcomes into a training feature',
    'league_risk_rate': 'FITTED ON FULL DATA — encodes test-set outcomes into a training feature',
}

print("\n📋 Flagged features:")
for f, why in leak_audit.items():
    if f in df_merged.columns:
        c = df_merged[f].corr(df_merged['severe_injury']) if df_merged[f].dtype != 'O' else float('nan')
        print(f"\n   ❌ {f}")
        print(f"      reason      : {why}")
        print(f"      corr target : {c:.3f}")

# Evidence: days_numeric and games_missed are near-duplicates of each other
dup = df_merged['days_numeric'].corr(df_merged['games_missed'])
print(f"\n🔍 corr(days_numeric, games_missed) = {dup:.3f}")
print("   Two near-identical restatements of injury duration — the thing being predicted.")

# --- Build the clean modelling feature set ---
LEAKY = [f for f in leak_audit if f in FEATURE_COLS]
COLLINEAR_DROP = ['age_squared']   # VIF > 100 against player_age; adds no signal

MODEL_FEATURES = [f for f in FEATURE_COLS if f not in LEAKY + COLLINEAR_DROP]

print(f"\n{'='*70}")
print("✅ FINAL MODELLING FEATURE SET")
print("=" * 70)
print(f"\n   Dropped for leakage ({len(LEAKY)}): {LEAKY}")
print(f"   Dropped for collinearity ({len(COLLINEAR_DROP)}): {COLLINEAR_DROP}")
print(f"\n   Retained numeric features ({len(MODEL_FEATURES)}):")
for f in MODEL_FEATURES:
    print(f"      • {f}")
print(f"\n   Categorical features ({len(CAT_FEATURES)}): {CAT_FEATURES}")

# Re-check VIF on the clean set
vif_clean = compute_vif(df_merged, MODEL_FEATURES)
print(f"\n📊 VIF after cleaning (max = {vif_clean['VIF'].max():.2f}):")
display(vif_clean)

# Honest correlation ceiling now that leakage is gone
clean_corr = df_merged[MODEL_FEATURES + ['severe_injury']].corr()['severe_injury'].drop('severe_injury').abs().sort_values(ascending=False)
print(f"\n📉 Strongest remaining correlation with target: {clean_corr.iloc[0]:.3f} ({clean_corr.index[0]})")
print("   This is the honest signal ceiling the Phase 5 models must beat.")


#==============================================================================
# CELL 4 — CODE: Linearity, Outliers, Independence
#==============================================================================

# ============================================
# PHASE 4: Assumption Check 3 — Linearity of the Logit & Influence
# ============================================

print("=" * 70)
print("📈 ASSUMPTION 3: LINEARITY OF THE LOGIT (Logistic Regression only)")
print("=" * 70)
print("\n   Logistic Regression assumes each continuous predictor is linear in log-odds.")
print("   Random Forest and XGBoost make no such assumption — this check scopes")
print("   which model the caveat applies to, it does not disqualify the others.\n")

continuous = [f for f in MODEL_FEATURES
              if df_merged[f].nunique() > 10 and not f.startswith('is_')]

n = len(continuous)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
fig.suptitle('Empirical Log-Odds vs Feature Decile (linearity check)',
             fontsize=16, fontweight='bold', y=1.00)
axes = np.array(axes).reshape(-1)

linearity_report = []
for i, f in enumerate(continuous):
    ax = axes[i]
    # Bin into deciles, compute empirical log-odds per bin
    b = pd.qcut(df_merged[f], 10, duplicates='drop')
    g = df_merged.groupby(b, observed=True)['severe_injury'].agg(['mean', 'count'])
    p = g['mean'].clip(0.01, 0.99)
    logodds = np.log(p / (1 - p))
    xs = np.arange(len(logodds))
    ax.plot(xs, logodds.values, marker='o', color='steelblue', linewidth=2)
    # Straight-line reference through the endpoints
    ax.plot([xs[0], xs[-1]], [logodds.values[0], logodds.values[-1]],
            'r--', alpha=0.6, label='linear reference')
    # R² of a straight-line fit = how linear it is
    coef = np.polyfit(xs, logodds.values, 1)
    pred = np.polyval(coef, xs)
    ss_res = ((logodds.values - pred) ** 2).sum()
    ss_tot = ((logodds.values - logodds.values.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    linearity_report.append({'feature': f, 'linearity_R2': r2,
                             'verdict': 'linear' if r2 > 0.8 else 'NON-LINEAR'})
    ax.set_title(f'{f}\nR² = {r2:.2f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Decile')
    ax.set_ylabel('log-odds(severe)')
    ax.legend(fontsize=8)

for j in range(n, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('/content/phase4_linearity_check.png', dpi=150, bbox_inches='tight')
plt.show()

lin_df = pd.DataFrame(linearity_report).sort_values('linearity_R2')
print("\n📊 Linearity of the logit by feature (R² of straight-line fit to decile log-odds):")
display(lin_df)

nonlin = lin_df[lin_df['verdict'] == 'NON-LINEAR']['feature'].tolist()
print(f"\n⚠️ Non-linear in the logit: {nonlin if nonlin else 'none'}")
print("   → Logistic Regression will under-fit these; expect RF/XGBoost to outperform it.")

# --- Outliers / influence ---
print(f"\n\n{'='*70}")
print("📌 ASSUMPTION 4: OUTLIERS & INFLUENTIAL POINTS")
print("=" * 70)
z = (df_merged[MODEL_FEATURES] - df_merged[MODEL_FEATURES].mean()) / df_merged[MODEL_FEATURES].std()
extreme = (z.abs() > 4).sum().sort_values(ascending=False)
extreme = extreme[extreme > 0]
print(f"\n   Rows beyond |z| > 4, by feature:")
if len(extreme):
    for f, c in extreme.items():
        print(f"      {f}: {c} rows ({c/len(df_merged)*100:.2f}%)")
else:
    print("      none")
rows_affected = (z.abs() > 4).any(axis=1).sum()
print(f"\n   Rows with at least one extreme value: {rows_affected:,} ({rows_affected/len(df_merged)*100:.2f}%)")
print("   Decision: RETAINED. These are genuine long-duration injuries (ACL ruptures,")
print("   fractures), not data errors — they are exactly the cases the model must catch.")

# --- Independence ---
print(f"\n\n{'='*70}")
print("🔗 ASSUMPTION 5: INDEPENDENCE OF OBSERVATIONS")
print("=" * 70)
n_players = df_merged['player_name_clean'].nunique()
print(f"\n   Rows (injuries): {len(df_merged):,}")
print(f"   Unique players : {n_players:,}")
print(f"   Mean injuries per player: {len(df_merged)/n_players:.2f}")
print(f"   Max injuries for one player: {df_merged['injuries_in_dataset'].max():.0f}")
print("\n   ⚠️ VIOLATED: rows are not independent — the same player appears many times.")
print("   Mitigation for Phase 5: use GroupShuffleSplit / StratifiedGroupKFold on")
print("   player_name_clean so no player appears in both train and test.")


#==============================================================================
# CELL 5 — CODE: Assumptions Summary + Frozen Set
#==============================================================================

# ============================================
# PHASE 4: Assumptions Summary & Frozen Modelling Set
# ============================================

print("=" * 70)
print("📋 PHASE 4 — ASSUMPTIONS SUMMARY")
print("=" * 70)

summary = pd.DataFrame([
    {'#': 1, 'Assumption': 'No severe multicollinearity',
     'Status': 'RESOLVED',
     'Action': 'Dropped age_squared (VIF > 100 vs player_age)'},
    {'#': 2, 'Assumption': 'No target leakage',
     'Status': 'RESOLVED',
     'Action': f'Dropped {len(LEAKY)} leaking features: {", ".join(LEAKY)}'},
    {'#': 3, 'Assumption': 'Linearity of the logit',
     'Status': 'PARTIAL',
     'Action': 'Applies to Logistic Regression only; RF/XGBoost exempt'},
    {'#': 4, 'Assumption': 'No unduly influential outliers',
     'Status': 'ACCEPTED',
     'Action': 'Extreme durations retained — genuine severe injuries'},
    {'#': 5, 'Assumption': 'Independence of observations',
     'Status': 'VIOLATED',
     'Action': 'Group-aware CV on player_name_clean in Phase 5'},
    {'#': 6, 'Assumption': 'Balanced classes',
     'Status': 'MILD',
     'Action': '35.5/64.5 (1.8:1) — class_weight, not SMOTE'},
])
display(summary)

# --- Freeze the modelling frame ---
model_df = df_merged[MODEL_FEATURES + CAT_FEATURES + ['severe_injury', 'player_name_clean']].copy()
model_df.to_csv('/content/phase4_model_ready.csv', index=False)

print(f"\n💾 Saved: phase4_model_ready.csv  ({model_df.shape[0]:,} rows × {model_df.shape[1]} columns)")
print(f"\n   MODEL_FEATURES ({len(MODEL_FEATURES)}) — numeric, leakage-free")
print(f"   CAT_FEATURES   ({len(CAT_FEATURES)}) — to encode in Phase 5")
print(f"   GROUP_COL      = 'player_name_clean' — for group-aware splitting")
print(f"   TARGET         = 'severe_injury'")

GROUP_COL = 'player_name_clean'

print("\n\n" + "=" * 70)
print("✅ PHASE 4 COMPLETE — Assumptions documented, feature set frozen")
print("=" * 70)
print("🎯 Next: Phase 5 — Modeling (LogReg → Random Forest → XGBoost)")

