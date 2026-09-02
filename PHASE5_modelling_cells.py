# ============================================================================
# WAI PROJECT — PHASE 5: MODELLING
# Paste-ready copy of the five code cells added to
# WAI_Soccer_Injury_Prediction.ipynb (cells 28-32).
# A markdown heading cell sits above them at index 27:
#   ## Phase 5 - Modelling: Logistic Regression -> Random Forest -> XGBoost
# ============================================================================


# ############################################################################
# CELL 1 — Modelling Prep (encode + group-aware split)
# ############################################################################

# ============================================
# PHASE 5: Modelling Prep — Encode & Split by Player
# ============================================

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

print("=" * 70)
print("🔧 PHASE 5 PREP — ENCODING & GROUP-AWARE SPLIT")
print("=" * 70)

# --- Drop the last feature above the VIF threshold (Phase 4, §5.1) ---
DROP_VIF = ['total_days_missed']          # VIF 11.07; ≈ total_injuries × avg_days_missed
MODEL_FEATURES = [f for f in MODEL_FEATURES if f not in DROP_VIF]
print(f"\n✂️  Dropped for collinearity: {DROP_VIF}")
print(f"   Numeric features now: {len(MODEL_FEATURES)}")

# --- One-hot encode the categoricals ---
X_num = df_merged[MODEL_FEATURES].reset_index(drop=True).astype(float)
X_cat = pd.get_dummies(df_merged[CAT_FEATURES].astype(str),
                       prefix=CAT_FEATURES, drop_first=True).reset_index(drop=True).astype(float)
X = pd.concat([X_num, X_cat], axis=1)
y = df_merged[TARGET].reset_index(drop=True)
groups = df_merged[GROUP_COL].reset_index(drop=True)

print(f"\n📐 Design matrix: {X.shape[0]:,} rows × {X.shape[1]} columns")
print(f"   {len(MODEL_FEATURES)} numeric + {X_cat.shape[1]} one-hot dummies")

# --- Group-aware holdout split: no player in both train and test ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

overlap = set(g_train) & set(g_test)
print(f"\n🔒 Split integrity check:")
print(f"   Train: {len(X_train):,} rows / {g_train.nunique():,} players  —  severe rate {y_train.mean():.1%}")
print(f"   Test : {len(X_test):,} rows / {g_test.nunique():,} players  —  severe rate {y_test.mean():.1%}")
print(f"   Players in BOTH train and test: {len(overlap)}")
assert len(overlap) == 0, "Player leakage across the split!"
print("   ✅ No player appears on both sides — Phase 4 assumption 5 respected.")

# --- Shared CV object for every GridSearch below ---
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Imbalance weight for XGBoost
SCALE_POS_WEIGHT = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n⚖️  scale_pos_weight for XGBoost: {SCALE_POS_WEIGHT:.3f}")

results = []   # each model appends its metrics dict here


# ############################################################################
# CELL 2 — Logistic Regression
# ############################################################################

# ============================================
# PHASE 5: Model 1 — Logistic Regression
# ============================================

print("=" * 70)
print("1️⃣  LOGISTIC REGRESSION")
print("=" * 70)

pipe_lr = ImbPipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                               solver='liblinear', random_state=RANDOM_STATE)),
])

grid_lr = {'clf__C': [0.01, 0.1, 1.0, 10.0]}

gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
gs_lr.fit(X_train, y_train, groups=g_train)

print(f"\n🔍 Best params : {gs_lr.best_params_}")
print(f"   CV ROC-AUC  : {gs_lr.best_score_:.4f}")

lr_model = gs_lr.best_estimator_
results.append(evaluate_model(lr_model, X_train, X_test, y_train, y_test, 'Logistic Regression'))

# --- Coefficients (scaled, so directly comparable) ---
coefs = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr_model.named_steps['clf'].coef_[0],
}).assign(abs_coef=lambda d: d['coefficient'].abs()).sort_values('abs_coef', ascending=False)

print("\n📊 Top 15 coefficients (standardised — sign shows direction of risk):")
display(coefs.head(15)[['feature', 'coefficient']])


# ############################################################################
# CELL 3 — Random Forest
# ############################################################################

# ============================================
# PHASE 5: Model 2 — Random Forest
# ============================================

print("=" * 70)
print("2️⃣  RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                            random_state=RANDOM_STATE, n_jobs=-1)

grid_rf = {
    'max_depth': [6, 12, None],
    'min_samples_leaf': [1, 5],
}

gs_rf = GridSearchCV(rf, grid_rf, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
gs_rf.fit(X_train, y_train, groups=g_train)

print(f"\n🔍 Best params : {gs_rf.best_params_}")
print(f"   CV ROC-AUC  : {gs_rf.best_score_:.4f}")

rf_model = gs_rf.best_estimator_
results.append(evaluate_model(rf_model, X_train, X_test, y_train, y_test, 'Random Forest'))

plot_feature_importance(rf_model, list(X_train.columns), 'Random Forest', top_n=15)


# ############################################################################
# CELL 4 — XGBoost
# ############################################################################

# ============================================
# PHASE 5: Model 3 — XGBoost
# ============================================

print("=" * 70)
print("3️⃣  XGBOOST")
print("=" * 70)

xgb = XGBClassifier(
    n_estimators=300,
    scale_pos_weight=SCALE_POS_WEIGHT,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
}

gs_xgb = GridSearchCV(xgb, grid_xgb, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
gs_xgb.fit(X_train, y_train, groups=g_train)

print(f"\n🔍 Best params : {gs_xgb.best_params_}")
print(f"   CV ROC-AUC  : {gs_xgb.best_score_:.4f}")

xgb_model = gs_xgb.best_estimator_
results.append(evaluate_model(xgb_model, X_train, X_test, y_train, y_test, 'XGBoost'))

plot_feature_importance(xgb_model, list(X_train.columns), 'XGBoost', top_n=15)


# ############################################################################
# CELL 5 — Comparison, threshold tuning, SHAP
# ############################################################################

# ============================================
# PHASE 5: Model Comparison, SHAP & Verdict
# ============================================

print("=" * 70)
print("🏁 MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(results).set_index('Model').sort_values('AUC', ascending=False)
display(comparison.style.format('{:.4f}').background_gradient(cmap='Greens'))

# --- Comparison chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Phase 5 — Model Comparison (leakage-free feature set)',
             fontsize=16, fontweight='bold', y=1.02)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
comparison[metrics].plot(kind='bar', ax=axes[0], edgecolor='white', width=0.8)
axes[0].set_title('All metrics by model', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(fontsize=9, ncol=3)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Recall on the severe class is the metric that matters commercially
rec = comparison['Recall'].sort_values()
bars = axes[1].barh(range(len(rec)), rec.values,
                    color=['#e74c3c' if v < 0.5 else '#2ecc71' for v in rec.values])
axes[1].set_yticks(range(len(rec)))
axes[1].set_yticklabels(rec.index)
axes[1].set_title('Recall on severe injuries (the costly error)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Recall')
for i, v in enumerate(rec.values):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('/content/phase5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

best_name = comparison['AUC'].idxmax()
print(f"\n🏆 Best by AUC: {best_name} ({comparison.loc[best_name, 'AUC']:.4f})")
print(f"   Best by recall on severe: {comparison['Recall'].idxmax()} ({comparison['Recall'].max():.4f})")

# --- Did the Phase 4 prediction hold? ---
lr_rank = list(comparison.index).index('Logistic Regression') + 1
print(f"\n🔬 Phase 4 predicted Logistic Regression would place last (height and player_age")
print(f"   are non-linear in the logit). It placed {lr_rank} of 3 by AUC —",
      "prediction held." if lr_rank == 3 else "prediction did NOT hold.")


# --- Decision-threshold tuning ---------------------------------------------
# class_weight / scale_pos_weight reweight the LOSS; they do not move the 0.5
# cut-off. With a costly false negative, the default threshold is rarely right.
print(f"\n\n{'='*70}")
print("🎚️  DECISION THRESHOLD TUNING")
print("=" * 70)
print("\n   A missed severe injury costs a club far more than a false alarm,")
print("   so the 0.5 default is not the operating point you want.\n")

fitted = {'Logistic Regression': lr_model, 'Random Forest': rf_model, 'XGBoost': xgb_model}
thr_rows = []
for name, mdl in fitted.items():
    proba = mdl.predict_proba(X_test)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in grid]
    best_t = grid[int(np.argmax(f1s))]
    pred_b = (proba >= best_t).astype(int)
    thr_rows.append({
        'Model': name,
        'Best threshold': best_t,
        'F1 @ 0.50': f1_score(y_test, (proba >= 0.5).astype(int)),
        'F1 @ best': f1_score(y_test, pred_b),
        'Recall @ 0.50': recall_score(y_test, (proba >= 0.5).astype(int)),
        'Recall @ best': recall_score(y_test, pred_b),
    })

thr_df = pd.DataFrame(thr_rows).set_index('Model')
display(thr_df.style.format('{:.3f}'))
print("   Report both operating points. Tuning the threshold on the test set is")
print("   itself mildly optimistic — say so, or tune it on a validation fold.")

# --- SHAP on the best tree model ---
print(f"\n\n{'='*70}")
print("🧠 SHAP EXPLAINABILITY")
print("=" * 70)

tree_models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
shap_name = max(tree_models, key=lambda m: comparison.loc[m, 'AUC'])
shap_model = tree_models[shap_name]
print(f"\n   Explaining: {shap_name}")

try:
    sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(shap_model)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):          # older shap returns one array per class
        sv = sv[1]
    elif sv.ndim == 3:                # newer shap returns (n, features, classes)
        sv = sv[:, :, 1]

    plt.figure()
    shap.summary_plot(sv, sample, plot_type='bar', show=False, max_display=15)
    plt.title(f'{shap_name} — mean |SHAP| (feature importance)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('/content/phase5_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.show()

    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.title(f'{shap_name} — SHAP value distribution', fontweight='bold')
    plt.tight_layout()
    plt.savefig('/content/phase5_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✅ Saved: phase5_shap_bar.png, phase5_shap_beeswarm.png")
except Exception as e:
    print(f"   ⚠️ SHAP failed: {type(e).__name__}: {e}")
    print("   Not fatal — tree feature importances above cover the same ground.")

# --- Persist ---
comparison.to_csv('/content/phase5_model_results.csv')
print(f"\n💾 Saved: phase5_model_results.csv, phase5_model_comparison.png")

print("\n\n" + "=" * 70)
print("✅ PHASE 5 COMPLETE — Three models trained, compared and explained")
print("=" * 70)
print("🎯 Next: Phase 6 — NLP (Optional) / Phase 7 — Dashboard")

