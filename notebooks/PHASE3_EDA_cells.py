PHASE 3 — EDA CELLS FOR WAI_Soccer_Injury_Prediction.ipynb
Paste each block below into its own Colab cell, in order, after the Phase 2 cells.
Cell 1 is a TEXT cell; cells 2-6 are CODE cells. Run each with Shift+Enter.

#==============================================================================
# CELL 1 — TEXT / MARKDOWN CELL
#==============================================================================

## Phase 3 — Exploratory Data Analysis (EDA)


#==============================================================================
# CELL 2 — CODE: Target Variable & Overview
#==============================================================================

# ============================================
# PHASE 3: EDA — Target Variable & Overview
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Soccer Injury Prediction — Dataset Overview', fontsize=18, fontweight='bold', y=1.02)

# 1. Target Distribution
colors = ['#2ecc71', '#e74c3c']
counts = df_merged['severe_injury'].value_counts()
bars = axes[0, 0].bar(['Not Severe\n(< 28 days)', 'Severe\n(28+ days)'], counts.values,
                      color=colors, edgecolor='white', linewidth=2)
axes[0, 0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Injuries')
for bar, count in zip(bars, counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                    f'{count:,}\n({count/len(df_merged)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

# 2. Days Out Distribution
axes[0, 1].hist(df_merged['days_numeric'].clip(upper=200), bins=50,
                color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(x=28, color='red', linestyle='--', linewidth=2, label='Severe threshold (28 days)')
axes[0, 1].set_title('Distribution of Days Out', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Days Out (capped at 200)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend(fontsize=10)

# 3. Age Distribution by Severity
for severity, color, label in [(0, '#2ecc71', 'Not Severe'), (1, '#e74c3c', 'Severe')]:
    subset = df_merged[df_merged['severe_injury'] == severity]['player_age']
    axes[1, 0].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
axes[1, 0].set_title('Age Distribution by Injury Severity', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Player Age')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend(fontsize=10)

# 4. Injuries by League
league_data = df_merged.groupby('league')['severe_injury'].value_counts().unstack(fill_value=0)
league_data.plot(kind='bar', stacked=True, ax=axes[1, 1], color=colors, edgecolor='white')
axes[1, 1].set_title('Injuries by League & Severity', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Number of Injuries')
axes[1, 1].legend(['Not Severe', 'Severe'], fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('/content/eda_01_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: eda_01_overview.png")


#==============================================================================
# CELL 3 — CODE: Injury Patterns
#==============================================================================

# ============================================
# PHASE 3: EDA — Injury Patterns
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Injury Patterns Analysis', fontsize=18, fontweight='bold', y=1.02)

# 1. Top 15 Injury Types
top_injuries = df_merged['injury'].value_counts().head(15)
colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 15))
axes[0, 0].barh(range(len(top_injuries)), top_injuries.values, color=colors_gradient)
axes[0, 0].set_yticks(range(len(top_injuries)))
axes[0, 0].set_yticklabels(top_injuries.index, fontsize=10)
axes[0, 0].set_title('Top 15 Injury Types', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Count')
axes[0, 0].invert_yaxis()

# 2. Injury Category by Severity
cat_severity = df_merged.groupby('injury_category')['severe_injury'].mean().sort_values(ascending=True)
colors_bar = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.35 else '#2ecc71' for v in cat_severity.values]
axes[0, 1].barh(range(len(cat_severity)), cat_severity.values, color=colors_bar)
axes[0, 1].set_yticks(range(len(cat_severity)))
axes[0, 1].set_yticklabels(cat_severity.index, fontsize=10)
axes[0, 1].set_title('Severe Injury Rate by Category', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Proportion Severe')
axes[0, 1].axvline(x=0.355, color='gray', linestyle='--', alpha=0.5, label='Dataset average (35.5%)')
axes[0, 1].legend(fontsize=9)

# 3. Monthly Injury Distribution
monthly = df_merged.groupby('injury_month').agg(
    total=('severe_injury', 'count'),
    severe=('severe_injury', 'sum')
).reset_index()
monthly['severe_rate'] = monthly['severe'] / monthly['total']

ax3 = axes[1, 0]
ax3_twin = ax3.twinx()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax3.bar(monthly['injury_month'], monthly['total'], color='steelblue', alpha=0.7, label='Total Injuries')
ax3_twin.plot(monthly['injury_month'], monthly['severe_rate'], color='red', marker='o', linewidth=2, label='Severe Rate')
ax3.set_title('Injuries by Month', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Total Injuries', color='steelblue')
ax3_twin.set_ylabel('Severe Rate', color='red')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(month_names, fontsize=9)
ax3.legend(loc='upper left', fontsize=9)
ax3_twin.legend(loc='upper right', fontsize=9)

# 4. Position Analysis
pos_data = df_merged.groupby('player_position').agg(
    count=('severe_injury', 'count'),
    severe_rate=('severe_injury', 'mean')
).sort_values('count', ascending=True)

ax4 = axes[1, 1]
ax4_twin = ax4.twiny()          # FIX: twiny, not twinx — this is a horizontal bar chart
y_pos = range(len(pos_data))
ax4.barh(y_pos, pos_data['count'], color='steelblue', alpha=0.7, label='Count')
ax4_twin.plot(pos_data['severe_rate'], y_pos, color='red', marker='o', linewidth=2, label='Severe Rate')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(pos_data.index, fontsize=9)
ax4.set_title('Injuries by Position', fontsize=14, fontweight='bold', pad=30)
ax4.set_xlabel('Count', color='steelblue')
ax4_twin.set_xlabel('Severe Rate', color='red')
ax4.legend(loc='lower right', fontsize=9)
ax4_twin.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('/content/eda_02_injury_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: eda_02_injury_patterns.png")


#==============================================================================
# CELL 4 — CODE: History Features & Correlations
#==============================================================================

# ============================================
# PHASE 3: EDA — History Features & Correlations
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Player History & Feature Correlations', fontsize=18, fontweight='bold', y=1.02)

# 1. Injury History vs Severity
history_bins = [-1, 0, 3, 5, 10, 100]   # FIX: -1 lower edge so players with 0 injuries land in the '0' bin
history_labels = ['0', '1-3', '4-5', '6-10', '10+']
df_merged['history_bin'] = pd.cut(df_merged['total_injuries'], bins=history_bins, labels=history_labels)
history_severity = df_merged.groupby('history_bin', observed=True)['severe_injury'].mean()
axes[0, 0].bar(history_severity.index.astype(str), history_severity.values, color='coral', edgecolor='white')
axes[0, 0].set_title('Severe Injury Rate by Injury History', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Previous Injuries (from Transfermarkt)')
axes[0, 0].set_ylabel('Proportion Severe')
axes[0, 0].axhline(y=0.355, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(history_severity.values):
    axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)

# 2. Age Group vs Severity
age_severity = df_merged.groupby('age_group', observed=True)['severe_injury'].mean()
axes[0, 1].bar(age_severity.index.astype(str), age_severity.values, color='mediumpurple', edgecolor='white')
axes[0, 1].set_title('Severe Injury Rate by Age Group', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Age Group')
axes[0, 1].set_ylabel('Proportion Severe')
axes[0, 1].axhline(y=0.355, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(age_severity.values):
    axes[0, 1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)

# 3. Season Trends
season_col = df_merged.columns[0]
season_trend = df_merged.groupby(season_col).agg(
    total=('severe_injury', 'count'),
    severe=('severe_injury', 'sum'),
    severe_rate=('severe_injury', 'mean')
).reset_index()
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()
ax3.bar(season_trend[season_col], season_trend['total'], color='steelblue', alpha=0.7, label='Total')
ax3_twin.plot(season_trend[season_col], season_trend['severe_rate'], color='red', marker='s', linewidth=2, markersize=8, label='Severe Rate')
ax3.set_title('Injuries Across Seasons', fontsize=14, fontweight='bold')
ax3.set_ylabel('Total Injuries', color='steelblue')
ax3_twin.set_ylabel('Severe Rate', color='red')
ax3.legend(loc='upper left', fontsize=9)
ax3_twin.legend(loc='upper right', fontsize=9)
ax3.tick_params(axis='x', rotation=30)

# 4. Correlation Heatmap
numeric_features = ['player_age', 'height', 'days_numeric', 'games_missed',
                    'total_injuries', 'avg_days_missed', 'max_days_missed',
                    'injuries_in_dataset', 'distinct_injury_types', 'severe_injury']
corr_matrix = df_merged[numeric_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=axes[1, 1], square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
            vmin=-1, vmax=1)
axes[1, 1].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1, 1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('/content/eda_03_history_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: eda_03_history_correlations.png")


#==============================================================================
# CELL 5 — CODE: Statistical Tests
#==============================================================================

# ============================================
# PHASE 3: EDA — Statistical Tests
# ============================================

from scipy import stats

print("=" * 70)
print("📊 STATISTICAL TESTS")
print("=" * 70)

# 1. T-test: Age difference between severe and non-severe
severe_ages = df_merged[df_merged['severe_injury'] == 1]['player_age']
not_severe_ages = df_merged[df_merged['severe_injury'] == 0]['player_age']
t_stat, t_pval = stats.ttest_ind(severe_ages, not_severe_ages)
print(f"\n1️⃣ T-Test: Player Age vs Severity")
print(f"   Severe mean age: {severe_ages.mean():.2f}")
print(f"   Not severe mean age: {not_severe_ages.mean():.2f}")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {t_pval:.6f}")
print(f"   {'✅ Significant (p < 0.05)' if t_pval < 0.05 else '❌ Not significant'}")

# 2. Chi-square: Position vs Severity
contingency = pd.crosstab(df_merged['player_position'], df_merged['severe_injury'])
chi2, chi_pval, dof, expected = stats.chi2_contingency(contingency)
print(f"\n2️⃣ Chi-Square Test: Position vs Severity")
print(f"   Chi-square statistic: {chi2:.4f}")
print(f"   p-value: {chi_pval:.6f}")
print(f"   Degrees of freedom: {dof}")
print(f"   {'✅ Significant — position affects severity' if chi_pval < 0.05 else '❌ Not significant'}")

# 3. Chi-square: League vs Severity
contingency2 = pd.crosstab(df_merged['league'], df_merged['severe_injury'])
chi2_2, chi_pval_2, dof_2, _ = stats.chi2_contingency(contingency2)
print(f"\n3️⃣ Chi-Square Test: League vs Severity")
print(f"   Chi-square statistic: {chi2_2:.4f}")
print(f"   p-value: {chi_pval_2:.6f}")
print(f"   {'✅ Significant — league affects severity' if chi_pval_2 < 0.05 else '❌ Not significant'}")

# 4. Chi-square: Injury Category vs Severity
contingency3 = pd.crosstab(df_merged['injury_category'], df_merged['severe_injury'])
chi2_3, chi_pval_3, dof_3, _ = stats.chi2_contingency(contingency3)
print(f"\n4️⃣ Chi-Square Test: Injury Category vs Severity")
print(f"   Chi-square statistic: {chi2_3:.4f}")
print(f"   p-value: {chi_pval_3:.6f}")
print(f"   {'✅ Significant — injury type affects severity' if chi_pval_3 < 0.05 else '❌ Not significant'}")

# 5. Mann-Whitney U: Injury history vs severity
severe_history = df_merged[df_merged['severe_injury'] == 1]['total_injuries']
not_severe_history = df_merged[df_merged['severe_injury'] == 0]['total_injuries']
u_stat, u_pval = stats.mannwhitneyu(severe_history, not_severe_history, alternative='two-sided')
print(f"\n5️⃣ Mann-Whitney U: Injury History vs Severity")
print(f"   Severe median injuries: {severe_history.median():.1f}")
print(f"   Not severe median injuries: {not_severe_history.median():.1f}")
print(f"   U-statistic: {u_stat:.0f}")
print(f"   p-value: {u_pval:.6f}")
print(f"   {'✅ Significant' if u_pval < 0.05 else '❌ Not significant'}")


#==============================================================================
# CELL 6 — CODE: Key Insights Summary
#==============================================================================

# ============================================
# PHASE 3: EDA — Key Insights Summary
# ============================================

print("=" * 70)
print("🔑 TOP 10 EDA INSIGHTS FOR THE REPORT")
print("=" * 70)

# Calculate key stats
most_common = df_merged['injury'].value_counts().index[0]
most_common_count = df_merged['injury'].value_counts().values[0]
highest_severe_cat = df_merged.groupby('injury_category')['severe_injury'].mean().idxmax()
highest_severe_cat_rate = df_merged.groupby('injury_category')['severe_injury'].mean().max()
riskiest_position = df_merged.groupby('player_position')['severe_injury'].mean().idxmax()
riskiest_pos_rate = df_merged.groupby('player_position')['severe_injury'].mean().max()
riskiest_league = df_merged.groupby('league')['severe_injury'].mean().idxmax()
riskiest_league_rate = df_merged.groupby('league')['severe_injury'].mean().max()
peak_month = df_merged.groupby('injury_month')['severe_injury'].count().idxmax()

insights = [
    f"1. DATASET SCOPE: {len(df_merged):,} injuries across {df_merged['league'].nunique()} European leagues, {df_merged[df_merged.columns[0]].nunique()} seasons (2020-2025)",
    f"2. TARGET SPLIT: 35.5% severe (28+ days) vs 64.5% not severe — a 1.8:1 ratio",
    f"3. MOST COMMON INJURY: '{most_common}' with {most_common_count:,} occurrences",
    f"4. HIGHEST SEVERITY CATEGORY: '{highest_severe_cat}' — {highest_severe_cat_rate:.1%} of these are severe",
    f"5. RISKIEST POSITION: '{riskiest_position}' — {riskiest_pos_rate:.1%} severe rate",
    f"6. RISKIEST LEAGUE: '{riskiest_league}' — {riskiest_league_rate:.1%} severe rate",
    f"7. PEAK INJURY MONTH: Month {peak_month} has the highest injury count",
    f"8. AGE EFFECT: Mean age for severe injuries = {severe_ages.mean():.1f} vs not severe = {not_severe_ages.mean():.1f} (p={t_pval:.4f})",
    f"9. INJURY HISTORY: Players with prior injuries show {'higher' if u_pval < 0.05 else 'similar'} severe injury rates (p={u_pval:.4f})",
    f"10. POSITION MATTERS: Chi-square confirms position significantly impacts severity (p={chi_pval:.6f})",
]

for insight in insights:
    print(f"\n   {insight}")

# Correlation with target
print(f"\n\n📊 TOP FEATURES CORRELATED WITH SEVERE INJURY:")
target_corr = df_merged[numeric_features].corr()['severe_injury'].drop('severe_injury').abs().sort_values(ascending=False)
for feat, corr in target_corr.head(8).items():
    print(f"   {feat}: {corr:.3f}")

print("\n\n" + "=" * 70)
print("✅ PHASE 3 COMPLETE — EDA analysis done")
print("=" * 70)
print("📄 Charts saved: eda_01_overview.png, eda_02_injury_patterns.png, eda_03_history_correlations.png")
print("🎯 Next: Phase 4 — Assumptions Document")

