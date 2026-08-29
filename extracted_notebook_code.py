
# --- Cell 3 ---
# ── Core libraries ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings('ignore')

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (10, 5),
                     'axes.titleweight': 'bold'})

SEED = 42
np.random.seed(SEED)
print('✅ Libraries loaded successfully!')

# --- Cell 5 ---
def generate_dataset(n=5000, seed=42):
    """Generate a realistic student lifestyle & exam-score dataset."""
    rng = np.random.default_rng(seed)

    # ── Raw lifestyle features ────────────────────────────────────────────────
    sleep_hours          = rng.normal(7.0, 1.2, n).clip(3, 12)
    study_hours          = rng.normal(4.5, 1.5, n).clip(0, 12)
    mobile_usage_hours   = rng.normal(3.5, 1.3, n).clip(0, 10)
    tv_hours             = rng.normal(2.0, 1.0, n).clip(0, 8)
    exercise_hours       = rng.normal(1.5, 0.8, n).clip(0, 5)
    extracurricular_hrs  = rng.normal(1.2, 0.7, n).clip(0, 5)
    friends_time_hrs     = rng.normal(2.5, 1.0, n).clip(0, 8)
    family_time_hrs      = rng.normal(2.0, 0.9, n).clip(0, 8)
    attendance_pct       = rng.normal(78, 12, n).clip(40, 100)
    stress_level         = rng.integers(1, 11, n)          # 1–10 scale
    gender               = rng.choice(['Male', 'Female'], n, p=[0.50, 0.50])

    # ── Score formula (research-aligned weights) ──────────────────────────────
    score = (
          8.0  * study_hours
        + 3.5  * sleep_hours
        + 0.35 * attendance_pct
        + 2.5  * exercise_hours
        + 1.5  * extracurricular_hrs
        + 0.8  * family_time_hrs
        - 3.0  * mobile_usage_hours
        - 2.0  * tv_hours
        - 2.5  * stress_level
        - 1.0  * friends_time_hrs
        + rng.normal(0, 5, n)   # noise
        - 10                    # baseline offset
    )

    # Normalise to [30, 100]
    score = ((score - score.min()) / (score.max() - score.min())) * 70 + 30

    df = pd.DataFrame({
        'sleep_hours':         sleep_hours.round(2),
        'study_hours':         study_hours.round(2),
        'mobile_usage_hours':  mobile_usage_hours.round(2),
        'tv_hours':            tv_hours.round(2),
        'exercise_hours':      exercise_hours.round(2),
        'extracurricular_hrs': extracurricular_hrs.round(2),
        'friends_time_hrs':    friends_time_hrs.round(2),
        'family_time_hrs':     family_time_hrs.round(2),
        'attendance_pct':      attendance_pct.round(1),
        'stress_level':        stress_level,
        'gender':              gender,
        'exam_score':          score.round(2)
    })
    return df

df = generate_dataset()
df.to_csv('student_lifestyle_data.csv', index=False)
print(f'✅ Dataset generated: {df.shape[0]:,} rows × {df.shape[1]} columns')
df.head(10)

# --- Cell 7 ---
print('='*60)
print('            DATASET OVERVIEW')
print('='*60)
print(f'Shape       : {df.shape}')
print(f'Memory      : {df.memory_usage(deep=True).sum()/1024:.1f} KB')
print()
print('Dtypes:')
print(df.dtypes)
print()
print('Missing values:')
print(df.isnull().sum())

# --- Cell 8 ---
print('Statistical Summary – Numerical Features')
df.describe().round(2)

# --- Cell 9 ---
# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df['exam_score'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(df['exam_score'].mean(), color='red', linestyle='--', label=f"Mean={df['exam_score'].mean():.1f}")
axes[0].axvline(df['exam_score'].median(), color='orange', linestyle='--', label=f"Median={df['exam_score'].median():.1f}")
axes[0].set_title('Distribution of Exam Scores')
axes[0].set_xlabel('Exam Score')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].boxplot(df['exam_score'], patch_artist=True,
                boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Box Plot – Exam Score')
axes[1].set_ylabel('Exam Score')
axes[1].set_xticks([1], ['Exam Score'])

plt.suptitle('Target Variable Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot_target_distribution.png', bbox_inches='tight')
plt.show()
print(f"Skewness: {df['exam_score'].skew():.3f}  |  Kurtosis: {df['exam_score'].kurt():.3f}")

# --- Cell 11 ---
# =============================================================
# SECTION 4 - DATA PREPROCESSING
# Changes applied per project requirements:
#   1. Mobile + TV      --> merged as 'screen_time'   (total)
#   2. Exercise + Extra --> merged as 'active_time'   (total)
#   3. Friends + Family --> merged as 'social_time'   (total)
#   4. Stress (1-10)    --> binary 'stressed' (Yes/No -> 1/0)
#   5. Gender           --> label-encoded (Female=0, Male=1)
#   6. Grade category   --> derived from exam_score
#   7. Outlier report   --> IQR method on merged columns
# =============================================================

df_processed = df.copy()

# ── 1. Merge Screen Time: Mobile + TV ----------------------------------------
df_processed['screen_time'] = (
    df_processed['mobile_usage_hours'] + df_processed['tv_hours']
).round(2)
# Keep originals for EDA reference but drop before modelling
print(f'screen_time range : {df_processed["screen_time"].min():.1f} - {df_processed["screen_time"].max():.1f} hrs')

# ── 2. Merge Active Time: Exercise + Extracurricular -------------------------
df_processed['active_time'] = (
    df_processed['exercise_hours'] + df_processed['extracurricular_hrs']
).round(2)
print(f'active_time range : {df_processed["active_time"].min():.1f} - {df_processed["active_time"].max():.1f} hrs')

# ── 3. Merge Social Time: Friends + Family ------------------------------------
df_processed['social_time'] = (
    df_processed['friends_time_hrs'] + df_processed['family_time_hrs']
).round(2)
print(f'social_time range : {df_processed["social_time"].min():.1f} - {df_processed["social_time"].max():.1f} hrs')

# ── 4. Stress: numeric (1-10) --> binary Yes/No -> 1/0 -----------------------
# Threshold: stress_level >= 6 is considered 'stressed'
df_processed['stressed'] = (df_processed['stress_level'] >= 6).astype(int)
stress_pct = df_processed['stressed'].mean() * 100
print(f'stressed=1 (Yes) : {stress_pct:.1f}%  |  stressed=0 (No) : {100-stress_pct:.1f}%')

# ── 5. Gender: encode --------------------------------------------------------
le = LabelEncoder()
df_processed['gender_encoded'] = le.fit_transform(df_processed['gender'])  # Female=0, Male=1
print(f'gender encoded    : {dict(zip(le.classes_, le.transform(le.classes_)))}')

# ── 6. Grade category --------------------------------------------------------
def grade_category(score):
    if score >= 85:   return 'A (Excellent)'
    elif score >= 70: return 'B (Good)'
    elif score >= 55: return 'C (Average)'
    elif score >= 40: return 'D (Below Avg)'
    else:             return 'F (Fail)'

df_processed['grade'] = df_processed['exam_score'].apply(grade_category)

# ── 7. Outlier report (IQR) on final merged columns --------------------------
num_cols = [
    'sleep_hours', 'study_hours', 'screen_time',
    'active_time', 'social_time', 'attendance_pct', 'exam_score'
]
outlier_counts = {}
for col in num_cols:
    Q1, Q3 = df_processed[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((df_processed[col] < Q1 - 1.5*IQR) |
             (df_processed[col] > Q3 + 1.5*IQR)).sum()
    outlier_counts[col] = n_out

print('\nOutlier Count per Feature (IQR method):')
print(pd.Series(outlier_counts, name='Outliers').sort_values(ascending=False))

# ── 8. Show processed column summary -----------------------------------------
print('\nProcessed Dataset Columns:')
display_cols = ['sleep_hours','study_hours','screen_time','active_time',
                'social_time','attendance_pct','stressed','gender_encoded',
                'exam_score','grade']
print(df_processed[display_cols].head(8).to_string(index=False))

# --- Cell 12 ---
# Grade distribution
grade_order  = ['A (Excellent)','B (Good)','C (Average)','D (Below Avg)','F (Fail)']
grade_colors = ['#2ecc71','#3498db','#f1c40f','#e67e22','#e74c3c']

grade_counts = df_processed['grade'].value_counts().reindex(grade_order)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Bar chart
axes[0].bar(grade_order, grade_counts, color=grade_colors, edgecolor='white')
axes[0].set_title('Grade Distribution (Count)')
axes[0].set_xlabel('Grade Category')
axes[0].set_ylabel('Count')
for i, v in enumerate(grade_counts):
    axes[0].text(i, v+20, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(grade_counts, labels=grade_order, colors=grade_colors,
            autopct='%1.1f%%', startangle=140)
axes[1].set_title('Grade Proportion')

# Stressed vs Not Stressed by grade
stress_grade = df_processed.groupby('grade')['stressed'].mean() * 100
stress_grade = stress_grade.reindex(grade_order)
axes[2].bar(grade_order, stress_grade, color=grade_colors, edgecolor='white')
axes[2].set_title('% Stressed Students by Grade')
axes[2].set_ylabel('% Stressed (Yes)')
axes[2].set_ylim(0, 100)
for i, v in enumerate(stress_grade):
    axes[2].text(i, v+1, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('Academic Performance & Stress Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_grade_distribution.png', bbox_inches='tight')
plt.show()


# --- Cell 14 ---
# Correlation heatmap using MERGED feature set
corr_cols = ['sleep_hours','study_hours','screen_time','active_time',
             'social_time','attendance_pct','stressed','gender_encoded','exam_score']
corr_matrix = df_processed[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap (Merged Features)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('plot_correlation_heatmap.png', bbox_inches='tight')
plt.show()

# Top correlations with exam_score
score_corr = corr_matrix['exam_score'].drop('exam_score').sort_values(key=abs, ascending=False)
print('Feature correlations with Exam Score (merged):')
print(score_corr.round(3).to_string())


# --- Cell 15 ---
# Scatter plots: top features vs exam score (using merged features)
merged_features = ['study_hours','screen_time','active_time',
                   'attendance_pct','sleep_hours','social_time']
colors_map = {'A (Excellent)':'#2ecc71','B (Good)':'#3498db',
              'C (Average)':'#f1c40f','D (Below Avg)':'#e67e22','F (Fail)':'#e74c3c'}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feat in enumerate(merged_features):
    for grade, grp in df_processed.groupby('grade'):
        axes[i].scatter(grp[feat], grp['exam_score'],
                        c=colors_map[grade], alpha=0.3, s=12, label=grade)
    m, b = np.polyfit(df_processed[feat], df_processed['exam_score'], 1)
    x_line = np.linspace(df_processed[feat].min(), df_processed[feat].max(), 100)
    axes[i].plot(x_line, m*x_line+b, 'k--', linewidth=1.5, label='Trend')
    r = df_processed[[feat,'exam_score']].corr().iloc[0,1]
    axes[i].set_title(f'{feat}  (r={r:.2f})')
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Exam Score')

handles = [mpatches.Patch(color=c, label=g) for g, c in colors_map.items()]
fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01))
plt.suptitle('Merged Feature Relationships with Exam Score', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_scatter_top_features.png', bbox_inches='tight')
plt.show()


# --- Cell 16 ---
# Violin plots using merged features
plot_features = ['study_hours','sleep_hours','screen_time',
                 'attendance_pct','active_time','social_time']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, feat in enumerate(plot_features):
    sns.violinplot(data=df_processed, x='grade', y=feat,
                   order=grade_order, palette=grade_colors,
                   inner='quartile', ax=axes[i])
    label = feat.replace('_',' ').title()
    if feat == 'screen_time':  label = 'Screen Time (Mobile+TV)'
    if feat == 'active_time':  label = 'Active Time (Exercise+Extra)'
    if feat == 'social_time':  label = 'Social Time (Friends+Family)'
    axes[i].set_title(label)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=15)

plt.suptitle('Merged Feature Distribution by Grade Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_violin_by_grade.png', bbox_inches='tight')
plt.show()


# --- Cell 17 ---
# Gender analysis using merged features
plot_features = ['study_hours','sleep_hours','screen_time',
                 'attendance_pct','active_time','social_time']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Score distribution by gender
for gender, color in [('Male','steelblue'), ('Female','salmon')]:
    subset = df_processed[df_processed['gender']==gender]['exam_score']
    axes[0].hist(subset, bins=30, alpha=0.6, label=gender, color=color, edgecolor='white')
axes[0].set_title('Score Distribution by Gender')
axes[0].set_xlabel('Exam Score')
axes[0].legend()

# Mean scores per merged feature by gender
gender_means = df_processed.groupby('gender')[plot_features].mean().T
x = np.arange(len(gender_means))
w = 0.35
axes[1].bar(x-w/2, gender_means['Female'], w, label='Female', color='salmon')
axes[1].bar(x+w/2, gender_means['Male'],   w, label='Male',   color='steelblue')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f.replace('_',' ') for f in gender_means.index],
                         rotation=30, ha='right', fontsize=8)
axes[1].set_title('Mean Feature Values by Gender (Merged)')
axes[1].legend()

# Stressed distribution by gender
stress_gender = df_processed.groupby('gender')['stressed'].mean() * 100
axes[2].bar(stress_gender.index, stress_gender.values,
            color=['salmon','steelblue'], edgecolor='white', width=0.4)
axes[2].set_title('% Stressed Students by Gender')
axes[2].set_ylabel('% Stressed (Yes)')
axes[2].set_ylim(0, 100)
for i, (g, v) in enumerate(stress_gender.items()):
    axes[2].text(i, v+1.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.suptitle('Gender Analysis (Merged Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_gender_analysis.png', bbox_inches='tight')
plt.show()

print('Mean Exam Score by Gender:')
print(df_processed.groupby('gender')['exam_score'].agg(['mean','std','count']).round(2))


# --- Cell 18 ---
# Pair plot using merged features (n=500 sample)
sample_df = df_processed.sample(500, random_state=SEED)
pair_features = ['study_hours','sleep_hours','screen_time','attendance_pct','exam_score']

g = sns.pairplot(sample_df[pair_features + ['grade']],
                 hue='grade', hue_order=grade_order,
                 palette=dict(zip(grade_order, grade_colors)),
                 diag_kind='kde', plot_kws={'alpha':0.4, 's':20})
g.fig.suptitle('Pair Plot - Merged Key Features by Grade (n=500)',
               fontsize=14, y=1.01)
plt.savefig('plot_pairplot.png', bbox_inches='tight')
plt.show()


# --- Cell 20 ---
df_model = df_processed.copy()

# FEATURES now use MERGED columns from preprocessing:
#   screen_time  = mobile_usage_hours + tv_hours
#   active_time  = exercise_hours     + extracurricular_hrs
#   social_time  = friends_time_hrs   + family_time_hrs
#   stressed     = 1 if stress_level >= 6 else 0

# ── Engineered features (built on top of merged columns) ----------------------
# 1. Study efficiency: study relative to distraction
df_model['study_efficiency']   = df_model['study_hours'] / (df_model['screen_time'] + 1)

# 2. Health score composite (uses original sub-cols still present in df)
df_model['health_score']       = (
    df_model['sleep_hours'] / 8
    + df_model['exercise_hours'] / 2
    - df_model['stressed'] * 0.4   # stressed binary penalises health score
) / 3 * 100

# 3. Sleep adequacy flag
df_model['sleep_adequate']     = (df_model['sleep_hours'] >= 7).astype(int)

# 4. High screen time flag
df_model['high_screen']        = (df_model['screen_time'] > 5).astype(int)

# 5. Interaction: study x attendance
df_model['study_x_attendance'] = df_model['study_hours'] * df_model['attendance_pct'] / 100

# 6. Stress-adjusted study (stressed=1 reduces effective study)
df_model['stress_adj_study']   = df_model['study_hours'] * (1 - df_model['stressed'] * 0.25)

# ── Final feature set ---------------------------------------------------------
FEATURES = [
    # Merged lifestyle features (8 inputs -> 7 features after merging)
    'sleep_hours', 'study_hours', 'screen_time',
    'active_time', 'social_time', 'attendance_pct',
    'stressed', 'gender_encoded',
    # Engineered
    'study_efficiency', 'health_score', 'sleep_adequate',
    'high_screen', 'study_x_attendance', 'stress_adj_study'
]
TARGET = 'exam_score'

X = df_model[FEATURES]
y = df_model[TARGET]

print(f'Feature matrix shape : {X.shape}')
print(f'\nFinal Features ({len(FEATURES)}):')
for f in FEATURES:
    print(f'  - {f}')
print()
X.describe().round(2)


# --- Cell 21 ---
# Quick feature importance via correlation to target
feat_corr = X.corrwith(y).abs().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
colors = ['#e74c3c' if v > 0 else '#3498db'
          for v in X.corrwith(y).reindex(feat_corr.index)]
ax.barh(feat_corr.index, feat_corr.values, color=colors)
ax.set_title('Absolute Correlation of Features with Exam Score')
ax.set_xlabel('|Correlation|')
for i, v in enumerate(feat_corr.values):
    ax.text(v+0.003, i, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('plot_feature_correlation.png', bbox_inches='tight')
plt.show()

# --- Cell 23 ---
# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

print(f'Training set : {X_train.shape[0]:,} samples')
print(f'Testing  set : {X_test.shape[0]:,} samples')

# --- Cell 24 ---
# ── Compare 8 baseline models ─────────────────────────────────────────────────
models = {
    'Linear Regression':          LinearRegression(),
    'Ridge Regression':           Ridge(alpha=1.0),
    'Lasso Regression':           Lasso(alpha=0.1),
    'Decision Tree':              DecisionTreeRegressor(max_depth=8, random_state=SEED),
    'Random Forest':              RandomForestRegressor(n_estimators=100, random_state=SEED),
    'Gradient Boosting':          GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    'Extra Trees':                ExtraTreesRegressor(n_estimators=100, random_state=SEED),
    'K-Nearest Neighbors':        KNeighborsRegressor(n_neighbors=7),
}

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

results = []
for name, model in models.items():
    X_tr = X_train_sc if name in ['Linear Regression','Ridge Regression',
                                   'Lasso Regression','K-Nearest Neighbors','SVR'] else X_train
    X_te = X_test_sc  if name in ['Linear Regression','Ridge Regression',
                                   'Lasso Regression','K-Nearest Neighbors','SVR'] else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
    results.append({
        'Model':    name,
        'R2':       r2_score(y_test, y_pred),
        'MAE':      mean_absolute_error(y_test, y_pred),
        'RMSE':     np.sqrt(mean_squared_error(y_test, y_pred)),
        'CV_R2':    cv.mean(),
        'CV_Std':   cv.std()
    })
    print(f"  {name:<28} R²={r2_score(y_test,y_pred):.4f}  MAE={mean_absolute_error(y_test,y_pred):.3f}")

results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)
print('\n📊 Model Comparison:')
results_df.round(4)

# --- Cell 25 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

bar_kw = dict(edgecolor='white')
palette = sns.color_palette('viridis', len(results_df))

# R²
axes[0].barh(results_df['Model'], results_df['R2'], color=palette, **bar_kw)
axes[0].set_title('R² Score (higher = better)')
axes[0].set_xlim(0, 1)
for i, v in enumerate(results_df['R2']):
    axes[0].text(v+0.003, i, f'{v:.3f}', va='center', fontsize=8)

# MAE
axes[1].barh(results_df['Model'], results_df['MAE'], color=palette, **bar_kw)
axes[1].set_title('MAE (lower = better)')
for i, v in enumerate(results_df['MAE']):
    axes[1].text(v+0.05, i, f'{v:.2f}', va='center', fontsize=8)

# RMSE
axes[2].barh(results_df['Model'], results_df['RMSE'], color=palette, **bar_kw)
axes[2].set_title('RMSE (lower = better)')
for i, v in enumerate(results_df['RMSE']):
    axes[2].text(v+0.05, i, f'{v:.2f}', va='center', fontsize=8)

plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_model_comparison.png', bbox_inches='tight')
plt.show()

# --- Cell 26 ---
# ── Hyperparameter tuning for best model (Random Forest / Gradient Boosting) ─
print('🔧 Tuning Gradient Boosting...')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [3, 5, 7],
    'learning_rate':[0.05, 0.1, 0.2],
    'subsample':    [0.8, 1.0]
}
gb = GradientBoostingRegressor(random_state=SEED)
gs = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

best_params = gs.best_params_
print(f'Best params : {best_params}')
print(f'Best CV R²  : {gs.best_score_:.4f}')

best_model = gs.best_estimator_
y_pred_best = best_model.predict(X_test)

print('\n🏆 Tuned Gradient Boosting on Test Set:')
print(f'  R²   : {r2_score(y_test, y_pred_best):.4f}')
print(f'  MAE  : {mean_absolute_error(y_test, y_pred_best):.3f}')
print(f'  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_best)):.3f}')
print(f'  MAPE : {mean_absolute_percentage_error(y_test, y_pred_best)*100:.2f}%')

# --- Cell 27 ---
# ── Evaluation plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 1. Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.3, s=12, color='steelblue')
lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect Fit')
axes[0].set_xlabel('Actual Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'Actual vs Predicted  (R²={r2_score(y_test,y_pred_best):.3f})')
axes[0].legend()

# 2. Residuals
residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.3, s=12, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Predicted Score')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residual Plot')

# 3. Residual histogram
axes[2].hist(residuals, bins=40, color='mediumseagreen', edgecolor='white', alpha=0.85)
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Frequency')
axes[2].set_title(f'Residual Distribution  (μ={residuals.mean():.2f}, σ={residuals.std():.2f})')

plt.suptitle('Model Evaluation – Tuned Gradient Boosting', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_model_evaluation.png', bbox_inches='tight')
plt.show()

# --- Cell 28 ---
# ── Feature importance ────────────────────────────────────────────────────────
importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
colors = ['#e74c3c' if f in ['study_hours','attendance_pct','study_x_attendance',
                               'study_efficiency','stress_adj_study']
          else '#3498db' for f in importance.index]
ax.barh(importance.index, importance.values, color=colors)
ax.set_title('Feature Importance – Gradient Boosting', fontsize=13)
ax.set_xlabel('Importance Score')
for i, v in enumerate(importance.values):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
red_p  = mpatches.Patch(color='#e74c3c', label='Study/Academic')
blue_p = mpatches.Patch(color='#3498db', label='Lifestyle')
ax.legend(handles=[red_p, blue_p])
plt.tight_layout()
plt.savefig('plot_feature_importance.png', bbox_inches='tight')
plt.show()

# --- Cell 29 ---
# ── Save model & scaler ───────────────────────────────────────────────────────
model_package = {
    'model':    best_model,
    'scaler':   scaler,
    'features': FEATURES,
    'metadata': {
        'r2':  r2_score(y_test, y_pred_best),
        'mae': mean_absolute_error(y_test, y_pred_best),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_best))
    }
}
with open('scoresense_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print('✅ Model saved to scoresense_model.pkl')

# --- Cell 31 ---
def predict_score(student_data: dict) -> dict:
    """
    Predict exam score for a single student.

    Accepts the MERGED input format used after preprocessing:
      screen_time  = mobile + tv combined
      active_time  = exercise + extracurricular combined
      social_time  = friends + family combined
      stressed     = 'Yes' or 'No'
      gender       = 'Male' or 'Female'

    Returns dict: predicted_score, grade, confidence_interval
    """
    d = student_data.copy()

    # -- Encode categorical inputs -------------------------------------------
    d['gender_encoded'] = 1 if d.get('gender', 'Male') == 'Male' else 0
    d['stressed']       = 1 if d.get('stressed', 'No') == 'Yes' else 0

    # -- Sub-estimates from merged totals ------------------------------------
    exercise_est = d['active_time'] * 0.60

    # -- Engineered features (mirrors Cell 19 exactly) -----------------------
    d['study_efficiency']   = d['study_hours'] / (d['screen_time'] + 1)
    d['health_score']       = (
        d['sleep_hours'] / 8
        + exercise_est   / 2
        - d['stressed']  * 0.4
    ) / 3 * 100
    d['sleep_adequate']     = int(d['sleep_hours'] >= 7)
    d['high_screen']        = int(d['screen_time'] > 5)
    d['study_x_attendance'] = d['study_hours'] * d['attendance_pct'] / 100
    d['stress_adj_study']   = d['study_hours'] * (1 - d['stressed'] * 0.25)

    # -- Build feature row and predict ---------------------------------------
    row   = pd.DataFrame([[d[f] for f in FEATURES]], columns=FEATURES)
    score = float(best_model.predict(row)[0])
    score = max(30.0, min(100.0, score))

    rmse  = model_package['metadata']['rmse']
    ci_lo = max(0,   round(score - 1.96 * rmse, 2))
    ci_hi = min(100, round(score + 1.96 * rmse, 2))

    return {
        'predicted_score':     round(score, 2),
        'grade':               grade_category(score),
        'confidence_interval': (ci_lo, ci_hi)
    }


# -- Demo students using MERGED input format ---------------------------------
students = {
    'High Performer': {
        'sleep_hours':   8.0,
        'study_hours':   7.0,
        'screen_time':   1.5,   # mobile(0.9) + tv(0.6)
        'active_time':   4.0,   # exercise(2.4) + extra(1.6)
        'social_time':   3.5,   # friends(1.75) + family(1.75)
        'attendance_pct': 95.0,
        'stressed':      'No',
        'gender':        'Female'
    },
    'Average Student': {
        'sleep_hours':   7.0,
        'study_hours':   4.0,
        'screen_time':   5.5,   # mobile(3.3) + tv(2.2)
        'active_time':   2.0,   # exercise(1.2) + extra(0.8)
        'social_time':   4.5,   # friends(2.25) + family(2.25)
        'attendance_pct': 78.0,
        'stressed':      'No',
        'gender':        'Male'
    },
    'Struggling Student': {
        'sleep_hours':   5.5,
        'study_hours':   1.5,
        'screen_time':   9.5,   # mobile(5.7) + tv(3.8)
        'active_time':   0.5,   # exercise(0.3) + extra(0.2)
        'social_time':   5.0,   # friends(2.5) + family(2.5)
        'attendance_pct': 55.0,
        'stressed':      'Yes',
        'gender':        'Male'
    }
}

print('=' * 62)
print('  SCORESENSE AI -- PREDICTION RESULTS')
print('=' * 62)
for name, data in students.items():
    result = predict_score(data)
    print(f"\n  {name}")
    print(f"    Predicted Score : {result['predicted_score']}")
    print(f"    Grade           : {result['grade']}")
    print(f"    95% CI          : {result['confidence_interval']}")


# --- Cell 33 ---
# Benchmarks from top-quartile students using MERGED column names
top_quartile = df_processed[
    df_processed['exam_score'] >= df_processed['exam_score'].quantile(0.75)
]
BENCHMARKS = top_quartile[[
    'study_hours', 'sleep_hours', 'screen_time',
    'active_time', 'social_time', 'attendance_pct'
]].mean().round(2)
BENCHMARKS['stressed'] = 0   # top students are mostly not stressed

print('Top-Quartile Benchmarks (merged features):')
print(BENCHMARKS)


def generate_recommendations(student_data: dict) -> dict:
    """
    Generate personalised recommendations via gap analysis.
    Accepts the same MERGED input dict as predict_score().
    """
    pred  = predict_score(student_data)
    score = pred['predicted_score']
    grade = pred['grade']
    recs  = []
    tips  = []
    d     = student_data

    # Sub-estimates for display
    mobile_est   = round(d['screen_time'] * 0.60, 1)
    tv_est       = round(d['screen_time'] * 0.40, 1)
    exercise_est = round(d['active_time'] * 0.60, 1)
    extra_est    = round(d['active_time'] * 0.40, 1)

    # 1. Study hours
    gap = BENCHMARKS['study_hours'] - d['study_hours']
    if gap > 0.5:
        recs.append({
            'category': 'Study Hours',
            'current':  d['study_hours'],
            'target':   BENCHMARKS['study_hours'],
            'message':  f"Increase by {gap:.1f}h/day. Use the Pomodoro technique (25-min focused blocks).",
            'impact':   0.40
        })

    # 2. Sleep
    if d['sleep_hours'] < 7:
        recs.append({
            'category': 'Sleep Quality',
            'current':  d['sleep_hours'],
            'target':   7.5,
            'message':  f"Aim for 7-9 hrs. Poor sleep reduces memory consolidation. Gain {7 - d['sleep_hours']:.1f}h more.",
            'impact':   0.20
        })
    elif d['sleep_hours'] > 9.5:
        recs.append({
            'category': 'Sleep Quality',
            'current':  d['sleep_hours'],
            'target':   8.0,
            'message':  'Oversleeping reduces alertness. Target 7-9 hours.',
            'impact':   0.10
        })

    # 3. Screen time
    if d['screen_time'] > BENCHMARKS['screen_time'] + 0.8:
        excess = d['screen_time'] - BENCHMARKS['screen_time']
        recs.append({
            'category': 'Screen Time',
            'current':  d['screen_time'],
            'target':   BENCHMARKS['screen_time'],
            'message':  (f"Total screen {d['screen_time']:.1f}h (~{mobile_est}h mobile + ~{tv_est}h TV). "
                         f"Reduce by {excess:.1f}h. Enable app timers."),
            'impact':   0.15
        })

    # 4. Attendance
    if d['attendance_pct'] < BENCHMARKS['attendance_pct'] - 5:
        recs.append({
            'category': 'Attendance',
            'current':  d['attendance_pct'],
            'target':   BENCHMARKS['attendance_pct'],
            'message':  (f"Your attendance is {d['attendance_pct']:.0f}%. "
                         f"Top students average {BENCHMARKS['attendance_pct']:.0f}%. "
                         "Missing class creates gaps hard to recover."),
            'impact':   0.18
        })

    # 5. Active time
    if d['active_time'] < BENCHMARKS['active_time'] - 0.4:
        recs.append({
            'category': 'Active Time',
            'current':  d['active_time'],
            'target':   BENCHMARKS['active_time'],
            'message':  (f"Active time {d['active_time']:.1f}h (~{exercise_est}h exercise + ~{extra_est}h extra). "
                         "Exercise boosts cognition; extracurriculars reduce burnout."),
            'impact':   0.08
        })

    # 6. Stress
    if d.get('stressed') == 'Yes':
        recs.append({
            'category': 'Stress Management',
            'current':  'Yes',
            'target':   'No',
            'message':  ('High stress impairs memory. Try: time-blocking, '
                         '10-min meditation, breaking tasks into small goals.'),
            'impact':   0.12
        })

    recs.sort(key=lambda x: x['impact'], reverse=True)

    if not recs:
        tips.append('Great habits! Maintain consistency and consider peer tutoring.')

    return {
        'predicted_score':     score,
        'grade':               grade,
        'confidence_interval': pred['confidence_interval'],
        'recommendations':     recs,
        'general_tips':        tips
    }


print('Recommendation engine ready!')


# --- Cell 34 ---
# ── Demo: recommendations for Struggling Student ──────────────────────────────
target_student = students['Struggling Student']
report = generate_recommendations(target_student)

print('='*65)
print('  ✨ SCORESENSE AI – PERSONALIZED IMPROVEMENT REPORT')
print('='*65)
print(f"  Predicted Score : {report['predicted_score']}")
print(f"  Grade           : {report['grade']}")
print(f"  95% CI          : {report['confidence_interval']}")
print()
print('  📌 RECOMMENDATIONS (ranked by impact):')
print('-'*65)
for i, rec in enumerate(report['recommendations'], 1):
    print(f"  {i}. {rec['category']}")
    print(f"     Current: {rec['current']}  →  Target: {rec['target']}  (Impact: {rec['impact']:.0%})")
    print(f"     💬 {rec['message']}")
    print()

if report['general_tips']:
    for tip in report['general_tips']:
        print(f'  {tip}')

# --- Cell 35 ---
def radar_chart(student_data, title='Student vs Benchmark'):
    """Radar chart using merged feature set."""
    d          = student_data
    stressed_n = 1 if d.get('stressed') == 'Yes' else 0
    categories = ['Study\nHours', 'Sleep\nHours', 'Attendance',
                  'Active\nTime', 'Low\nScreen', 'Not\nStressed']

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    student_vals = [
        norm(d['study_hours'],          0,  10),
        norm(d['sleep_hours'],          3,  10),
        norm(d['attendance_pct'],      40, 100),
        norm(d['active_time'],          0,   8),
        norm(12 - d['screen_time'],     2,  12),
        norm(1 - stressed_n,            0,   1),
    ]
    bench_vals = [
        norm(BENCHMARKS['study_hours'],    0,  10),
        norm(BENCHMARKS['sleep_hours'],    3,  10),
        norm(BENCHMARKS['attendance_pct'],40, 100),
        norm(BENCHMARKS['active_time'],    0,   8),
        norm(12 - BENCHMARKS['screen_time'], 2, 12),
        1.0,   # top students: not stressed
    ]

    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    student_vals += [student_vals[0]]
    bench_vals   += [bench_vals[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.plot(angles, bench_vals,   'g-', linewidth=2, label='Top-Quartile Benchmark')
    ax.fill(angles, bench_vals,   'g',  alpha=0.15)
    ax.plot(angles, student_vals, 'r-', linewidth=2, label='This Student')
    ax.fill(angles, student_vals, 'r',  alpha=0.15)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('plot_radar_chart.png', bbox_inches='tight')
    plt.show()


radar_chart(students['Struggling Student'],
            title='Struggling Student vs Top-Quartile Benchmark')


# --- Cell 36 ---
# Compare all three demo students using MERGED feature labels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
feature_labels = ['Study\nHrs', 'Sleep\nHrs', 'Screen\nTime',
                   'Active\nTime', 'Social\nTime', 'Attend\n/10']

for ax, (name, data) in zip(axes, students.items()):
    vals = [
        data['study_hours'],
        data['sleep_hours'],
        data['screen_time'],
        data['active_time'],
        data['social_time'],
        data['attendance_pct'] / 10
    ]
    bench = [
        BENCHMARKS['study_hours'],
        BENCHMARKS['sleep_hours'],
        BENCHMARKS['screen_time'],
        BENCHMARKS['active_time'],
        BENCHMARKS['social_time'],
        BENCHMARKS['attendance_pct'] / 10
    ]
    x = np.arange(len(vals))
    ax.bar(x - 0.2, vals,  0.4, label='Student',   color='steelblue', alpha=0.85)
    ax.bar(x + 0.2, bench, 0.4, label='Benchmark', color='green',     alpha=0.60)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=8)
    res = predict_score(data)
    stressed_label = data.get('stressed', 'No')
    ax.set_title(
        f"{name}\nScore: {res['predicted_score']} | {res['grade']}\nStressed: {stressed_label}",
        fontsize=9
    )
    ax.legend(fontsize=8)

plt.suptitle('Student Profiles vs Top-Quartile Benchmark (Merged Features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_student_profiles.png', bbox_inches='tight')
plt.show()


# --- Cell 38 ---
# Redesigned Streamlit App - Save as app.py and run: streamlit run app.py
# All features properly wired to the trained model
app_source = '''# -*- coding: utf-8 -*-
# ============================================================
#  ScoreSense AI  -  Official Web Interface
#  Run: streamlit run app.py
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

st.set_page_config(
    page_title="ScoreSense AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
#  PROFESSIONAL CSS
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ═══════════════════════════════════════
     ROOT TOKENS
  ═══════════════════════════════════════ */
  :root {
    --bg-base:      #060910;
    --bg-surface:   #0c1220;
    --bg-elevated:  #111827;
    --bg-card:      #141e2e;
    --border:       rgba(255,255,255,0.07);
    --border-glow:  rgba(99,179,237,0.25);
    --text-primary: #f0f4ff;
    --text-sec:     #8899bb;
    --text-muted:   #445577;
    --accent:       #3b82f6;
    --accent-glow:  rgba(59,130,246,0.35);
    --gold:         #f59e0b;
    --gold-glow:    rgba(245,158,11,0.3);
    --green:        #10b981;
    --red:          #ef4444;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: var(--bg-base) !important;
  }

  /* ═══════════════════════════════════════
     HEADER BANNER
  ═══════════════════════════════════════ */
  .ss-header {
    position: relative;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 0 0 20px 20px;
    padding: 28px 36px 24px;
    margin-bottom: 28px;
    overflow: hidden;
  }
  .ss-header::before {
    content: 'SS';
    position: absolute;
    right: 30px; top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif;
    font-size: 9rem;
    font-weight: 800;
    color: rgba(59,130,246,0.04);
    letter-spacing: -8px;
    pointer-events: none;
    line-height: 1;
  }
  .ss-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
  }
  .ss-header-title span { color: var(--accent); }
  .ss-header-sub {
    color: var(--text-sec);
    font-size: 0.88rem;
    margin: 0 0 14px 0;
    font-weight: 400;
  }
  .ss-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(59,130,246,0.1);
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-right: 6px;
    margin-bottom: 4px;
  }
  .ss-badge-gold {
    background: rgba(245,158,11,0.1);
    color: #fcd34d;
    border-color: rgba(245,158,11,0.2);
  }
  .ss-badge-green {
    background: rgba(16,185,129,0.1);
    color: #6ee7b7;
    border-color: rgba(16,185,129,0.2);
  }

  /* ═══════════════════════════════════════
     LEFT PANEL
  ═══════════════════════════════════════ */
  .input-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 20px;
  }
  .input-panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    letter-spacing: 0.02em;
  }
  .input-panel-sub {
    color: var(--text-muted);
    font-size: 0.74rem;
    margin: 0 0 18px 0;
  }
  .divider-thin {
    height: 1px;
    background: var(--border);
    margin: 14px 0;
  }

  /* Field labels */
  .f-label {
    color: var(--text-sec);
    font-size: 0.69rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 14px 0 1px 0;
  }
  .f-hint {
    color: var(--text-muted);
    font-size: 0.71rem;
    margin: 0 0 4px 0;
  }
  .f-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.13);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #7dd3fc;
    margin: 3px 0 8px 0;
  }

  /* ═══════════════════════════════════════
     SCORE CARD
  ═══════════════════════════════════════ */
  .score-card {
    position: relative;
    border-radius: 18px;
    padding: 32px 28px;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .score-card-noise {
    position: absolute;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    opacity: 0.4;
  }
  .score-card-orb {
    position: absolute;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    top: -60px; right: -60px;
  }
  .score-card-orb2 {
    position: absolute;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
    bottom: -40px; left: 20px;
  }
  .score-card-emoji {
    font-size: 2.4rem;
    margin-bottom: 4px;
    display: block;
  }
  .score-card-num {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    margin: 0 0 2px 0;
    letter-spacing: -2px;
  }
  .score-card-grade {
    font-size: 1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.8);
    letter-spacing: 0.04em;
    margin: 6px 0 4px 0;
  }
  .score-card-msg {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    margin: 0;
  }
  .score-card-ci {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,0,0,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
    margin-top: 12px;
  }

  /* ═══════════════════════════════════════
     METRIC TILES
  ═══════════════════════════════════════ */
  .metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
  }
  .metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 12px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .metric-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
  }
  .mt-blue::after  { background: var(--accent); }
  .mt-gray::after  { background: var(--text-muted); }
  .mt-green::after { background: var(--green); }
  .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 3px;
  }
  .metric-lbl {
    font-size: 0.64rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 4px;
  }
  .metric-delta-pos { font-size: 0.73rem; color: var(--green); font-weight: 600; }
  .metric-delta-neg { font-size: 0.73rem; color: var(--red); font-weight: 600; }
  .metric-delta-neu { font-size: 0.73rem; color: var(--text-sec); font-weight: 500; }

  /* ═══════════════════════════════════════
     PROGRESS BAR
  ═══════════════════════════════════════ */
  .prog-track {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0 4px;
  }
  .prog-fill {
    height: 100%;
    border-radius: 8px;
    position: relative;
  }
  .prog-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 40px; height: 100%;
    background: rgba(255,255,255,0.3);
    border-radius: 0 8px 8px 0;
    filter: blur(4px);
  }
  .prog-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 14px;
  }

  /* ═══════════════════════════════════════
     SECTION HEADING
  ═══════════════════════════════════════ */
  .sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 12px 0;
  }
  .sec-head-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
  }
  .sec-head-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .sec-head-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }

  /* ═══════════════════════════════════════
     HABIT TABLE
  ═══════════════════════════════════════ */
  .habit-table { width: 100%; }
  .habit-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .habit-row:last-child { border-bottom: none; }
  .h-feat {
    font-size: 0.78rem;
    color: var(--text-sec);
    min-width: 90px;
  }
  .h-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-primary);
    font-weight: 500;
  }
  .h-tgt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    color: var(--green);
    opacity: 0.7;
  }
  .h-warn { color: #fbbf24; }

  /* ═══════════════════════════════════════
     RECOMMENDATIONS
  ═══════════════════════════════════════ */
  .rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
  }
  .rec-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
  }
  .rec-high::before   { background: var(--red); }
  .rec-medium::before { background: var(--gold); }
  .rec-low::before    { background: var(--accent); }
  .rec-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .rec-impact-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    padding: 1px 7px;
    border-radius: 20px;
    font-weight: 600;
    margin-left: auto;
  }
  .rec-high   .rec-impact-badge { background: rgba(239,68,68,0.15);  color: #fca5a5; }
  .rec-medium .rec-impact-badge { background: rgba(245,158,11,0.15); color: #fcd34d; }
  .rec-low    .rec-impact-badge { background: rgba(59,130,246,0.15); color: #93c5fd; }
  .rec-body {
    font-size: 0.8rem;
    color: var(--text-sec);
    line-height: 1.55;
  }

  /* ═══════════════════════════════════════
     EMPTY STATE
  ═══════════════════════════════════════ */
  .empty-hero {
    text-align: center;
    padding: 36px 24px 20px;
  }
  .empty-hero-icon {
    font-size: 3.5rem;
    margin-bottom: 14px;
    display: block;
  }
  .empty-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }
  .empty-hero-sub {
    color: var(--text-sec);
    font-size: 0.88rem;
    line-height: 1.6;
    max-width: 360px;
    margin: 0 auto 24px;
  }
  .feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 24px;
  }
  .feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    text-align: left;
  }
  .feature-card-icon { font-size: 1.4rem; margin-bottom: 6px; }
  .feature-card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 3px;
  }
  .feature-card-desc {
    font-size: 0.74rem;
    color: var(--text-muted);
    line-height: 1.45;
  }

  /* ═══════════════════════════════════════
     PREDICT BUTTON
  ═══════════════════════════════════════ */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 16px 20px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.3),
                0 4px 20px rgba(37,99,235,0.4) !important;
    transition: all 0.25s ease !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 0 1px rgba(59,130,246,0.5),
                0 8px 30px rgba(37,99,235,0.6) !important;
    transform: translateY(-2px) !important;
  }

  /* ═══════════════════════════════════════
     SLIDER / RADIO TWEAKS
  ═══════════════════════════════════════ */
  div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
  }
  div[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    text-align: center !important;
  }

  /* ═══════════════════════════════════════
     HIDE CHROME
  ═══════════════════════════════════════ */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open("scoresense_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    pkg      = load_model()
    model    = pkg["model"]
    FEATURES = pkg["features"]
except FileNotFoundError:
    st.error("scoresense_model.pkl not found. Please run the notebook first to train the model.")
    st.stop()


# ============================================================
#  CONSTANTS
# ============================================================
BENCHMARKS = {
    "study_hours":    5.8,
    "sleep_hours":    7.4,
    "screen_time":    3.9,
    "active_time":    3.3,
    "social_time":    4.2,
    "attendance_pct": 87.0,
    "stressed":       0,
}

GRADE_CONFIG = {
    "A (Excellent)": {"color": "#059669", "bg": "linear-gradient(135deg,#059669,#047857)", "emoji": "🏆", "msg": "Outstanding performance!"},
    "B (Good)":      {"color": "#2563eb", "bg": "linear-gradient(135deg,#2563eb,#1d4ed8)", "emoji": "👍", "msg": "Good work, keep it up!"},
    "C (Average)":   {"color": "#d97706", "bg": "linear-gradient(135deg,#d97706,#b45309)", "emoji": "📘", "msg": "Room for improvement."},
    "D (Below Avg)": {"color": "#ea580c", "bg": "linear-gradient(135deg,#ea580c,#c2410c)", "emoji": "⚠️",  "msg": "Needs focused effort."},
    "F (Fail)":      {"color": "#dc2626", "bg": "linear-gradient(135deg,#dc2626,#b91c1c)", "emoji": "❗", "msg": "Immediate action required!"},
}

REC_ICONS = {
    "high":   ("🔴", "#ef4444"),
    "medium": ("🟡", "#f59e0b"),
    "low":    ("🔵", "#3b82f6"),
}

def grade_category(score):
    if score >= 85:   return "A (Excellent)"
    elif score >= 70: return "B (Good)"
    elif score >= 55: return "C (Average)"
    elif score >= 40: return "D (Below Avg)"
    else:             return "F (Fail)"


# ============================================================
#  FEATURE BUILDER & PREDICTOR
# ============================================================
def build_features(d: dict):
    screen_time  = d["screen_time"]
    active_time  = d["active_time"]
    social_time  = d["social_time"]
    stressed_bin = 1 if d["stressed"] == "Yes" else 0
    gender_enc   = 1 if d["gender"] == "Male" else 0
    study_hours  = d["study_hours"]
    sleep_hours  = d["sleep_hours"]
    attendance   = d["attendance_pct"]

    mobile_est   = round(screen_time * 0.60, 1)
    tv_est       = round(screen_time * 0.40, 1)
    exercise_est = round(active_time * 0.60, 1)
    extra_est    = round(active_time * 0.40, 1)

    study_efficiency   = study_hours / (screen_time + 1)
    health_score       = (sleep_hours / 8 + exercise_est / 2 - stressed_bin * 0.4) / 3 * 100
    sleep_adequate     = int(sleep_hours >= 7)
    high_screen        = int(screen_time > 5)
    study_x_attendance = study_hours * attendance / 100
    stress_adj_study   = study_hours * (1 - stressed_bin * 0.25)

    feature_map = {
        "sleep_hours": sleep_hours, "study_hours": study_hours,
        "screen_time": screen_time, "active_time": active_time,
        "social_time": social_time, "attendance_pct": attendance,
        "stressed": stressed_bin,   "gender_encoded": gender_enc,
        "study_efficiency": study_efficiency, "health_score": health_score,
        "sleep_adequate": sleep_adequate, "high_screen": high_screen,
        "study_x_attendance": study_x_attendance, "stress_adj_study": stress_adj_study,
    }
    row = pd.DataFrame([[feature_map[f] for f in FEATURES]], columns=FEATURES)
    expanded = {"mobile_est": mobile_est, "tv_est": tv_est,
                "exercise_est": exercise_est, "extra_est": extra_est}
    return row, expanded

def predict(d: dict):
    row, expanded = build_features(d)
    score = float(model.predict(row)[0])
    score = max(30.0, min(100.0, score))
    return round(score, 2), grade_category(score), expanded


# ============================================================
#  RECOMMENDATIONS
# ============================================================
def make_recommendations(d: dict, expanded: dict):
    recs = []
    gap = BENCHMARKS["study_hours"] - d["study_hours"]
    if gap > 0.5:
        recs.append({"icon": "📚", "title": "Increase Study Time",
            "detail": (f"You study {d['study_hours']:.1f}h/day. "
                       f"Aim for {BENCHMARKS['study_hours']:.1f}h. "
                       f"Add {gap:.1f}h using Pomodoro (25-min focused blocks + 5-min breaks)."),
            "impact": 0.40, "level": "high"})

    if d["sleep_hours"] < 7:
        recs.append({"icon": "😴", "title": "Improve Sleep Duration",
            "detail": (f"You sleep {d['sleep_hours']:.1f}h. Aim for 7-9 hours. "
                       "Poor sleep reduces memory consolidation by up to 40%."),
            "impact": 0.20, "level": "medium"})
    elif d["sleep_hours"] > 9.5:
        recs.append({"icon": "😴", "title": "Reduce Oversleeping",
            "detail": "Sleeping 10+ hours causes grogginess. Target 7-9h for peak cognitive performance.",
            "impact": 0.10, "level": "low"})

    if d["screen_time"] > BENCHMARKS["screen_time"] + 0.8:
        excess = d["screen_time"] - BENCHMARKS["screen_time"]
        recs.append({"icon": "📱", "title": "Cut Down Screen Time",
            "detail": (f"Your screen time: {d['screen_time']:.1f}h "
                       f"(~{expanded['mobile_est']:.1f}h mobile + ~{expanded['tv_est']:.1f}h TV). "
                       f"Reduce by {excess:.1f}h — try grayscale mode and app timers."),
            "impact": 0.15, "level": "medium"})

    if d["attendance_pct"] < BENCHMARKS["attendance_pct"] - 5:
        recs.append({"icon": "🏫", "title": "Improve Class Attendance",
            "detail": (f"Attendance: {d['attendance_pct']:.0f}%. "
                       f"Target: {BENCHMARKS['attendance_pct']:.0f}%. "
                       "Every missed class = 3x more self-study needed to catch up."),
            "impact": 0.18, "level": "medium"})

    if d["active_time"] < BENCHMARKS["active_time"] - 0.4:
        recs.append({"icon": "🏃", "title": "Increase Active Time",
            "detail": (f"Active time: {d['active_time']:.1f}h "
                       f"(~{expanded['exercise_est']:.1f}h exercise + ~{expanded['extra_est']:.1f}h extracurricular). "
                       "Exercise increases BDNF — a brain protein that improves learning speed."),
            "impact": 0.08, "level": "low"})

    if d["stressed"] == "Yes":
        recs.append({"icon": "🧘", "title": "Manage Stress",
            "detail": "Chronic stress shrinks the hippocampus (memory centre). "
                      "Try: 10-min daily meditation, time-blocking, or speaking to a counselor.",
            "impact": 0.12, "level": "low"})

    recs.sort(key=lambda x: x["impact"], reverse=True)
    return recs


# ============================================================
#  RADAR CHART  —  You Now vs Your Potential
# ============================================================
def radar_chart(d: dict):
    """Radar: You Now (orange solid) vs Your Potential (blue dashed). Dark-native."""
    stressed_bin = 1 if d["stressed"] == "Yes" else 0
    categories   = ["Study\nHours", "Sleep\nHours", "Attendance",
                    "Active\nTime", "Low\nScreen", "Not\nStressed"]

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    current_vals = [
        norm(d["study_hours"],        0,  10),
        norm(d["sleep_hours"],        3,  10),
        norm(d["attendance_pct"],    40, 100),
        norm(d["active_time"],        0,   8),
        norm(12 - d["screen_time"],   2,  12),
        norm(1 - stressed_bin,        0,   1),
    ]
    HEALTHY = {
        "study_hours":    min(d["study_hours"] + 1.5, 8.0),
        "sleep_hours":    max(min(d["sleep_hours"], 9.0), 7.0),
        "attendance_pct": min(d["attendance_pct"] + 10, 95.0),
        "active_time":    min(d["active_time"] + 1.0, 5.0),
        "screen_time":    max(d["screen_time"] - 1.5, 2.0),
        "stressed":       0,
    }
    potential_vals = [
        norm(HEALTHY["study_hours"],        0,  10),
        norm(HEALTHY["sleep_hours"],        3,  10),
        norm(HEALTHY["attendance_pct"],    40, 100),
        norm(HEALTHY["active_time"],        0,   8),
        norm(12 - HEALTHY["screen_time"],   2,  12),
        1.0,
    ]
    potential_vals = [max(c, p) for c, p in zip(current_vals, potential_vals)]

    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    cv     = current_vals   + [current_vals[0]]
    pv     = potential_vals + [potential_vals[0]]

    BG = "#0c1220"

    # Larger figure for proper visibility
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor=BG)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Grid rings with subtle glow
    for r in [0.25, 0.5, 0.75, 1.0]:
        ring_a = [r] * (N + 1)
        lw  = 0.8 if r < 1.0 else 1.2
        alp = 0.18 if r < 1.0 else 0.30
        ax.plot(angles, ring_a, color="white", linewidth=lw, alpha=alp)

    # Spokes
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 1], color="white", linewidth=0.5, alpha=0.10)

    # Potential zone (blue dashed)
    ax.fill(angles, pv, color="#3b82f6", alpha=0.12)
    ax.plot(angles, pv, color="#60a5fa", linewidth=2.2,
            linestyle="--", dashes=(7, 3))
    # Dots on potential
    ax.scatter(angles[:-1], potential_vals,
               color="#93c5fd", s=28, zorder=4, linewidths=0)

    # Current zone (orange solid)
    ax.fill(angles, cv, color="#f97316", alpha=0.22)
    ax.plot(angles, cv, color="#fb923c", linewidth=2.8)
    # Dots on current with glow effect
    ax.scatter(angles[:-1], current_vals,
               color="#fbbf24", s=55, zorder=6,
               linewidths=1.8, edgecolors="#f97316")

    # Ring percentage labels
    for r, lbl in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
        ax.text(np.pi / 2, r + 0.05, lbl, ha="center", va="bottom",
                fontsize=7, color="#445577", fontweight="500")

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, color="#94a3b8",
                       fontweight="600", linespacing=1.3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # Legend
    p_now  = mpatches.Patch(facecolor="#fb923c", alpha=0.85,
                             edgecolor="none", label="You Now")
    p_pot  = mpatches.Patch(facecolor="#60a5fa", alpha=0.65,
                             edgecolor="none", label="Your Potential")
    legend = ax.legend(
        handles=[p_now, p_pot],
        loc="upper right", bbox_to_anchor=(1.45, 1.22),
        fontsize=9, framealpha=0.25,
        facecolor="#0c1220", edgecolor=(1,1,1,0.10),
        labelcolor="#e2e8f0"
    )

    ax.set_title("You Now  vs  Your Potential",
                 fontsize=11, fontweight="bold",
                 color="#e2e8f0", pad=24, loc="center")

    fig.tight_layout(pad=2.0)
    return fig


# ============================================================
#  FEATURE IMPACT CHART  (shown before prediction)
# ============================================================
def impact_chart():
    """Horizontal bar chart — dark themed, showing feature weights."""
    labels  = ["Study Hours", "Sleep Quality", "Attendance",
               "Screen Time", "Stress", "Active Time", "Social Time"]
    impacts = [0.40, 0.20, 0.18, 0.15, 0.12, 0.08, 0.05]
    colors  = ["#10b981", "#8b5cf6", "#3b82f6", "#ef4444",
               "#f97316", "#06b6d4", "#64748b"]
    BG = "#0c1220"

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    bars = ax.barh(labels, impacts, color=colors, height=0.58,
                   edgecolor="none", alpha=0.88)

    for bar, val in zip(bars, impacts):
        ax.text(val + 0.007, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=9,
                color="#e2e8f0", fontweight="700")

    ax.set_xlabel("Relative Impact on Exam Score",
                  color="#445577", fontsize=8.5)
    ax.set_title("What Drives Your Score?", fontsize=12,
                 fontweight="bold", color="#f0f4ff", pad=14)
    ax.set_xlim(0, 0.58)
    ax.invert_yaxis()
    ax.tick_params(colors="#8899bb", labelsize=9.5, left=False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=(1,1,1,0.05), linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    fig.tight_layout(pad=1.8)
    return fig


# ============================================================
#  SYNCED SLIDER + NUMBER INPUT
# ============================================================
def slider_with_input(label, key, min_val, max_val, default, step,
                      hint=None, estimate_html=None):
    val_key    = f"val_{key}"
    slider_key = f"sl_{key}"
    num_key    = f"ni_{key}"

    if val_key not in st.session_state:
        st.session_state[val_key] = float(default)

    def on_slider():
        st.session_state[val_key] = st.session_state[slider_key]

    def on_num():
        st.session_state[val_key] = float(st.session_state[num_key])

    st.markdown(f'<p class="field-label">{label}</p>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<p class="field-hint">{hint}</p>', unsafe_allow_html=True)

    col_s, col_n = st.columns([3, 1])
    with col_s:
        st.slider(f"_s_{key}", min_value=float(min_val), max_value=float(max_val),
                  value=float(st.session_state[val_key]), step=float(step),
                  label_visibility="collapsed", key=slider_key, on_change=on_slider)
    with col_n:
        st.number_input(f"_n_{key}", min_value=float(min_val), max_value=float(max_val),
                        value=float(st.session_state[val_key]), step=float(step),
                        label_visibility="collapsed", key=num_key, on_change=on_num)

    if estimate_html:
        st.markdown(f'<div class="field-estimate">{estimate_html}</div>',
                    unsafe_allow_html=True)

    return float(st.session_state[val_key])


# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="ss-header">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
    <span style="font-size:2.2rem;filter:drop-shadow(0 0 14px rgba(59,130,246,0.7));">🎓</span>
    <div>
      <div class="ss-header-title">Score<span>Sense</span> AI</div>
      <div class="ss-header-sub">Intelligent Academic Scoring &amp; Personalised Recommendation Engine</div>
    </div>
  </div>
  <div>
    <span class="ss-badge">⚡ ML Powered</span>
    <span class="ss-badge ss-badge-gold">🏅 Gradient Boosting</span>
    <span class="ss-badge ss-badge-green">📊 5,000 Students</span>
    <span class="ss-badge">🔮 Real-Time Prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  MAIN LAYOUT
# ============================================================
left_col, right_col = st.columns([1, 1.65], gap="large")

# ── LEFT PANEL ────────────────────────────────────────────────
with left_col:
    st.markdown('### 📋 Your Daily Habits')
    st.caption('Drag the slider **or** type a value in the box on the right.')
    st.markdown('---')

    # Gender
    st.markdown('<p class="f-label">Gender</p>', unsafe_allow_html=True)
    gender = st.radio("_g", ["Male", "Female"], horizontal=True, label_visibility="collapsed")

    # Study
    study_hours = slider_with_input(
        "Study Hours / Day", "study", 0.0, 12.0, 4.5, 0.5,
        hint="Active studying only — not scrolling or passive reading."
    )

    # Sleep
    sleep_hours = slider_with_input(
        "Sleep Hours / Day", "sleep", 3.0, 12.0, 7.0, 0.5,
        hint="Average hours of sleep per night."
    )

    # Screen Time
    screen_time = slider_with_input(
        "Screen Time / Day (Mobile + TV)", "screen", 0.0, 14.0, 5.5, 0.5,
        hint="Combined phone + TV/streaming hours.",
        estimate_html=f"📱 Mobile ~{round(st.session_state.get('val_screen',5.5)*0.6,1)}h &nbsp;|&nbsp; 📺 TV ~{round(st.session_state.get('val_screen',5.5)*0.4,1)}h"
    )

    # Active Time
    active_time = slider_with_input(
        "Active Time / Day (Exercise + Extra)", "active", 0.0, 10.0, 2.7, 0.5,
        hint="Physical workout + clubs/hobbies combined.",
        estimate_html=f"🏃 Exercise ~{round(st.session_state.get('val_active',2.7)*0.6,1)}h &nbsp;|&nbsp; 🎨 Extra ~{round(st.session_state.get('val_active',2.7)*0.4,1)}h"
    )

    # Social Time
    social_time = slider_with_input(
        "Social Time / Day (Friends + Family)", "social", 0.0, 12.0, 4.5, 0.5,
        hint="Time with friends and family combined.",
        estimate_html=f"👫 Friends ~{round(st.session_state.get('val_social',4.5)*0.5,1)}h &nbsp;|&nbsp; 🏠 Family ~{round(st.session_state.get('val_social',4.5)*0.5,1)}h"
    )

    # Attendance
    attendance_pct = slider_with_input(
        "Attendance (%)", "attend", 40.0, 100.0, 78.0, 1.0,
        hint="Percentage of classes you attend."
    )

    # Stressed
    st.markdown('<p class="f-label">Experiencing Stress?</p>', unsafe_allow_html=True)
    st.markdown('<p class="f-hint">From studies or personal life</p>', unsafe_allow_html=True)
    stressed = st.selectbox("_st", ["No", "Yes"], label_visibility="collapsed")

    st.markdown("")

    predict_btn = st.button(
        "🔮  Predict My Score & Get Recommendations",
        use_container_width=True, type="primary"
    )


# ── RIGHT PANEL ───────────────────────────────────────────────
with right_col:

    if predict_btn:
        student = {
            "gender": gender, "study_hours": study_hours,
            "sleep_hours": sleep_hours, "screen_time": screen_time,
            "active_time": active_time, "social_time": social_time,
            "attendance_pct": attendance_pct, "stressed": stressed,
        }
        score, grade, expanded = predict(student)
        cfg   = GRADE_CONFIG[grade]
        rmse  = pkg.get("metadata", {}).get("rmse", 4.0)
        ci_lo = max(30, round(score - 1.96 * rmse, 1))
        ci_hi = min(100, round(score + 1.96 * rmse, 1))
        recs  = make_recommendations(student, expanded)
        potential_score = min(100, round(score + len(recs) * 4.5, 1))

        # ── SCORE CARD ────────────────────────────────────────
        st.markdown(
            f'<div class="score-card" style="background:{cfg["bg"]};">'
            f'  <div class="score-card-noise"></div>'
            f'  <div class="score-card-orb"></div>'
            f'  <div class="score-card-orb2"></div>'
            f'  <span class="score-card-emoji">{cfg["emoji"]}</span>'
            f'  <div class="score-card-num">{score}</div>'
            f'  <div class="score-card-grade">{grade}</div>'
            f'  <div class="score-card-msg">{cfg["msg"]}</div>'
            f'  <div><span class="score-card-ci">📊 95% CI: {ci_lo} – {ci_hi}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── METRIC TILES ──────────────────────────────────────
        avg_delta = score - 65
        pot_delta = potential_score - score
        avg_sign  = "pos" if avg_delta >= 0 else "neg"
        avg_word  = "above avg" if avg_delta >= 0 else "below avg"

        st.markdown(
            f'<div class="metric-grid">'
            f'  <div class="metric-tile mt-blue">'
            f'    <div class="metric-lbl">Your Score</div>'
            f'    <div class="metric-val">{score}</div>'
            f'    <div class="metric-delta-{avg_sign}">{avg_delta:+.1f} {avg_word}</div>'
            f'  </div>'
            f'  <div class="metric-tile mt-gray">'
            f'    <div class="metric-lbl">Class Average</div>'
            f'    <div class="metric-val">~65</div>'
            f'    <div class="metric-delta-neu">baseline</div>'
            f'  </div>'
            f'  <div class="metric-tile mt-green">'
            f'    <div class="metric-lbl">Your Potential</div>'
            f'    <div class="metric-val">~{potential_score}</div>'
            f'    <div class="metric-delta-pos">+{pot_delta:.1f} pts possible</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Progress bar
        pct = int(score)
        st.markdown(
            f'<div class="prog-track">'
            f'  <div class="prog-fill" style="width:{pct}%;background:{cfg["color"]};"></div>'
            f'</div>'
            f'<div class="prog-labels"><span>0</span><span>Score: {score}</span><span>100</span></div>',
            unsafe_allow_html=True
        )

        # ── RADAR + HABIT TABLE ───────────────────────────────
        st.markdown(
            '<div class="sec-head"><div class="sec-head-dot"></div>'
            '<div class="sec-head-text">Habit Profile</div>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )

        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            st.pyplot(radar_chart(student), use_container_width=True)

        with rc2:
            st.markdown(
                '<div class="sec-head" style="margin-top:8px"><div class="sec-head-dot" style="background:#10b981;box-shadow:0 0 8px rgba(16,185,129,0.4);"></div>'
                '<div class="sec-head-text">You vs Target</div>'
                '<div class="sec-head-line"></div></div>',
                unsafe_allow_html=True
            )

            rows = [
                ("Gender",       gender,                   "—"),
                ("Study/Day",    f"{study_hours}h",        f"{BENCHMARKS['study_hours']}h",
                 study_hours < BENCHMARKS["study_hours"] - 0.5),
                ("Sleep/Day",    f"{sleep_hours}h",        "7–9h",
                 sleep_hours < 7 or sleep_hours > 9.5),
                ("Screen Time",  f"{screen_time}h",        f"<{BENCHMARKS['screen_time']}h",
                 screen_time > BENCHMARKS["screen_time"] + 0.8),
                ("Active Time",  f"{active_time}h",        f"{BENCHMARKS['active_time']}h",
                 active_time < BENCHMARKS["active_time"] - 0.4),
                ("Social Time",  f"{social_time}h",        f"~{BENCHMARKS['social_time']}h", False),
                ("Attendance",   f"{attendance_pct:.0f}%", f"{BENCHMARKS['attendance_pct']:.0f}%",
                 attendance_pct < BENCHMARKS["attendance_pct"] - 5),
                ("Stress",       stressed,                 "No", stressed == "Yes"),
            ]

            html_rows = ""
            for row in rows:
                if len(row) == 3:
                    feat, val, tgt, warn_flag = row[0], row[1], row[2], False
                else:
                    feat, val, tgt, warn_flag = row
                warn = " ⚠️" if warn_flag else ""
                val_class = "h-val h-warn" if warn_flag else "h-val"
                html_rows += (
                    f'<div class="habit-row">'
                    f'<span class="h-feat">{feat}</span>'
                    f'<span class="{val_class}">{val}{warn}</span>'
                    f'<span class="h-tgt">/ {tgt}</span>'
                    f'</div>'
                )
            st.markdown(f'<div class="habit-table">{html_rows}</div>', unsafe_allow_html=True)

        # ── RECOMMENDATIONS ───────────────────────────────────
        st.markdown(
            '<div class="sec-head"><div class="sec-head-dot" style="background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,0.4);"></div>'
            '<div class="sec-head-text">💡 Personalised Recommendations</div>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )

        if not recs:
            st.success(
                "🌟 **Exceptional habits!** You're already performing at the top level. "
                "Maintain this consistency — consider peer tutoring or mentoring others."
            )
        else:
            for i, rec in enumerate(recs, 1):
                lvl = rec.get("level", "low")
                st.markdown(
                    f'<div class="rec-card rec-{lvl}">'
                    f'  <div class="rec-title">'
                    f'    {rec["icon"]} {i}. {rec["title"]}'
                    f'    <span class="rec-impact-badge">Impact {rec["impact"]:.0%}</span>'
                    f'  </div>'
                    f'  <div class="rec-body">{rec["detail"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── EMPTY STATE ───────────────────────────────────────────
    else:
        st.markdown("""
        <div class="empty-hero">
          <span class="empty-hero-icon">🎯</span>
          <div class="empty-hero-title">Ready to Predict?</div>
          <div class="empty-hero-sub">
            Enter your daily habits on the left and click the button
            to receive your AI-powered exam score prediction with
            personalised action plan.
          </div>
        </div>
        <div class="feature-grid">
          <div class="feature-card">
            <div class="feature-card-icon">🎯</div>
            <div class="feature-card-title">Predicted Score</div>
            <div class="feature-card-desc">ML-powered prediction with 95% confidence interval</div>
          </div>
          <div class="feature-card">
            <div class="feature-card-icon">📊</div>
            <div class="feature-card-title">Habit Radar</div>
            <div class="feature-card-desc">You Now vs Your Potential — see your growth gap</div>
          </div>
          <div class="feature-card">
            <div class="feature-card-icon">💡</div>
            <div class="feature-card-title">Action Plan</div>
            <div class="feature-card-desc">Ranked recommendations sorted by score impact</div>
          </div>
          <div class="feature-card">
            <div class="feature-card-icon">📈</div>
            <div class="feature-card-title">Growth Metrics</div>
            <div class="feature-card-desc">See exactly how many points you can gain</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="sec-head"><div class="sec-head-dot"></div>'
            '<div class="sec-head-text">What Drives Your Score?</div>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )
        st.pyplot(impact_chart(), use_container_width=True)'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_source.lstrip('\n'))
print('app.py saved successfully!')
print('Run with: streamlit run app.py')


# --- Cell 40 ---
# Final summary dashboard
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# 1. Model metrics bar chart
ax1 = fig.add_subplot(gs[0, 0])
metrics  = ['R2', 'MAE', 'RMSE']
values   = [r2_score(y_test, y_pred_best),
             mean_absolute_error(y_test, y_pred_best),
             np.sqrt(mean_squared_error(y_test, y_pred_best))]
colors_m = ['#2ecc71', '#3498db', '#e67e22']
bars = ax1.bar(metrics, values, color=colors_m, edgecolor='white')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontweight='bold')
ax1.set_title('Best Model Metrics')
ax1.set_ylim(0, max(values) * 1.2)

# 2. Feature importance (top 8)
ax2 = fig.add_subplot(gs[0, 1:])
top8 = importance.tail(8)
ax2.barh(top8.index, top8.values, color='steelblue')
ax2.set_title('Top 8 Feature Importances')
ax2.set_xlabel('Importance')

# 3. Actual vs Predicted
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_test, y_pred_best, alpha=0.2, s=8, color='steelblue')
ax3.plot([30, 100], [30, 100], 'r--')
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
ax3.set_title('Actual vs Predicted')

# 4. Grade distribution pie
ax4 = fig.add_subplot(gs[1, 1])
grade_order  = ['A (Excellent)', 'B (Good)', 'C (Average)', 'D (Below Avg)', 'F (Fail)']
grade_colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
grade_counts = df_processed['grade'].value_counts().reindex(grade_order)
ax4.pie(grade_counts, labels=grade_order, colors=grade_colors,
        autopct='%1.0f%%', startangle=140)
ax4.set_title('Grade Distribution')

# 5. All model R2 comparison
ax5 = fig.add_subplot(gs[1, 2])
short_names = (results_df['Model']
               .str.replace(' Regression', '', regex=False)
               .str.replace('Gradient Boosting', 'GB', regex=False)
               .str.replace('Random Forest', 'RF', regex=False)
               .str.replace('Extra Trees', 'ExTree', regex=False)
               .str.replace('K-Nearest Neighbors', 'KNN', regex=False)
               .str.replace('Decision Tree', 'DT', regex=False))
ax5.bar(range(len(results_df)), results_df['R2'],
        color=sns.color_palette('viridis', len(results_df)))
ax5.set_xticks(range(len(results_df)))
ax5.set_xticklabels(short_names, fontsize=7, rotation=30, ha='right')
ax5.set_title('All Models R2')
ax5.set_ylabel('R2')

plt.suptitle('ScoreSense AI -- Project Summary Dashboard',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('plot_summary_dashboard.png', bbox_inches='tight')
plt.show()

print()
print('=' * 65)
print('  SCORESENSE AI -- KEY FINDINGS')
print('=' * 65)
print(f'  Best Model : Gradient Boosting (tuned)')
print(f'  R2         : {r2_score(y_test, y_pred_best):.4f}')
print(f'  MAE        : {mean_absolute_error(y_test, y_pred_best):.3f} points')
print(f'  RMSE       : {np.sqrt(mean_squared_error(y_test, y_pred_best)):.3f} points')
print(f'  MAPE       : {mean_absolute_percentage_error(y_test, y_pred_best)*100:.2f}%')
print()
print('  Top 3 Feature Importances:')
top3 = importance.sort_values(ascending=False).head(3)
for feat, imp in top3.items():
    print(f'    {feat:<30} importance={imp:.4f}')
print()
print('  Files: scoresense_model.pkl | app.py | student_lifestyle_data.csv')
print('=' * 65)

