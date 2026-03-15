import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

print("=" * 55)
print("   PraanAI — Model Training (Improved Version)")
print("=" * 55)

# ── Step 1: Load Dataset ──────────────────────
print("\n🔵 Step 1: Loading dataset...")

df = pd.read_csv(
    'dataset/triage.csv',
    sep=';',
    encoding='latin1'
)
print(f"   Raw shape: {df.shape}")

# ── Step 2: Clean Data ────────────────────────
print("\n🔵 Step 2: Cleaning data...")

# Fix European decimal format (5,00 → 5.00)
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass

# Keep only needed columns
cols = ['Age', 'Arrival mode', 'Chief_complain', 'Pain',
        'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT',
        'Saturation', 'KTAS_expert']

df = df[cols].dropna()

# Force all numeric columns to float
# (some may still be strings after comma replacement)
numeric_cols = ['Age', 'Arrival mode', 'Pain', 'NRS_pain',
                'SBP', 'DBP', 'HR', 'RR', 'BT',
                'Saturation', 'KTAS_expert']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
print(f"   Clean shape: {df.shape}")

# ── Step 3: Feature Engineering ──────────────
print("\n🔵 Step 3: Engineering new features...")

# Pulse Pressure — difference between SBP and DBP
# High pulse pressure can indicate cardiovascular issues
df['pulse_pressure'] = df['SBP'] - df['DBP']

# Shock Index — HR divided by SBP
# Value > 1.0 indicates potential shock/critical condition
df['shock_index'] = df['HR'] / df['SBP'].replace(0, 1)

# Fever flag — body temperature above 38°C
df['fever'] = (df['BT'] > 38).astype(int)

# Low oxygen flag — SpO2 below 94%
df['low_spo2'] = (df['Saturation'] < 94).astype(int)

# High pain flag — NRS pain score above 7
df['high_pain'] = (df['NRS_pain'] > 7).astype(int)

# Hypertension flag — SBP above 140
df['hypertension'] = (df['SBP'] > 140).astype(int)

# Hypotension flag — SBP below 90 (danger zone)
df['hypotension'] = (df['SBP'] < 90).astype(int)

# Tachycardia flag — HR above 100
df['tachycardia'] = (df['HR'] > 100).astype(int)

print(f"   Total features: {len(df.columns) - 1} (was 11, now {len(df.columns) - 1})")

# ── Step 4: Encode Labels ─────────────────────
print("\n🔵 Step 4: Encoding labels...")

le = LabelEncoder()
df['Chief_complain'] = le.fit_transform(df['Chief_complain'].astype(str))

# Save label encoder
import os
os.makedirs('model', exist_ok=True)
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("   ✅ Label encoder saved")

# ── Step 5: Prepare Features ──────────────────
print("\n🔵 Step 5: Preparing features and target...")

feature_cols = [
    'Age', 'Arrival mode', 'Chief_complain', 'Pain', 'NRS_pain',
    'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation',
    'pulse_pressure', 'shock_index', 'fever', 'low_spo2',
    'high_pain', 'hypertension', 'hypotension', 'tachycardia'
]

X = df[feature_cols]
y = df['KTAS_expert'] - 2  # Convert ESI 2-5 → 0-3

print(f"   Features: {X.shape[1]}")
print(f"   Samples:  {X.shape[0]}")
print(f"   Class distribution:")
for level, count in sorted(y.value_counts().items()):
    print(f"     ESI {level+2}: {count} patients ({count/len(y)*100:.1f}%)")

# ── Step 6: Train/Test Split ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

# ── Step 7: Baseline XGBoost ──────────────────
print("\n🔵 Step 6: Baseline XGBoost (before tuning)...")

baseline = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"   Baseline Accuracy: {baseline_acc:.2%}")

# ── Step 8: Hyperparameter Tuning ────────────
print("\n🔵 Step 7: Hyperparameter tuning (this may take a few minutes)...")

param_grid = {
    'n_estimators':  [100, 200, 300],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample':     [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

grid_search = GridSearchCV(
    estimator  = xgb,
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    verbose    = 1,
    n_jobs     = -1   # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"\n   ✅ Best Parameters: {grid_search.best_params_}")
print(f"   ✅ Best CV Accuracy: {grid_search.best_score_:.2%}")

best_xgb = grid_search.best_estimator_
xgb_acc  = accuracy_score(y_test, best_xgb.predict(X_test))
print(f"   ✅ Tuned XGBoost Test Accuracy: {xgb_acc:.2%}")

# ── Step 9: Random Forest Comparison ─────────
print("\n🔵 Step 8: Comparing with Random Forest...")

rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 10,
    random_state = 42,
    n_jobs       = -1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"   Random Forest Accuracy: {rf_acc:.2%}")

# ── Step 10: Pick Best Model ──────────────────
print("\n🔵 Step 9: Selecting best model...")

if xgb_acc >= rf_acc:
    best_model      = best_xgb
    best_model_name = "XGBoost (Tuned)"
    best_acc        = xgb_acc
else:
    best_model      = rf
    best_model_name = "Random Forest"
    best_acc        = rf_acc

print(f"   🏆 Winner: {best_model_name} with {best_acc:.2%} accuracy")

# ── Step 11: Cross Validation ─────────────────
print("\n🔵 Step 10: Cross validation (5-fold)...")

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"   CV Scores: {[f'{s:.2%}' for s in cv_scores]}")
print(f"   Mean CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# ── Step 12: Classification Report ───────────
print("\n🔵 Step 11: Classification Report...")

y_pred = best_model.predict(X_test)
print(classification_report(
    y_test, y_pred,
    target_names=['ESI 2 Emergent', 'ESI 3 Urgent',
                  'ESI 4 Less Urgent', 'ESI 5 Non Urgent']
))

# ── Step 13: Feature Importance ───────────────
print("\n🔵 Step 12: Top 10 Most Important Features...")

if best_model_name.startswith("XGBoost"):
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

feat_imp = sorted(
    zip(feature_cols, importances),
    key=lambda x: x[1], reverse=True
)
for feat, imp in feat_imp[:10]:
    bar = '█' * int(imp * 100)
    print(f"   {feat:<20} {bar} {imp:.4f}")

# ── Step 14: Save Best Model ──────────────────
print("\n🔵 Step 13: Saving best model...")

with open('model/triage_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Also save feature columns so app.py knows the order
with open('model/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("   ✅ model/triage_model.pkl saved")
print("   ✅ model/feature_cols.pkl saved")
print("   ✅ model/label_encoder.pkl saved")

# ── Summary ───────────────────────────────────
print("\n" + "=" * 55)
print("   TRAINING SUMMARY")
print("=" * 55)
print(f"   Baseline XGBoost Accuracy : {baseline_acc:.2%}")
print(f"   Tuned XGBoost Accuracy    : {xgb_acc:.2%}")
print(f"   Random Forest Accuracy    : {rf_acc:.2%}")
print(f"   Best Model                : {best_model_name}")
print(f"   Final Accuracy            : {best_acc:.2%}")
print(f"   CV Mean Accuracy          : {cv_scores.mean():.2%}")
print(f"   Features Used             : {len(feature_cols)}")
print("=" * 55)
print("\n✅ Training complete! Run python app.py to start the server.\n")