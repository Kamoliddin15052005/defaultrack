import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Gradient Boosting kutubxonalarini import qilish
try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

try:
    import catboost as cb
    catboost_available = True
except ImportError:
    catboost_available = False

# 1. MA'LUMOTLARNI YUKLASH
print("=" * 70)
print("KREDIT TAHLIL VA BASHORAT TIZIMI (CPU)".center(70))
print("=" * 70)

df = pd.read_csv('data/full.csv')
print(f"\n✓ Dataset yuklandi: {df.shape[0]} qator, {df.shape[1]} ustun")

# 2. BOSHLANG'ICH TAHLIL
print("\n" + "=" * 70)
print("1. MA'LUMOTLAR TAHLILI")
print("=" * 70)
print(f"\nDefault taqsimoti:\n{df['default'].value_counts()}")
print(f"Default foizi: {df['default'].mean() * 100:.2f}%")

# 3. KERAKSIZ USTUNLARNI O'CHIRISH
drop_columns = ['customer_id', 'referral_code']
df = df.drop(columns=drop_columns, errors='ignore')

# 4. TARGET VA FEATURES
y = df['default'].copy()
X = df.drop('default', axis=1)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Bo'sh qiymatlarni to'ldirish
for col in numerical_cols:
    X[col].fillna(X[col].median(), inplace=True)
for col in categorical_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)

# 5. ENCODE
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 6. FEATURE ENGINEERING
if 'monthly_income_$' in X.columns and 'existing_monthly_debt_$' in X.columns:
    X['debt_burden'] = X['existing_monthly_debt_$'] / (X['monthly_income_$'] + 1)

if 'credit_score' in X.columns:
    X['credit_risk_category'] = pd.cut(X['credit_score'], 
                                        bins=[0, 580, 670, 740, 850], 
                                        labels=[0, 1, 2, 3]).astype(float)

if 'age' in X.columns:
    X['age_group'] = pd.cut(X['age'], 
                             bins=[0, 25, 35, 50, 100], 
                             labels=[0, 1, 2, 3]).astype(float)

# 7. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. STANDARDIZATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. MODEL O'QITISH
print("\n" + "=" * 70)
print("2. MODELLARNI O'QITISH")
print("=" * 70)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

import time
models_dict = {}
training_times = {}

# XGBoost
print(f"\n🚀 XGBoost o'qitilmoqda...")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight, random_state=42,
    n_jobs=-1, eval_metric='auc', early_stopping_rounds=30,
    verbosity=0
)
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
xgb_model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
training_times['XGBoost'] = time.time() - start
models_dict['XGBoost'] = xgb_model
print(f"   ✓ Tayyor! Vaqt: {training_times['XGBoost']:.2f}s")

# LightGBM
if lightgbm_available:
    print(f"\n🚀 LightGBM o'qitilmoqda...")
    start = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    training_times['LightGBM'] = time.time() - start
    models_dict['LightGBM'] = lgb_model
    print(f"   ✓ Tayyor! Vaqt: {training_times['LightGBM']:.2f}s")

# CatBoost
if catboost_available:
    print(f"\n🚀 CatBoost o'qitilmoqda...")
    start = time.time()
    cat_model = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        subsample=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42, thread_count=-1, verbose=False
    )
    cat_model.fit(X_train_scaled, y_train)
    training_times['CatBoost'] = time.time() - start
    models_dict['CatBoost'] = cat_model
    print(f"   ✓ Tayyor! Vaqt: {training_times['CatBoost']:.2f}s")

# XGBoost #2
print(f"\n🚀 XGBoost #2 o'qitilmoqda...")
start = time.time()
xgb_model2 = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9, gamma=0.05,
    min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
    scale_pos_weight=scale_pos_weight, random_state=123,
    n_jobs=-1, eval_metric='auc', verbosity=0
)
xgb_model2.fit(X_train_scaled, y_train, verbose=False)
training_times['XGBoost #2'] = time.time() - start
models_dict['XGBoost #2'] = xgb_model2
print(f"   ✓ Tayyor! Vaqt: {training_times['XGBoost #2']:.2f}s")

print(f"\n⏱️ Umumiy vaqt: {sum(training_times.values()):.2f}s")

# 10. BAHOLASH
print("\n" + "=" * 70)
print("3. MODELLARNI BAHOLASH")
print("=" * 70)

results_dict = {}
for name, model in models_dict.items():
    pred = model.predict(X_test_scaled)
    prob = model.predict_proba(X_test_scaled)[:, 1]
    results_dict[name] = {
        'model': model, 'predictions': pred, 'probabilities': prob,
        'accuracy': accuracy_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, prob),
        'training_time': training_times.get(name, 0)
    }
    print(f"\n🎯 {name}:")
    print(f"   Accuracy: {results_dict[name]['accuracy'] * 100:.2f}%")
    print(f"   ROC-AUC: {results_dict[name]['roc_auc']:.4f}")

# Ensemble
all_probs = np.array([results_dict[name]['probabilities'] for name in results_dict.keys()])
ensemble_prob = all_probs.mean(axis=0)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_roc = roc_auc_score(y_test, ensemble_prob)

print(f"\n🎯 ENSEMBLE:")
print(f"   Accuracy: {ensemble_acc * 100:.2f}%")
print(f"   ROC-AUC: {ensemble_roc:.4f}")

results_dict['Ensemble'] = {
    'predictions': ensemble_pred, 'probabilities': ensemble_prob,
    'accuracy': ensemble_acc, 'roc_auc': ensemble_roc
}

# Eng yaxshi model
best_model_name = max(results_dict.keys(), key=lambda x: results_dict[x]['roc_auc'])
best_result = results_dict[best_model_name]
print(f"\n🏆 ENG YAXSHI: {best_model_name} (ROC-AUC: {best_result['roc_auc']:.4f})")

best_pred = best_result['predictions']
best_prob = best_result['probabilities']
best_model = results_dict.get(best_model_name, {}).get('model', models_dict.get('XGBoost'))

# 11. BUTUN DATASET UCHUN BASHORAT
print("\n" + "=" * 70)
print("4. BUTUN DATASET UCHUN BASHORAT")
print("=" * 70)

df_full = pd.read_csv('data/full.csv')
customer_ids = df_full['customer_id'].copy()

df_processed = df_full.drop(columns=drop_columns + ['default'], errors='ignore')

# Encode
for col in categorical_cols:
    if col in df_processed.columns:
        le = label_encoders[col]
        df_processed[col] = df_processed[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Feature engineering
if 'monthly_income_$' in df_processed.columns and 'existing_monthly_debt_$' in df_processed.columns:
    df_processed['debt_burden'] = df_processed['existing_monthly_debt_$'] / (df_processed['monthly_income_$'] + 1)

if 'credit_score' in df_processed.columns:
    df_processed['credit_risk_category'] = pd.cut(df_processed['credit_score'], 
                                                    bins=[0, 580, 670, 740, 850], 
                                                    labels=[0, 1, 2, 3]).astype(float)

if 'age' in df_processed.columns:
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                        bins=[0, 25, 35, 50, 100], 
                                        labels=[0, 1, 2, 3]).astype(float)

for col in df_processed.columns:
    if df_processed[col].dtype in ['int64', 'float64']:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    else:
        df_processed[col].fillna(0, inplace=True)

df_processed_scaled = scaler.transform(df_processed)

# Bashorat
if best_model_name == 'Ensemble':
    all_predictions = []
    for name, result in results_dict.items():
        if name != 'Ensemble' and 'model' in result:
            model = result['model']
            prob = model.predict_proba(df_processed_scaled)[:, 1]
            all_predictions.append(prob)
    probabilities = np.array(all_predictions).mean(axis=0)
    predictions = (probabilities >= 0.5).astype(int)
else:
    predictions = best_model.predict(df_processed_scaled)
    probabilities = best_model.predict_proba(df_processed_scaled)[:, 1]

# 12. RESULTS.CSV SAQLASH
results_df = pd.DataFrame({
    'customer_id': customer_ids,
    'prob': probabilities.round(5),
    'default': predictions
})

results_df.to_csv('results.csv', index=False)
print(f"\n✓ results.csv saqlandi: {len(results_df)} qator")
print(f"  Tasdiqlangan: {(predictions == 0).sum()}")
print(f"  Rad etilgan: {(predictions == 1).sum()}")

# 13. RISK GURUHLARI
results_df['risk_category'] = pd.cut(
    results_df['prob'],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=['Past', "O'rtacha", 'Yuqori', 'Juda yuqori']
)

print(f"\n📊 Risk guruhlari:")
for cat, count in results_df['risk_category'].value_counts().sort_index().items():
    print(f"  {cat:12s}: {count:4d} ({count/len(results_df)*100:5.1f}%)")

# 14. KENGAYTIRILGAN NATIJALAR
print("\n" + "=" * 70)
print("5. KENGAYTIRILGAN NATIJALAR")
print("=" * 70)

df_full_copy = df_full.copy()
available_cols = ['customer_id']
optional_cols = ['credit_score', 'annual_income_$', 'loan_amount', 
                 'debt_to_income_ratio', 'age', 'loan_type']

for col in optional_cols:
    if col in df_full_copy.columns:
        available_cols.append(col)

results_extended = results_df.merge(df_full_copy[available_cols], on='customer_id', how='left')

def get_recommendation(prob):
    if prob < 0.3:
        return "✅ Berish mumkin"
    elif prob < 0.5:
        return "⚠️ Tekshiruv kerak"
    elif prob < 0.7:
        return "❌ Tavsiya etilmaydi"
    else:
        return "🚫 Yuqori risk"

results_extended['recommendation'] = results_extended['prob'].apply(get_recommendation)
results_extended.to_csv('results_detailed.csv', index=False)
print(f"✓ results_detailed.csv saqlandi")

# 15. MODEL SAQLASH
import pickle
model_data = {
    'model': best_model,
    'all_models': models_dict if best_model_name == 'Ensemble' else None,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist(),
    'model_name': best_model_name,
    'accuracy': best_result['accuracy'],
    'roc_auc': best_result['roc_auc'],
    'is_ensemble': best_model_name == 'Ensemble'
}

with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print(f"✓ credit_model.pkl saqlandi")

# 16. YAKUNIY HISOBOT
print("\n" + "=" * 70)
print("YAKUNIY HISOBOT".center(70))
print("=" * 70)

print(f"""
✅ MODEL: {best_model_name}
   Accuracy: {best_result['accuracy'] * 100:.2f}%
   ROC-AUC: {best_result['roc_auc']:.4f}

✅ NATIJALAR:
   Jami: {len(results_df)}
   Tasdiqlangan: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions)*100:.1f}%)
   Rad etilgan: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(predictions)*100:.1f}%)

✅ FAYLLAR:
   • results.csv
   • results_detailed.csv
   • credit_model.pkl
""")

print("=" * 70)
print("TUGADI!".center(70))
print("=" * 70)