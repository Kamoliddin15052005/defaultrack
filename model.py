import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Grafik sozlamalari
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

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

df = pd.read_csv('data_new/full.csv')
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

# VIZUALIZATSIYA 1: Modellarni taqqoslash
print("\n📊 Vizualizatsiya yaratilmoqda...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('KREDIT TAHLIL - MODEL TAQQOSLASH', fontsize=16, fontweight='bold')

# 1. Model Accuracy
ax = axes[0, 0]
model_names = list(results_dict.keys())
accuracies = [results_dict[m]['accuracy'] * 100 for m in model_names]
colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in model_names]
bars = ax.bar(model_names, accuracies, color=colors, alpha=0.8)
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Model Aniqlik Taqqoslashi', fontweight='bold')
ax.set_ylim([min(accuracies) - 5, 100])
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. ROC-AUC Score
ax = axes[0, 1]
roc_scores = [results_dict[m]['roc_auc'] for m in model_names]
bars = ax.bar(model_names, roc_scores, color=colors, alpha=0.8)
ax.set_ylabel('ROC-AUC Score', fontweight='bold')
ax.set_title('ROC-AUC Taqqoslashi', fontweight='bold')
ax.set_ylim([min(roc_scores) - 0.05, 1.0])
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Training Time
ax = axes[0, 2]
train_times = [results_dict[m].get('training_time', 0) for m in model_names if m != 'Ensemble']
train_names = [m for m in model_names if m != 'Ensemble']
bars = ax.bar(train_names, train_times, color='#e74c3c', alpha=0.8)
ax.set_ylabel("Vaqt (soniya)", fontweight='bold')
ax.set_title("O'qitish Vaqti", fontweight='bold')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4. ROC Curve
ax = axes[1, 0]
for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, results_dict[name]['probabilities'])
    auc_score = results_dict[name]['roc_auc']
    lw = 3 if name == best_model_name else 2
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=lw)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.5000)')
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curve - Barcha Modellar', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# 5. Confusion Matrix (eng yaxshi model)
ax = axes[1, 1]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Bashorat', fontweight='bold')
ax.set_ylabel('Haqiqiy', fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
ax.set_xticklabels(['Berish (0)', 'Rad (1)'])
ax.set_yticklabels(['Berish (0)', 'Rad (1)'])

# 6. Feature Importance (eng yaxshi model)
ax = axes[1, 2]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    feature_names = [X.columns[i] for i in indices]
    ax.barh(range(len(indices)), importances[indices], color='#9b59b6', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel('Ahamiyat', fontweight='bold')
    ax.set_title(f'TOP 10 Muhim Xususiyatlar - {best_model_name}', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Feature importance\nmavjud emas', 
            ha='center', va='center', fontsize=12)
    ax.set_title('Feature Importance', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ model_comparison.png saqlandi")
plt.close()

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

# VIZUALIZATSIYA 2: Bashorat natijalari
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('KREDIT BASHORAT NATIJALARI', fontsize=16, fontweight='bold')

# 1. Default Taqsimoti
ax = axes[0, 0]
default_counts = results_df['default'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax.pie(default_counts, labels=['Tasdiqlangan (0)', 'Rad etilgan (1)'],
                                    autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                    explode=(0.05, 0.05), shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)
ax.set_title('Kredit Qarorlari Taqsimoti', fontweight='bold', fontsize=12)

# 2. Risk Guruhlari
ax = axes[0, 1]
risk_counts = results_df['risk_category'].value_counts().sort_index()
colors_risk = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
bars = ax.bar(risk_counts.index, risk_counts.values, color=colors_risk, alpha=0.8)
ax.set_ylabel('Mijozlar soni', fontweight='bold')
ax.set_title('Risk Guruhlari bo\'yicha Taqsimot', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(results_df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Probability Distribution
ax = axes[1, 0]
ax.hist(results_df['prob'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax.axvline(x=0.3, color='#2ecc71', linestyle='--', linewidth=2, label='Past risk chegara (0.3)')
ax.axvline(x=0.5, color='#f39c12', linestyle='--', linewidth=2, label='Qaror chegara (0.5)')
ax.axvline(x=0.7, color='#e74c3c', linestyle='--', linewidth=2, label='Yuqori risk chegara (0.7)')
ax.set_xlabel('Default Ehtimoli', fontweight='bold')
ax.set_ylabel('Mijozlar soni', fontweight='bold')
ax.set_title('Default Ehtimoli Taqsimoti', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Top Features (agar mavjud bo'lsa)
ax = axes[1, 1]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    top_indices = np.argsort(importances)[-15:]
    top_features = [X.columns[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_indices)))
    bars = ax.barh(range(len(top_indices)), top_importances, color=colors_feat, alpha=0.8)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xlabel('Ahamiyat darajasi', fontweight='bold')
    ax.set_title('TOP 15 Eng Muhim Xususiyatlar', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'Feature Importance\nMavjud Emas',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance', fontweight='bold')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
print("✓ prediction_results.png saqlandi")
plt.close()

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

# VIZUALIZATSIYA 3: Ma'lumotlar tahlili
print("\n📊 Ma'lumotlar tahlili vizualizatsiyasi...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("MA'LUMOTLAR TAHLILI", fontsize=16, fontweight='bold')

# 1. Credit Score vs Default
if 'credit_score' in results_extended.columns:
    ax = axes[0, 0]
    for default_val in [0, 1]:
        data = results_extended[results_extended['default'] == default_val]['credit_score'].dropna()
        ax.hist(data, bins=30, alpha=0.6, 
               label=f"Default={default_val} ({len(data)})",
               color='#2ecc71' if default_val == 0 else '#e74c3c')
    ax.set_xlabel('Credit Score', fontweight='bold')
    ax.set_ylabel('Soni', fontweight='bold')
    ax.set_title('Credit Score Taqsimoti', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes[0, 0].text(0.5, 0.5, 'Credit Score\nMavjud emas', 
                    ha='center', va='center', fontsize=12)

# 2. Income vs Loan Amount
if 'annual_income' in results_extended.columns and 'loan_amount' in results_extended.columns:
    ax = axes[0, 1]
    for default_val in [0, 1]:
        mask = results_extended['default'] == default_val
        ax.scatter(
            results_extended[mask]['annual_income'],
            results_extended[mask]['loan_amount'],
            alpha=0.6, s=50,
            label=f"Default={default_val}",
            color='#2ecc71' if default_val == 0 else '#e74c3c'
        )
    ax.set_xlabel('Yillik Daromad ($)', fontweight='bold')
    ax.set_ylabel('Kredit Miqdori ($)', fontweight='bold')
    ax.set_title('Daromad vs Kredit Miqdori', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'Income/Loan\nMavjud emas',
                    ha='center', va='center', fontsize=12)


# 4. Debt to Income Ratio
if 'debt_to_income_ratio' in results_extended.columns:
    ax = axes[1, 0]
    for default_val in [0, 1]:
        data = results_extended[results_extended['default'] == default_val]['debt_to_income_ratio'].dropna()
        ax.hist(data, bins=30, alpha=0.6,
               label=f"Default={default_val}",
               color='#2ecc71' if default_val == 0 else '#e74c3c')
    ax.set_xlabel('Debt to Income Ratio', fontweight='bold')
    ax.set_ylabel('Soni', fontweight='bold')
    ax.set_title('Qarz-Daromad Nisbati', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'Debt Ratio\nMavjud emas', 
                    ha='center', va='center', fontsize=12)

# 5. Probability by Risk Category
ax = axes[1, 1]
risk_data = []
risk_labels = []
for cat in ['Past', "O'rtacha", 'Yuqori', 'Juda yuqori']:
    data = results_extended[results_extended['risk_category'] == cat]['prob'].values
    if len(data) > 0:
        risk_data.append(data)
        risk_labels.append(cat)

if risk_data:
    bp = ax.boxplot(risk_data, labels=risk_labels, patch_artist=True)
    colors_box = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box[:len(risk_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Default Ehtimoli', fontweight='bold')
    ax.set_title('Risk Guruhlari bo\'yicha Ehtimollik', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
else:
    ax.text(0.5, 0.5, 'Risk data\nMavjud emas', ha='center', va='center', fontsize=12)

# 6. Top 10 Mijozlar (eng yuqori risk)
ax = axes[1, 2]
top_risk = results_extended.nlargest(10, 'prob')[['customer_id', 'prob']].reset_index(drop=True)
if len(top_risk) > 0:
    colors_top = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_risk)))
    bars = ax.barh(range(len(top_risk)), top_risk['prob'], color=colors_top, alpha=0.8)
    ax.set_yticks(range(len(top_risk)))
    ax.set_yticklabels([f"ID: {int(id)}" for id in top_risk['customer_id']], fontsize=9)
    ax.set_xlabel('Default Ehtimoli', fontweight='bold')
    ax.set_title('TOP 10 Yuqori Risk Mijozlar', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=8)
else:
    ax.text(0.5, 0.5, 'Top risk\nMavjud emas', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
print("✓ data_analysis.png saqlandi")
plt.close()

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