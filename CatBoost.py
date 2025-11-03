# ==================== IMPORTS ====================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool


# ==================== PATHS ====================
train_path = "/content/drive/MyDrive/playground-series-s5e11/train.csv"
test_path  = "/content/drive/MyDrive/playground-series-s5e11/test.csv"
sub_path   = "/content/drive/MyDrive/playground-series-s5e11/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
sub_df   = pd.read_csv(sub_path)

print("Raw shapes:", train_df.shape, test_df.shape)

# ==================== DROP ID ====================
# Keep test ids for submission, but drop id column from features everywhere
if 'id' in train_df.columns:
    train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns:
    test_ids = test_df['id'].values.copy()
    test_df = test_df.drop(columns=['id'])
else:
    test_ids = None

# ==================== SHOW CATEGORICAL DOMAINS (for info) ====================
categorical_cols = ['gender', 'marital_status', 'education_level',
                   'employment_status', 'loan_purpose', 'grade_subgrade']
print("\nCategorical unique values (sanity):")
for c in categorical_cols:
    if c in train_df.columns:
        print(c, train_df[c].unique())
    else:
        raise ValueError(f"Missing expected column: {c}")

# ==================== FEATURE ENGINEERING ====================
# Basic engineered features recommended earlier
def add_features(df):
    df = df.copy()
    # ensure numeric types for source columns
    # derived debt estimate
    df['estimated_debt'] = df['annual_income'] * df['debt_to_income_ratio']

    # loan_to_income_ratio (new) — keep original name conflict-check
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1e-9)

    # loan risk indicator
    df['loan_risk'] = df['loan_amount'] * df['debt_to_income_ratio']

    # credit to loan ratio
    df['credit_to_loan_ratio'] = df['credit_score'] / (df['loan_amount'] + 1e-9)

    # risk score: interest adjusted by credit score (credit_score could be zero -> guard)
    df['risk_score'] = df['interest_rate'] / (df['credit_score'] + 1e-9)

    # interaction features
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income']
    df['income_sq'] = df['annual_income'] ** 2
    df['normalized_interest'] = df['interest_rate'] / (df['annual_income'] + 1e-9)

    # monthly measures and burden
    df['monthly_income'] = df['annual_income'] / 12.0
    # assume interest_rate is annual percent (e.g. 10 -> 10%)
    df['annual_interest_fraction'] = df['interest_rate'] / 100.0
    # approximate monthly payment (very rough — assumes interest-only / simplified)
    df['monthly_payment'] = (df['loan_amount'] * df['annual_interest_fraction']) / 12.0
    df['payment_to_income_ratio'] = df['monthly_payment'] / (df['monthly_income'] + 1e-9)

    # debt relative to loan
    df['debt_to_loan_ratio'] = df['estimated_debt'] / (df['loan_amount'] + 1e-9)

    # effective risk combining debt ratio and interest
    df['effective_risk'] = df['debt_to_income_ratio'] * df['interest_rate']

    # credit / income
    df['credit_income_ratio'] = df['credit_score'] / (df['annual_income'] + 1e-9)
    df['credit_interest_interaction'] = df['credit_score'] * df['interest_rate']

    # flags / bins
    df['high_interest_flag'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    # quantile bins for income & debt ratio
    df['income_band'] = pd.qcut(df['annual_income'].rank(method='first'), 5, labels=False).astype(int)
    df['debt_ratio_band'] = pd.qcut(df['debt_to_income_ratio'].rank(method='first'), 5, labels=False).astype(int)

    return df

train_df = add_features(train_df)
test_df  = add_features(test_df)

# ==================== DEFINE FINAL FEATURE LISTS ====================
numerical_cols = [
 'annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate',
 'estimated_debt','loan_to_income_ratio','loan_risk','credit_to_loan_ratio','risk_score',
 'credit_income_interaction','income_sq','normalized_interest','monthly_income','monthly_payment',
 'payment_to_income_ratio','debt_to_loan_ratio','effective_risk','credit_income_ratio',
 'credit_interest_interaction','annual_interest_fraction'
]

# some engineered categorical-ish bands
categorical_cols += ['income_band','debt_ratio_band','high_interest_flag']

# ensure columns exist
for c in numerical_cols + categorical_cols:
    if c not in train_df.columns:
        raise ValueError(f"Missing expected column: {c}")

# ==================== CLEANING MISSING VALUES ====================
# Fill categorical with mode (train), numerical with median (train)
for col in categorical_cols:
    mode = train_df[col].mode().iloc[0]
    train_df[col] = train_df[col].fillna(mode).astype(str)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(mode).astype(str)

for col in numerical_cols:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med).astype(float)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(med).astype(float)

# ==================== SIMPLE TRANSFORMS ====================
# For heavy-tailed variables log1p (as before)
train_df['annual_income'] = np.log1p(train_df['annual_income'])
test_df['annual_income'] = np.log1p(test_df['annual_income'])

train_df['loan_amount'] = np.log1p(train_df['loan_amount'])
test_df['loan_amount'] = np.log1p(test_df['loan_amount'])

# Recompute any derived features that depend on those transforms if necessary
# (we keep earlier engineered features as-is; many are ratios so okay)

# ==================== LABEL ENCODING CATEGORICALS (safe) ====================
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train_df[col].astype(str).values)
    label_encoders[col] = le
    # transform both train/test
    train_df[col] = le.transform(train_df[col].astype(str).values)
    # for test, unseen labels -> assign new index = n_classes
    test_vals = test_df[col].astype(str).values
    classes = set(le.classes_)
    n_classes = len(le.classes_)
    test_mapped = []
    for v in test_vals:
        if v in classes:
            test_mapped.append(int(np.where(le.classes_ == v)[0][0]))
        else:
            test_mapped.append(n_classes)
    test_df[col] = np.array(test_mapped, dtype=int)

# For safety, convert all categorical_cols to int dtype
for col in categorical_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

# ==================== TRAIN/VAL SPLIT (STRATIFIED) ====================
target_col = 'loan_paid_back'
X = train_df[numerical_cols + categorical_cols]
y = train_df[target_col].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train/Val sizes:", X_train.shape, X_val.shape, y_train.value_counts().to_dict())

# ==================== LIGHTGBM TRAINING ====================
# compute scale_pos_weight = neg/pos
pos = (y_train==1).sum()
neg = (y_train==0).sum()
scale_pos_weight = neg / (pos + 1e-9)
print("scale_pos_weight:", scale_pos_weight)

# LightGBM dataset
lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False, categorical_feature=categorical_cols)
lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False, categorical_feature=categorical_cols)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'scale_pos_weight': scale_pos_weight,  # helps with class imbalance
    'n_jobs': -1
}

evals_result = {}
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train','valid'],
)

# ==================== VALIDATION METRICS ====================
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.5f}")

# feature importance
fi = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("\nTop features:\n", fi.head(20))

# ==================== PLOT AUC HISTORY (optional) ====================
# LightGBM stored evals_result
if 'valid' in evals_result and 'auc' in evals_result['valid']:
    plt.plot(evals_result['train']['auc'], label='train_auc')
    plt.plot(evals_result['valid']['auc'], label='val_auc')
    plt.xlabel('iter')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('LightGBM AUC')
    plt.show()

# ==================== CatBoost ====================

cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_cols]

cbc = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=100,
    verbose=100
)

cbc.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features_idx)
val_pred_cb = cbc.predict_proba(X_val)[:,1]
print("CatBoost val AUC:", roc_auc_score(y_val, val_pred_cb))




