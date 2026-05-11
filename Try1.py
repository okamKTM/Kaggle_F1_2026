import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

# ==========================================
# 1. データの読み込み
# ==========================================
# ファイルの場所を指定してください（デスクトップのフォルダを想定）
path = "/Users/okamura/Desktop/DataScience/Kaggle/F1/Kaggle_F1_2026/"

try:
    results = pd.read_csv(os.path.join(path, 'results.csv'))
    races = pd.read_csv(os.path.join(path, 'races.csv'))
    drivers = pd.read_csv(os.path.join(path, 'drivers.csv'))
    pit_stops = pd.read_csv(os.path.join(path, 'pit_stops.csv'))
    # 必要に応じて追加のファイルを読み込み
    # constructor_results = pd.read_csv(os.path.join(path, 'constructor_results.csv'))
    print("All CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# ==========================================
# 2. データの結合 (Merging)
# ==========================================
# ノートブックのメインロジックに従い、テーブルを結合します
df = pd.merge(results, races, on='raceId', how='left')
df = pd.merge(df, drivers, on='driverId', how='left')
df = pd.merge(df, pit_stops, on=['raceId', 'driverId'], how='left')

# 重複するカラム名や不要なカラムの整理
df.drop(['url_x', 'url_y', 'time_y', 'number_y'], axis=1, inplace=True, errors='ignore')

# ==========================================
# 3. 前処理と特徴量エンジニアリング
# ==========================================
# ターゲット変数の作成 (例: 順位が10位以内かどうか)
# ※ノートブックの目的変数設定に合わせて調整してください
df['is_top_10'] = df['rank'].apply(lambda x: 1 if str(x).isdigit() and int(x) <= 10 else 0)

# 欠損値処理
df['milliseconds_y'] = df['milliseconds_y'].fillna(df['milliseconds_y'].median())
df['stop'] = df['stop'].fillna(0)

# カテゴリ変数のエンコーディング
le = LabelEncoder()
df['driverRef'] = le.fit_transform(df['driverRef'].astype(str))

# 使用する特徴量の選定
features = [
    'grid', 'points', 'laps', 'milliseconds_x', 'stop', 
    'lap', 'milliseconds_y', 'driverRef', 'year'
]
X = df[features]
y = df['is_top_10']

# 数値データのクリーニング（文字列が含まれる場合の変換）
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# ==========================================
# 4. データの分割とスケーリング
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. モデルの構築と学習 (Multiple Models)
# ==========================================

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- XGBoost ---
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# ==========================================
# 6. 評価と可視化 (ROC/AUC)
# ==========================================
models = [('Random Forest', rf_model), ('XGBoost', xgb_model)]

plt.figure(figsize=(10, 7))

for name, model in models:
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{name} AUC Score: {auc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

# ROC曲線の装飾
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 特徴量の重要度表示 (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', title='Feature Importances (RF)')
plt.show()

print("\n--- Final Classification Report (XGBoost) ---")
print(classification_report(y_test, xgb_model.predict(X_test_scaled)))