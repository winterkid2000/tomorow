import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
df = pd.read_excel(r"c:\Users\MIM\Desktop\새 폴더 (14)\토-온\조영 인 힘\토탈-온코-조영증강X1_1.xlsx")
df_enhanced = df[df["Enhanced"] == 0].copy()
df_seg = pd.read_excel("seg_metrics.xlsx", index_col=0)
df_delta = pd.read_excel("delta_radiomics.xlsx", index_col=0)

# 2. 공통 환자 ID
common_ids = df_seg.index.intersection(df_delta.index)
X = pd.concat([df_delta.loc[common_ids], df_seg.loc[common_ids]], axis=1)

# 3. PCA 기반 proxy target 생성
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
Y_proxy = np.abs(pca.fit_transform(X_scaled).flatten())

# 4. XGBoost 회귀로 weight 추정
model = XGBRegressor(n_estimators=100, random_state=0)
model.fit(X, Y_proxy)
raw_pred = model.predict(X)
pred_weights = 1 / (raw_pred + 1e-6)

# 5. weight 데이터프레임 구성
df_pred_weights = pd.DataFrame(pred_weights, index=common_ids, columns=["Pred_Weight"])
df_pred_weights = df_pred_weights.reset_index().rename(columns={"index": "Patient_N"})
df_pred_weights.to_excel("weigh_check.xlsx", index=False)

# 6. Patient ID 타입 통일 (정수형)
df_enhanced["Patient_N"] = df_enhanced["Patient_N"].astype(int)
df_pred_weights["Patient_N"] = df_pred_weights["Patient_N"].astype(int)

# 7. train/test split & weight 매핑 확인
seed = 42
unique_patients = df_enhanced["Patient_N"].unique()
valid_ids = df_enhanced["Patient_N"].unique()
df_pred_weights = df_pred_weights[df_pred_weights["Patient_N"].isin(valid_ids)]
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=seed)

df_train = df_enhanced[df_enhanced["Patient_N"].isin(train_patients)]
df_train = df_train.merge(df_pred_weights, on="Patient_N", how="left")
df_train["Pred_Weight"] = df_train["Pred_Weight"].fillna(1.0)

# 8. 확인용 저장
df_train.to_excel("weight_mapping_check.xlsx", index=False)
print("✅ weight 매핑 결과가 'weight_mapping_check.xlsx'에 저장되었습니다.")

