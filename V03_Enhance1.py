import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# FFT_model.py에서 필요한 모든 클래스와 함수를 import
from FFT_model1 import FTTransformer, Discriminator, GradientReversalLayer, evaluate_thresholds, evaluate_metrics

# --------------------------------------------------------------------------
# 0. 유틸리티 함수 정의
# --------------------------------------------------------------------------
def summarize_shap_importance(shap_values, feature_names, top_k=20):
    """SHAP 값 기반으로 특징 중요도 요약"""
    shap_array = np.abs(shap_values).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": shap_array
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)
    return df_importance.head(top_k)

# --------------------------------------------------------------------------
# 1. 데이터 로딩 및 병합
# --------------------------------------------------------------------------
# [설정] 사용자 환경에 맞게 파일 경로를 수정하세요.
radiomics_path = r"c:\Users\MIM\Desktop\새 폴더 (14)\토-온\조영 인 힘\토탈-온코-조영증강X.xlsx" # 라디오믹스, Label, Patient_N, domain_label 포함
seg_metrics_path = r"c:\Users\MIM\Desktop\지표\Normal\PRE_Total-Onco_statistics_edit_2.xlsx"     # Patient_N, DSC, HD95 등 포함

# 데이터 불러오기
df_radiomics = pd.read_excel(radiomics_path)
df_seg_metrics = pd.read_excel(seg_metrics_path)

# 'Patient_N'을 기준으로 두 데이터프레임 병합
df_merged = pd.merge(df_radiomics, df_seg_metrics[['Patient_N', 'Dice_Sørensen_Coefficient(DSC)', 'Volume_Difference', 'Hausdorff_Distance', 'Hausdorff_Distance_95', 'Average_Surface_Distance', 'Normalised_Surface_Distance_1', 'Normalised_Surface_Distance_2']], on='Patient_N', how='left')

df_merged.fillna(0, inplace=True)


# 최종 학습에 사용할 데이터프레임 선택 (예: Enhanced 케이스만)
df = df_merged[df_merged["Enhanced"] == 0].copy()

# --------------------------------------------------------------------------
# 2. 전역 변수 설정
# --------------------------------------------------------------------------
# 라디오믹스 특징, seg 지표, 기타 정보 컬럼을 정의
info_cols = ['Label', 'Patient_N', 'domain_label', 'Enhanced'] # Enhanced 컬럼도 특징이 아니므로 제외 리스트에 추가
seg_metric_cols = ['Dice_Sørensen_Coefficient(DSC)', 'Volume_Difference', 'Hausdorff_Distance', 'Hausdorff_Distance_95', 'Average_Surface_Distance', 'Normalised_Surface_Distance_1', 'Normalised_Surface_Distance_2']

# 제외할 모든 컬럼 리스트를 하나로 합칩니다.
non_feature_cols = info_cols + seg_metric_cols

# 위 리스트에 포함되지 않은 모든 컬럼을 feature_cols로 정의합니다.
feature_cols = [col for col in df.columns if col not in non_feature_cols]

unique_patients = df['Patient_N'].unique()
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

# 결과 저장을 위한 딕셔너리 및 데이터프레임 초기화
df_recall_results = pd.DataFrame(index=metrics)
df_overall_results = pd.DataFrame(index=metrics)
all_predictions_raw = {}
all_shap_summaries = {}

# --------------------------------------------------------------------------
# 3. 30회 교차 검증 루프 시작
# --------------------------------------------------------------------------
for seed in range(30):
    print(f"--- Starting Fold {seed + 1}/30 ---")

    
    
    # 환자 단위로 Train/Test 분할
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=seed)
    df_train = df[df['Patient_N'].isin(train_patients)]
    df_test = df[df['Patient_N'].isin(test_patients)]

    if df_train.isnull().values.any():
        print(f"!!! [경고] Fold {seed+1}: df_train에 결측치가 있습니다. 아래는 결측치가 있는 컬럼 목록입니다.")
    # 각 컬럼별 결측치 개수를 출력합니다.
        print(df_train.isnull().sum()[df_train.isnull().sum() > 0])
    # [임시 조치] 결측치를 0으로 채워서 계속 진행해볼 수 있습니다.
    # df_train.fillna(0, inplace=True)
    # df_test.fillna(0, inplace=True)
    
    if df_test.isnull().values.any():
        print(f"!!! [경고] Fold {seed+1}: df_test에 결측치가 있습니다.")
        print(df_test.isnull().sum()[df_test.isnull().sum() > 0])
    # df_train.fillna(0, inplace=True)
    # df_test.fillna(0, inplace=True)
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_rad = scaler.fit_transform(df_train[feature_cols].values.astype(np.float32))
    X_test_rad = scaler.transform(df_test[feature_cols].values.astype(np.float32))

    # 라벨 및 지표 준비
    y_train_label = df_train['Label'].values.astype(np.float32)
    y_train_domain = df_train['domain_label'].values.astype(np.float32)
    y_test_label = df_test['Label'].values.astype(np.float32)
    
    X_train_seg = df_train[seg_metric_cols].values.astype(np.float32)
    X_test_seg = df_test[seg_metric_cols].values.astype(np.float32)
    
    # Tensor 변환
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_rad_tensor = torch.tensor(X_train_rad, dtype=torch.float32).to(device)
    y_train_label_tensor = torch.tensor(y_train_label, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_domain_tensor = torch.tensor(y_train_domain, dtype=torch.float32).unsqueeze(1).to(device)
    X_train_seg_tensor = torch.tensor(X_train_seg, dtype=torch.float32).to(device)
    
    X_test_rad_tensor = torch.tensor(X_test_rad, dtype=torch.float32).to(device)
    y_test_label_tensor = torch.tensor(y_test_label, dtype=torch.float32).unsqueeze(1).to(device)

    # 모델, 손실함수, 옵티마이저 정의
    d_model = 128
    predictor = FTTransformer(input_dim=X_train_rad_tensor.shape[1], d_model=d_model).to(device)
    discriminator = Discriminator(input_dim=d_model + len(seg_metric_cols)).to(device)

    criterion_pred = nn.BCELoss()
    criterion_domain = nn.BCELoss()

    optimizer_pred = torch.optim.AdamW(predictor.parameters(), lr=1e-5)
    optimizer_domain = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    
    best_val_loss = float("inf")
    best_predictor_state = None
    
    EPOCHS = 2000
    for epoch in range(EPOCHS):
        predictor.train()
        discriminator.train()
        optimizer_pred.zero_grad()
        optimizer_domain.zero_grad()

        p = float(epoch) / EPOCHS
        lambda_val = 2. / (1. + np.exp(-10. * p)) - 1

        # 순전파
        pred_clinical, feature_embedding = predictor(X_train_rad_tensor)
        discriminator_input = torch.cat([feature_embedding, X_train_seg_tensor], dim=1)
        reversed_input = GradientReversalLayer.apply(discriminator_input, lambda_val)
        pred_domain = discriminator(reversed_input)
        
        # 손실 계산
        loss_p = criterion_pred(pred_clinical, y_train_label_tensor)
        loss_d = criterion_domain(pred_domain, y_train_domain_tensor)
        total_loss = loss_p + loss_d

        # 역전파 및 최적화
        total_loss.backward()
        optimizer_pred.step()
        optimizer_domain.step()
        
        # 검증 및 최고 모델 저장
        predictor.eval()
        with torch.no_grad():
            val_outputs, _ = predictor(X_test_rad_tensor)
            val_loss = criterion_pred(val_outputs, y_test_label_tensor)
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_predictor_state = predictor.state_dict()
        
        if (epoch + 1) % 500 == 0:
            print(f"Fold {seed+1} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # ----------------------
    # 4. Fold별 평가 및 SHAP 분석
    # ----------------------
    predictor.load_state_dict(best_predictor_state)
    predictor.eval()

    with torch.no_grad():
        y_pred_prob, _ = predictor(X_test_rad_tensor)
        y_pred_prob = y_pred_prob.cpu().numpy()

    df_result = pd.DataFrame({
        "Patient_N": df_test['Patient_N'].values,
        "Pred_Prob": y_pred_prob.squeeze(),
        "Label": y_test_label.squeeze()
    })
    
    df_average = df_result.groupby("Patient_N").agg({"Pred_Prob": "mean", "Label": "first"}).reset_index()
    best_recall_thresh, best_overall_thresh = evaluate_thresholds(df_average["Label"], df_average["Pred_Prob"])

    # 평가 지표 계산
    df_average["Pred_Label_Recall"] = (df_average["Pred_Prob"] > best_recall_thresh).astype(int)
    df_recall_results[f"Fold {seed + 1}"] = evaluate_metrics(df_average["Label"], df_average["Pred_Label_Recall"])["Value"].values

    df_average["Pred_Label_Overall"] = (df_average["Pred_Prob"] > best_overall_thresh).astype(int)
    df_overall_results[f"Fold {seed + 1}"] = evaluate_metrics(df_average["Label"], df_average["Pred_Label_Overall"])["Value"].values
    
    all_predictions_raw[f"Fold {seed + 1}"] = df_average

    # SHAP 분석
    def model_predict_for_shap(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions, _ = predictor(x_tensor)
            return predictions.cpu().numpy()

    explainer = shap.KernelExplainer(model_predict_for_shap, X_train_rad[:50])
    shap_values = explainer.shap_values(X_test_rad, nsamples=100)
    all_shap_summaries[f"Fold {seed + 1}"] = summarize_shap_importance(np.array(shap_values).squeeze(), feature_cols)

# --------------------------------------------------------------------------
# 5. 최종 결과 취합 및 저장
# --------------------------------------------------------------------------
# [설정] 결과 저장 경로를 수정하세요.
output_dir = r"C:\Users\MIM\Desktop\지표\Normal\금"

# 성능 지표 저장
df_final_summary = pd.DataFrame(index=metrics)
df_final_summary["Avg Recall"] = df_recall_results.mean(axis=1)
df_final_summary["Std Recall"] = df_recall_results.std(axis=1)
df_final_summary["Avg Overall"] = df_overall_results.mean(axis=1)
df_final_summary["Std Overall"] = df_overall_results.std(axis=1)

df_recall_results.to_excel(f"{output_dir}/Folds_Best_Recall.xlsx")
df_overall_results.to_excel(f"{output_dir}/Folds_Best_Overall.xlsx")
df_final_summary.to_excel(f"{output_dir}/Final_Summary_Evaluation.xlsx")

# 예측 결과 및 SHAP 결과 저장
pd.concat([df.assign(Fold=f"Fold") for i, df in all_predictions_raw.items()]).to_excel(f"{output_dir}/All_Raw_Predictions.xlsx", index=False)
pd.concat([df.assign(Fold=f"Fold") for i, df in all_shap_summaries.items()]).to_excel(f"{output_dir}/All_SHAP_Summaries.xlsx", index=False)

print("--- 모든 작업이 완료되었습니다. ---")







