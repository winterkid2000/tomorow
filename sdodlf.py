import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from FFT_model import FTTransformer, evaluate_thresholds, evaluate_metrics
import torch.nn as nn
import numpy as np
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

def summarize_shap_importance(shap_values, feature_names, top_k=20):
    """
    SHAP 값 기반 중요도 요약 테이블 반환
    """
    shap_array = np.abs(shap_values).mean(axis=0)  # 평균 절댓값
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": shap_array
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)

    return df_importance.head(top_k)

df = pd.read_excel(r"c:\Users\MIM\Desktop\새 폴더 (14)\토-온\조영 인 힘\토탈-온코-조영증강X1_1.xlsx")
df_enhanced = df[df["Enhanced"] == 0].copy() ##### 1 = Enhanced, 0 = non-enhance
df_seg = pd.read_excel("seg_metrics.xlsx", index_col=0)
df_delta = pd.read_excel("delta_radiomics.xlsx", index_col=0)

# 2. 🎯 공통 환자 ID 정렬
common_ids = df_seg.index.intersection(df_delta.index)
X = pd.concat([df_delta.loc[common_ids], df_seg.loc[common_ids]], axis=1)

# 3. 🧪 proxy Y 생성: PCA 1st component (변동성 가장 큰 축)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=1)
Y_proxy = np.abs(pca.fit_transform(X_scaled).flatten())

# 4. 🧠 XGBoost 회귀 학습
model = XGBRegressor(n_estimators=100, random_state=0)
model.fit(X, Y_proxy)

# 5. 🧮 예측 weight 계산
raw_pred = model.predict(X)
pred_weights = 1 / (raw_pred + 1e-6)

# index는 그대로 숫자 ID (예: 1, 2, 3...)
df_pred_weights = pd.DataFrame(pred_weights, index=common_ids, columns=["Pred_Weight"])
df_pred_weights = df_pred_weights.reset_index().rename(columns={"index": "Patient_N"})

feature_cols = df_enhanced.drop(columns=["Label", "Patient_N"]).columns.tolist()

# 1. 환자 ID 목록
valid_ids = df_enhanced["Patient_N"].unique()

# 2. df_pred_weights에서 그 환자만 남기기
df_pred_weights = df_pred_weights[df_pred_weights["Patient_N"].isin(valid_ids)]

# 미리 metric 이름 설정
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

# 빈 DataFrame (행은 metric 이름, 열은 나중에 반복문으로 추가됨)
df_average_best_recall = pd.DataFrame(index=metrics)
df_average_best_overall = pd.DataFrame(index=metrics)
df_max_best_recall = pd.DataFrame(index=metrics)
df_max_best_overall = pd.DataFrame(index=metrics)
all_pred_raw ={}
all_summary_dfs_recall = {}
all_summary_dfs_average = {}



for seed in range(30):
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=seed)

    # 2. Train/Test 나누기
    df_train = df_enhanced[df_enhanced["Patient_N"].isin(train_patients)]
    df_test = df_enhanced[df_enhanced["Patient_N"].isin(test_patients)]

    df_train = df_train.merge(df_pred_weights, on="Patient_N", how="left")
    df_train["Pred_Weight"] = df_train["Pred_Weight"].fillna(1.0)

    # 3. Feature & Label 분리
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(df_test[feature_cols].values.astype(np.float32))

    y_train = df_train["Label"].values.astype(np.float32)
    y_test = df_test["Label"].values.astype(np.float32)
    test_patients = df_test["Patient_N"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Tensor 변환
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    # --- 가중치 텐서로 변환 ---
    train_weights = torch.tensor(df_train["Pred_Weight"].values, dtype=torch.float32).unsqueeze(1).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = FTTransformer(input_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss(reduction="none")

    best_val_loss = float("inf")
    best_model_state = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    EPOCHS = 2000
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss_per_sample = criterion(outputs, y_train.to(device))
        loss = (loss_per_sample * train_weights).mean()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test.to(device))
            val_loss = criterion(val_outputs, y_test.to(device))
            val_losses.append(val_loss.item())

            preds = (val_outputs > 0.5).float()
            correct = (preds.cpu() == y_test).sum().item()
            val_acc = correct / y_test.shape[0]
            val_accuracies.append(val_acc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()

        print(f"Epoch {epoch + 1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    import matplotlib.pyplot as plt

    model.load_state_dict(best_model_state)
    model.eval()

    y_pred_prob = model(X_test.to(device)).cpu().detach().numpy()
    # evaluate_thresholds(y_true, y_pred_prob)

    df_result = pd.DataFrame(
        {"Patient_N": test_patients, "Pred_Prob": y_pred_prob.squeeze(), "Label": y_test.squeeze()})

    ## Average: 환자 단위 평균
    df_average = df_result.groupby("Patient_N").agg({"Pred_Prob": "mean", "Label": "first"}).reset_index()

    # Best Threshold find
    best_recall_thresh, best_overall_thresh = evaluate_thresholds(df_average["Label"], df_average["Pred_Prob"])

    df_average["Pred_Label"] = (df_average["Pred_Prob"] > best_recall_thresh).astype(int)
    df_average_val_best_recall = evaluate_metrics(df_average["Label"], df_average["Pred_Label"])
    correct_idx_recall = np.where(df_average["Label"] == df_average["Pred_Label"])[0]

    df_average["Pred_Label"] = (df_average["Pred_Prob"] > best_overall_thresh).astype(int)
    df_average_val_best_overall = evaluate_metrics(df_average["Label"], df_average["Pred_Label"])
    correct_idx_overall = np.where(df_average["Label"] == df_average["Pred_Label"])[0]

    import shap
    import pandas as pd


    # 1. 모델 출력 함수 정의
    def model_predict(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            return model(x_tensor).cpu().numpy()


    # 2. 데이터 (입력은 features임!)
    X_val_np = X_test.cpu().numpy()
    X_train_np = X_train.cpu().numpy()

    # 3. SHAP 계산
    explainer = shap.KernelExplainer(model_predict, X_train_np[:50])  # 배경 샘플은 일부만
    feature_names = df.drop(columns=["Label", "Patient_N"], errors="ignore").columns.tolist()

    X_val_correct_overall = X_val_np[correct_idx_overall]
    X_val_correct_recall = X_val_np[correct_idx_recall]

    shap_values_overall = explainer.shap_values(X_val_correct_overall, nsamples=100)  # 입력은 예측할 feature들
    shap_values_recall = explainer.shap_values(X_val_correct_recall, nsamples=100)  # 입력은 예측할 feature들

    summary_df_overall = summarize_shap_importance(shap_values_overall.squeeze(), feature_names, top_k=25)
    summary_df_recall = summarize_shap_importance(shap_values_recall.squeeze(), feature_names, top_k=25)


    df_average_best_recall[f"Fold {seed + 1:}"] = df_average_val_best_recall["Value"].values
    df_average_best_overall[f"Fold {seed + 1:}"] = df_average_val_best_overall["Value"].values
    all_summary_dfs_average[f"Fold {seed + 1:}"] = summary_df_overall
    all_summary_dfs_recall[f"Fold {seed + 1:}"] = summary_df_recall
    all_pred_raw [f"Fold {seed + 1:}"] = df_average


cat_all_pred_raw = pd.concat([df.assign(Fold=fold) for fold, df in all_pred_raw.items()],ignore_index=True)

cat_all_pred_raw.to_excel(r"C:\Users\MIM\Desktop\새 폴더 (6)\새 폴더\all_pred_raw.xlsx", index=False)


df_all_average = pd.DataFrame(index=metrics)

df_all_average[f"Avr recall"] = df_average_best_recall.iloc[:].mean(axis=1)
df_all_average[f"Avr overall"] = df_average_best_overall.iloc[:].mean(axis=1)
df_all_average[f"Avr recall std"] = df_average_best_recall.iloc[:].std(axis=1)
df_all_average[f"Avr overall std"] = df_average_best_overall.iloc[:].std(axis=1)



df_average_best_recall.to_excel(r"C:\Users\MIM\Desktop\새 폴더 (6)\새 폴더\Best_recall.xlsx", index=False)
df_average_best_overall.to_excel(r"C:\Users\MIM\Desktop\새 폴더 (6)\새 폴더\Best_overall.xlsx", index=False)
df_all_average.to_excel(r"C:\Users\MIM\Desktop\새 폴더 (6)\새 폴더\All_evaluation.xlsx", index=False)

combined_df_overall = pd.concat([df.assign(Fold=fold) for fold, df in all_summary_dfs_average.items()],ignore_index=True)
combined_df_recall = pd.concat([df.assign(Fold=fold) for fold, df in all_summary_dfs_recall.items()],ignore_index=True)

combined_df_overall.to_excel(r"C:\Users\MIM\Desktop\새 폴더 (6)\새 폴더\SHAP_overall.xlsx", index=False)
combined_df_recall.to_excel(r"C:\Users\MIM\Desktop\새 폴더 (6)\새 폴더\SHAP_recall.xlsx", index=False)
