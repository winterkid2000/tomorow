mport torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, classification_report
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.lambda_), None

# 2. Discriminator (감별사) 클래스 추가
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # [중요] 이 Sigmoid 활성화 함수가 반드시 마지막에 있어야 합니다.
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)  

class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        
        # Classifier 직전의 벡터를 'feature_embedding'으로 정의
        feature_embedding = self.norm(x[:, 0, :])
        
        # 최종 예측값 계산
        final_prediction = self.classifier(feature_embedding)
        
        # 최종 예측값과 특징 벡터를 모두 반환
        return final_prediction, feature_embedding


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    auc
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    auc
)

def evaluate_thresholds(y_true, y_pred_prob, save_path=r"C:\Users\MIM\Desktop\50%\Onco\PRE\shap\precision_recall_vs_threshold01.png"):
    y_true = np.array(y_true).flatten()
    y_pred_prob = np.array(y_pred_prob).flatten()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    # 그래프 저장
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, recall[:-1], label='Recall', color='red')
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs Threshold (AUC = {pr_auc:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" 그래프 저장 완료: {save_path}")

    print("\n[ Summary at Selected Thresholds]")
    for thresh in [0.9, 0.7, 0.5, 0.4, 0.3]:
        y_pred = (y_pred_prob > thresh).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        class1 = report.get('1', {'precision': 0, 'recall': 0, 'f1-score': 0})
        print(f" Threshold = {thresh:.2f} | Acc = {acc:.4f} | Recall = {class1['recall']:.4f} | Precision = {class1['precision']:.4f} | F1 = {class1['f1-score']:.4f}")

    # 전체 threshold에서 평가
    print("\n[ Best Thresholds]")
    best_recall = 0
    best_recall_acc = 0
    best_threshold = 0
    best_overall_threshold = 0
    best_overall_score = 0

    for thresh in thresholds:
        y_pred = (y_pred_prob > thresh).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)

        # 1) Recall이 최대인 것 중 Accuracy가 가장 높은 threshold
        if rec > best_recall or (rec == best_recall and acc > best_recall_acc):
            best_recall = rec
            best_recall_acc = acc
            best_threshold = thresh

        # 2) 전체 평가 지표 평균값 기준으로 가장 높은 threshold
        avg_score = (rec + acc + f1 + prec) / 4
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_overall_threshold = thresh

    print(f"  Best for Recall + Acc => Threshold: {best_threshold:.4f} | Recall: {best_recall:.4f} | Accuracy: {best_recall_acc:.4f}")
    print(f"  Best Overall (Avg of All Scores) => Threshold: {best_overall_threshold:.4f} | Avg Score: {best_overall_score:.4f}")

    return best_threshold, best_overall_threshold


def evaluate_metrics(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_pred)
    }

    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    return df_metrics
