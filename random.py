import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def importance(seg_path, radio_path):

    X = pd.read_csv(seg_path, index_col=0)       
    Y = pd.read_csv(radio_path, index_col=0)   

    common_ids = X.index.intersection(Y.index)
    X = X.loc[common_ids]
    Y = Y.loc[common_ids]
  
    influence_matrix = []
    feature_names = Y.columns
    metric_names = X.columns

    for feature in feature_names:
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X, Y[feature])
        importances = rf.feature_importances_
        influence_matrix.append(importances)

    influence_df = pd.DataFrame(influence_matrix, index=feature_names, columns=metric_names)
    return influence_df

def main():
    print("Radiomics에 대한 Segmentation 지표 영향도 분석")
    seg_path = input("Segmentation 지표 CSV 경로 입력: ")
    radio_path = input("Radiomics CSV 경로 입력: ")          
    out_path = input("결과 저장 경로 (CSV): ")            
    influence_df = importance(seg_path, radio_path)
    influence_df.to_csv(out_path)
    print(f"저장 완료: {out_path}")

if __name__ == "__main__":
    main()


