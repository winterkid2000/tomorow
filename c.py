# A feature에 대해 DSC만 사용할 경우
from sklearn.metrics import r2_score

rf_single = RandomForestRegressor(n_estimators=100, random_state=0)
rf_single.fit(X[["DSC"]], Y["A_feature"])
r2_single = r2_score(Y["A_feature"], rf_single.predict(X[["DSC"]]))

# 전체 지표 사용할 경우
rf_full = RandomForestRegressor(n_estimators=100, random_state=0)
rf_full.fit(X, Y["A_feature"])
r2_full = r2_score(Y["A_feature"], rf_full.predict(X))

print(f"R² with DSC only: {r2_single:.4f}")
print(f"R² with all metrics: {r2_full:.4f}")
