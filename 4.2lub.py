import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(r"D:\python\lubs\healthcare-dataset-stroke-data.csv")

# заполнение пропусков медианой
bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)

# преобразование категориальных данных
df = pd.get_dummies(
    df,
    columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    drop_first=True,
)

# нормализация
scaler = MinMaxScaler()
scaler.fit(df[["age", "bmi", "avg_glucose_level"]])
df[["age", "bmi", "avg_glucose_level"]] = scaler.transform(
    df[["age", "bmi", "avg_glucose_level"]]
)


important_columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
X = df[important_columns]
y = df["stroke"]


rf_model = RandomForestClassifier(
    n_estimators=100, max_features="sqrt", oob_score=True, random_state=42, n_jobs=-1
)

rf_model.fit(X, y)

oob_accuracy = rf_model.oob_score_
print(f"\nСЛУЧАЙНЫЙ ЛЕСgit init")
print(f"OOB точность: {oob_accuracy:.4f}")
print(f"OOB ошибка: {1 - oob_accuracy:.4f}")

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print(f"\nВажность признаков:")
print(feature_importance.sort_values(ascending=False))
