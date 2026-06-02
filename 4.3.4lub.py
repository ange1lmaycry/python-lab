import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

df = pd.read_csv(r"D:\python\lubs\healthcare-dataset-stroke-data.csv")

bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)

df = pd.get_dummies(
    df,
    columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    drop_first=True,
)

scaler = MinMaxScaler()
scaler.fit(df[["age", "bmi", "avg_glucose_level"]])
df[["age", "bmi", "avg_glucose_level"]] = scaler.transform(
    df[["age", "bmi", "avg_glucose_level"]]
)

important_columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
X = df[important_columns]
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
ada_model.fit(X_train, y_train)
y_proba_ada = ada_model.predict_proba(X_test)[:, 1]
auc_ada = roc_auc_score(y_test, y_proba_ada)
print(f"AdaBoost AUC: {auc_ada:.4f}")

gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb_model.fit(X_train, y_train)
y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_proba_gb)
print(f"Gradient Boosting AUC: {auc_gb:.4f}")

plt.figure(figsize=(8, 6))

fpr_ada, tpr_ada, _ = roc_curve(y_test, y_proba_ada)
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {auc_ada:.3f})", linewidth=2)

fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые")
plt.legend()
plt.grid(True)
plt.show()
