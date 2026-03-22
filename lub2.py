import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv(r"D:\python\lubs\healthcare-dataset-stroke-data.csv")
print(df.head(10))
df.info()
print(df.dtypes)
cols = df.columns
print(cols)
#  кол-во пропущеных значений
print("пропущеные значения")
nan_matrix = df.isnull()
print(nan_matrix.sum())
#  заполнение значений медианой
print("заполение пропусков Медианой")
bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)
print(df["bmi"].isnull().sum())
# проверка на наличие пропусков
print("пропущеные значения")
nan_matrix = df.isnull()
print(nan_matrix.sum())
#  нормализация данных
scaler = MinMaxScaler()
scaler.fit(df[["age", "avg_glucose_level"]])
df[["age", "avg_glucose_level"]] = scaler.transform(df[["age", "avg_glucose_level"]])
print("Результат нормолизации:")
print(df[["age", "avg_glucose_level"]])
#  преобразование категориальных данных
df = pd.get_dummies(
    df,
    columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    drop_first=True,
)
print("после преобразования")
print(df.head())
# 1.	Разделить датасет, подготовленный на первой лабораторной работе, на обучающую и тестовую выборки
X = df.drop(["bmi"], axis=1)
y = df["bmi"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# 2.	Решить задачу регрессии для одного из непрерывных признаков в датасете
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)
print("Первые 10 предсказаний bmi:")
print(y_pred_test[:10])
# 3.	Оценить работу регрессионной модели.
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("проверка модели:")
print(RMSE)
# 4.	Решить задачу классификации
scaler = MinMaxScaler()
scaler.fit(df[["age", "bmi", "avg_glucose_level"]])
df[["age", "bmi", "avg_glucose_level"]] = scaler.transform(
    df[["age", "bmi", "avg_glucose_level"]]
)
X = df.drop(["stroke"], axis=1)
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
logreg_model = LogisticRegression(
    max_iter=10000, class_weight={0: 1, 1: 10}, random_state=42
)
logreg_model.fit(X_train, y_train)
y_pred_test = logreg_model.predict(X_test)
print("Первые 10 предсказаний:")
print(y_pred_test[:10])
# 5.	Оценить работу классификационной модели. При плохих результатах подумать как можно его улучшить
print(
    "Оцениваем баланс классов что бы подобрать чем мы будем оценивать классификационную модель"
)
counts = df["stroke"].value_counts()
print(counts.min() / counts.sum() * 100, end="")
print("%")
# используем матрицу ошибок
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="bwr")
plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
report = classification_report(y_test, y_pred_test)
print(report)
