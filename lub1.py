import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
df = pd.read_csv(r"D:\python\lubs\healthcare-dataset-stroke-data.csv")
# 2-ой пункт
# получение первых 10 строк датасета
print(df.head(10))
# получение информации о датасете
df.info()
# типы данных в столбцах
print(df.dtypes)
# получение названий колонок
cols = df.columns
print(cols)
# пункт 3 кол-во пропущеных значений
print("пропущеные значения")    
nan_matrix = df.isnull()
print(nan_matrix.sum())
# пункт 4 заполнение значений медианой
print("заполение пропусков Медианой")
bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)
print(df["bmi"].isnull().sum())
# проверка на наличие пропусков
print("пропущеные значения")
nan_matrix = df.isnull()
print(nan_matrix.sum())
# пункт 5 нормализация данных
scaler = MinMaxScaler()
scaler.fit(df[["age", "bmi", "avg_glucose_level"]])
df[["age", "bmi", "avg_glucose_level"]] = scaler.transform(
    df[["age", "bmi", "avg_glucose_level"]]
)
print("Результат нормолизации:")
print(df[["age", "bmi", "avg_glucose_level"]])
# пункт 6 преобразование категориальных данных
df = pd.get_dummies(
    df,
    columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    drop_first=True,
)
print("после преобразования")
print(df.head())
X = df.drop(['avg_glucose_level'], axis=1)
y = df['avg_glucose_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("оценка модели")
print(RMSE)
