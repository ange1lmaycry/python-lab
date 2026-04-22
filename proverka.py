import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor 
from sklearn import tree
df= pd.read_csv(r"D:/python/lubs/job_salary_prediction_dataset.csv")
print(df.head())
print(f"Количество объектов: {len(df)}")
df.info()
nan_matrix = df.isnull() 
print(nan_matrix.sum())
df = pd.get_dummies(
    df,
    columns=["job_title", "education_level", "location", "industry", "company_size", "remote_work"],
    drop_first=True,
)
print("после преобразования")
print(df.head())
X = df.drop(["salary"], axis=1)
y = df["salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

dt_regressor_model = DecisionTreeRegressor(max_depth=15, max_leaf_nodes =300 ) 
dt_regressor_model.fit(X_train, y_train)
tree.plot_tree(dt_regressor_model, feature_names=X.columns, filled=True)
print(plt.show())
y_pred_test = dt_regressor_model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("проверка модели:")
print(RMSE)
r2=r2_score(y_test,y_pred_test)
print(r2)







