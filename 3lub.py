import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
#  преобразование категориальных данных
df = pd.get_dummies(
    df,
    columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    drop_first=True,
)
print("после преобразования")
print(df.head())
# 1.	Разделить датасет, подготовленный на первой лабораторной работе, на обучающую и тuестовую выборки
X = df.drop(["stroke"], axis=1)
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
dt_classifier_model = DecisionTreeClassifier( 
max_depth=10, max_leaf_nodes =1000,class_weight='balanced')
dt_classifier_model.fit(X_train, y_train)
y_proba = dt_classifier_model.predict_proba(X_test) 
print(dt_classifier_model.classes_)
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 
1])
y_pred_test = dt_classifier_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.plot(fpr, tpr, marker='o') 
plt.ylim([0,1.1]) 
plt.xlim([0,1.1]) 
plt.ylabel('TPR') 
plt.xlabel('FPR') 
plt.title('ROC curve')
auc_metric = auc(fpr, tpr)
plt.show()
print(f"AUC = {auc_metric:.4f}")