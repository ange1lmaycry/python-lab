# Импортируем библиотеку pandas для работы с таблицами данных
import pandas as pd

# Импортируем инструменты для нормализации чисел из библиотеки scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Импортируем функцию для разделения данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

# Загружаем данные из интернета по ссылке
# sep='\t' говорит, что в файле разделитель - табуляция (это TSV формат)
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv", sep='\t')

# Выводим первые 5 строк датасета, чтобы посмотреть что внутри
print(df.head())

# Выводим информацию о датасете: типы данных, количество непустых значений
print(df.info())

# Выводим типы данных каждого столбца отдельно
print(df.dtypes)

# Сохраняем названия всех колонок в переменную cols
cols = df.columns

# Выводим названия колонок в виде обычного списка Python
print(cols.tolist())

# Считаем и выводим количество пропущенных значений в каждом столбце
print(df.isnull().sum())

# Заполняем пропуски в колонке choice_description
# mode()[0] - находим самое частое значение (моду)
# fillna() - заполняем этим значением все пустые места
df['choice_description'] = df['choice_description'].fillna(df['choice_description'].mode()[0])

# Проверяем что пропусков больше нет (должен вывести все нули)
print(df.isnull().sum())

# Превращаем цены из текста в числа
# str.replace('$', '') - убираем знак доллара (было "$2.39" стало "2.39")
# astype(float) - превращаем текст в число с плавающей точкой
df['item_price'] = df['item_price'].str.replace('$', '').astype(float)

# Создаем объект для MinMax нормализации (приводит числа к диапазону от 0 до 1)
scaler = MinMaxScaler()

# Применяем MinMax нормализацию к колонке с ценами и сохраняем результат в новую колонку
df['price_minmax'] = scaler.fit_transform(df[['item_price']])

# Создаем объект для Standard нормализации (Z-оценка, среднее=0, стандартное отклонение=1)
scaler2 = StandardScaler()

# Применяем Standard нормализацию и сохраняем в новую колонку
df['price_standard'] = scaler2.fit_transform(df[['item_price']])

# Преобразуем категориальные данные (названия товаров) в цифры через One-Hot Encoding
# columns=['item_name'] - указываем какую колонку преобразовывать
# drop_first=True - убираем первую колонку чтобы избежать дублирования (защита от переобучения)
df = pd.get_dummies(df, columns=['item_name'], drop_first=True)

# Разбиваем данные на обучающую (70%) и тестовую (30%) выборки
# test_size=0.3 - 30% данных пойдет в тест
# random_state=42 - фиксируем случайность, чтобы при каждом запуске деление было одинаковым
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Сохраняем обработанный полный датасет в CSV файл
# index=False - не сохраняем номера строк как отдельную колонку
df.to_csv("processed.csv", index=False)

# Сохраняем обучающую выборку
train.to_csv("train.csv", index=False)

# Сохраняем тестовую выборку
test.to_csv("test.csv", index=False)

# Выводим сообщение что все готово
print("Готово!")