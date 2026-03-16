import pandas as pd # для работы с данными в виде таблиц
import numpy as np # для работы с числами и массивами
import matplotlib.pyplot as plt # рисуте графики

from sklearn.model_selection import train_test_split # делит данные на обучающую и тестовую выборки
from sklearn.linear_model import LinearRegression # модель линейной регрессии
from sklearn.metrics import mean_absolute_error, r2_score # считает среднюю ошибку 
import pickle


df = pd.read_csv("data/car_data.csv")

drop_cols = [
    "car_ID", "symboling", "CarName", "doornumber",
    "enginetype", "cylindernumber", "enginelocation", "fuelsystem"
]
df = df.drop(drop_cols, axis=1)

# Преобразует буквы в числа
df = pd.get_dummies(df, drop_first=True)



# Разделяем X и y
X = df.drop("price", axis=1) # характеристики автомобиля
y = df["price"] # цена автомобиля

# Делим данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# x_train Характеристики машин для обучения
# y_train Цены машин для обучения
# x_test Характеристики новых машин, которые модель раньше не видела
# y_test Настоящие цены этих машин 


# Создаем модель и обучаем ее
model = LinearRegression()
model.fit(X_train, y_train)


# Сохраняем модель с помощью pickle

# Предсказывает цену для тестовых данных
y_pred = model.predict(X_test)

# Оценка
print("MAE:", mean_absolute_error(y_test, y_pred))
print("accuracy:", r2_score(y_test, y_pred))




# plt.figure(figsize=(8, 6)) # размер графика
# plt.scatter(y_test, y_pred, alpha=0.7, color='#1f77b4', edgecolor='k', label='Прогнозы модели') # рисуем точки на графике
# plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='red', lw=2, linestyle='-', label='Идеальное совпадение') # рисуем красную пунктирную линию для идеального совпадения
# plt.xlabel("Реальная цена", fontsize=12) # подпись для оси X
# plt.ylabel("Предсказанная цена", fontsize=12) # подпись для оси Y
# plt.title("Сравнение реальных и предсказанных цен", fontsize=14) # заголовок графика
# plt.xlim(15000, 25000) # устанавливаем пределы для оси X
# plt.ylim(15000, 25000) # устанавливаем пределы для оси Y
# plt.legend() # добавляем легенду
# plt.grid(True, linestyle=':') # добавляем сетку с пунктирными линиями
# plt.tight_layout() # оптимизируем расположение элементов на графике
# plt.show() 

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

plt.figure(figsize=(8, 6)) # размер графика
plt.scatter(y_test, y_pred, alpha=0.7, color='#1f77b4', edgecolor='k', label='Прогнозы модели') # рисуем точки на графике
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='red', lw=2, linestyle='-', label='Идеальное совпадение') # рисуем красную пунктирную линию для идеального совпадения
plt.xlabel("Реальная цена", fontsize=12) # подпись для оси X
plt.ylabel("Предсказанная цена", fontsize=12) # подпись для оси Y
plt.title("Сравнение реальных и предсказанных цен", fontsize=14) # заголовок графика
plt.xlim(0, 45000) # устанавливаем пределы для оси X
plt.ylim(0, 45000) # устанавливаем пределы для оси Y
plt.legend() # добавляем легенду
plt.grid(True, linestyle=':') # добавляем сетку с пунктирными линиями
plt.tight_layout() # оптимизируем расположение элементов на графике
# plt.savefig("predictions.png", dpi=300) # сохраняем график в файл с высоким разрешением
plt.show() 

