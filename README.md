#  Car Price Prediction

Веб-приложение для предсказания цены автомобиля на основе его технических характеристик. Модель обучена на датасете из 205 автомобилей.



 Структура проекта

```
project/
├── data/
│   └── car_data.csv        # Датасет
├── venv/                   # Виртуальное окружение
├── car_price_model.pkl     # Обученная модель
├── train.py                # Обучение модели
├── app.py                  # Flask-сервер
├── index.html              # Фронтенд
└── requirements.txt        # Зависимости
```

---

Модель

- **Алгоритм:** Linear Regression (sklearn)
- **Целевая переменная:** `price`
- **MAE:** 2680.26
- **Accuracy (R²):** 0.869

 Признаки, используемые для предсказания:

| Признак | Описание |
|---|---|
| `enginesize` | Объём двигателя |
| `horsepower` | Мощность (л.с.) |
| `curbweight` | Снаряжённая масса (фунты) |
| `carwidth` | Ширина кузова (дюймы) |
| `highwaympg` | Расход на шоссе (миль/галлон) |

Удалённые колонки при очистке:

`car_ID`, `symboling`, `CarName`, `doornumber`, `enginetype`, `cylindernumber`, `enginelocation`, `fuelsystem`

---
 Запуск

**1. Клонировать репозиторий**
```bash
git clone https://github.com/Chelovecheck-tech/car-price-prediction.git
cd car-price-prediction
```

2. Создать виртуальное окружение и установить зависимости**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

3. Обучить модель**
```bash
python train.py
```

4. Запустить сервер**
```bash
python app.py
```

5. Открыть в браузере**
```
http://localhost:5000
```

---

 Зависимости

```
flask
pandas
scikit-learn
```



📊 Датасет

Датасет содержит 205 записей об автомобилях разных марок (BMW, Toyota, Honda, Volkswagen и др.).

Источник: [Automobile Dataset — Kaggle](https://www.kaggle.com/datasets/toramky/automobile-dataset)
