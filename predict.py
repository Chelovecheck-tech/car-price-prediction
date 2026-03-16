import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ... остальной код predict
# загрузка модели
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    try:
        enginesize = float(input("Engine size: "))
        horsepower = float(input("Horsepower: "))
        curbweight = float(input("Curb weight: "))
        carwidth = float(input("Car width: "))
        highwaympg = float(input("Highway mpg: "))
        break
    except ValueError:
        print("Please enter valid numeric values for all features.")


# создаем dataframe с нужными колонками
data = {
    "enginesize": enginesize,
    "horsepower": horsepower,
    "curbweight": curbweight,
    "carwidth": carwidth,
    "highwaympg": highwaympg
}

df = pd.DataFrame([data]) 



df = df.reindex(columns=model.feature_names_in_, fill_value=0)
# предсказание
price = model.predict(df)

print("Predicted car price:", round(price[0], 2))