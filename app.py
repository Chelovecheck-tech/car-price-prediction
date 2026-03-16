from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd

app = Flask(__name__, static_folder=".", static_url_path="")

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    price = model.predict(df)[0]
    return jsonify({"price": round(float(price), 2)})

if __name__ == "__main__":
    app.run(debug=True)