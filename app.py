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

    # Собираем все 21 признак
    row = {
        "wheelbase":        data.get("wheelbase", 0),
        "carlength":        data.get("carlength", 0),
        "carwidth":         data.get("carwidth", 0),
        "carheight":        data.get("carheight", 0),
        "curbweight":       data.get("curbweight", 0),
        "enginesize":       data.get("enginesize", 0),
        "boreratio":        data.get("boreratio", 0),
        "stroke":           data.get("stroke", 0),
        "compressionratio": data.get("compressionratio", 0),
        "horsepower":       data.get("horsepower", 0),
        "peakrpm":          data.get("peakrpm", 0),
        "citympg":          data.get("citympg", 0),
        "highwaympg":       data.get("highwaympg", 0),
        "fueltype_gas":     data.get("fueltype_gas", 1),
        "aspiration_turbo": data.get("aspiration_turbo", 0),
        "carbody_hardtop":  data.get("carbody_hardtop", 0),
        "carbody_hatchback":data.get("carbody_hatchback", 0),
        "carbody_sedan":    data.get("carbody_sedan", 0),
        "carbody_wagon":    data.get("carbody_wagon", 0),
        "drivewheel_fwd":   data.get("drivewheel_fwd", 0),
        "drivewheel_rwd":   data.get("drivewheel_rwd", 0),
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    price = model.predict(df)[0]
    return jsonify({"price": round(float(price), 2)})

if __name__ == "__main__":
    app.run(debug=True)