from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("NEWlr_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("NEWscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            feature1 = float(request.form["feature1"])
            feature2 = float(request.form["feature2"])
            feature3 = float(request.form["feature3"])
            feature4 = float(request.form["feature4"])

            input_data = np.array([[feature1, feature2, feature3,feature4]])

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]

            return render_template("result.html", result=prediction)

        except Exception as e:
            return render_template("result.html", result=f"Error: {e}")

    return render_template("predict.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
