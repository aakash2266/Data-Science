from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 🔹 Load your trained regression model
with open("NEWlr_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("NEWscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # 1️⃣ Get data from the form
        try:
            feature1 = float(request.form["feature1"])
            feature2 = float(request.form["feature2"])
            feature3 = float(request.form["feature3"])
            feature4 = float(request.form["feature4"])
            # 👉 Add / remove features as per YOUR model

            # 2️⃣ Make it into a 2D array for sklearn
            input_data = np.array([[feature1, feature2, feature3,feature4]])

            input_scaled = scaler.transform(input_data)

            # 3️⃣ Predict using your model
            prediction = model.predict(input_scaled)[0]

            # 4️⃣ Send result to result.html
            return render_template("result.html", result=prediction)

        except Exception as e:
            # In case something goes wrong
            return render_template("result.html", result=f"Error: {e}")

    # If GET request -> just show the form page
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
