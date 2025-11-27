import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# === Linear Regression stuff (already there) ===
with open("models_and_datasets/NEWlr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models_and_datasets/NEWscaler.pkl", "rb") as f:
    lr_scaler = pickle.load(f)

# === SVM stuff (NEW) ===
with open("models_and_datasets/svm_models.pkl", "rb") as f:
    svm_models = pickle.load(f)   # {"Linear": ..., "Poly": ..., "RBF": ...}

with open("models_and_datasets/svm_scaler.pkl", "rb") as f:
    svm_scaler = pickle.load(f)
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

            input_data = np.array([[feature1, feature2, feature3, feature4]])
            input_scaled = lr_scaler.transform(input_data)
            prediction = lr_model.predict(input_scaled)[0]

            return render_template(
                "result.html",
                result=prediction,
                model_name="Linear Regression (Battery Drop)",
                show_chart=True,
                class_label=None
            )

        except Exception as e:
            return render_template(
                "result.html",
                result=f"Error: {e}",
                model_name="Linear Regression",
                show_chart=False,
                class_label=None
            )

    return render_template("predict.html")


@app.route("/predict_svm", methods=["GET", "POST"])
def predict_svm():
    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            kernel = request.form.get("kernel", "RBF")  # "Linear", "Poly", "RBF"

            input_data = np.array([[study_hours, attendance]])
            input_scaled = svm_scaler.transform(input_data)

            if kernel == "Linear":
                model = svm_models["Linear"]
            elif kernel == "Poly":
                model = svm_models["Poly"]
            else:
                model = svm_models["RBF"]

            prediction = model.predict(input_scaled)[0]

            # Map numeric prediction to label (0 -> Fail, 1 -> Pass)
            class_label = None
            try:
                pred_int = int(prediction)
                if pred_int == 1:
                    class_label = "Pass"
                elif pred_int == 0:
                    class_label = "Fail"
                else:
                    class_label = str(prediction)
            except:
                class_label = str(prediction)

            return render_template(
                "result.html",
                result=prediction,
                model_name=f"SVM ({kernel} kernel)",
                show_chart=False,          # ❌ no donut for SVM
                class_label=class_label    # ✅ visible Pass/Fail
            )

        except Exception as e:
            return render_template(
                "result.html",
                result=f"Error: {e}",
                model_name="SVM",
                show_chart=False,
                class_label=None
            )

    return render_template("svm_predict.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
