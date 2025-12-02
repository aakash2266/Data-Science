import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# === Linear Regression stuff ===
with open("models_and_datasets/NEWlr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models_and_datasets/NEWscaler.pkl", "rb") as f:
    lr_scaler = pickle.load(f)

# === SVM stuff ===
with open("models_and_datasets/svm_models.pkl", "rb") as f:
    svm_models = pickle.load(f)   # {"Linear": ..., "Poly": ..., "RBF": ...}

with open("models_and_datasets/svm_scaler.pkl", "rb") as f:
    svm_scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


# ---------- LINEAR REGRESSION ----------
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
                show_chart=True,      # show donut chart
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


# ---------- SVM ----------
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
            except Exception:
                class_label = str(prediction)

            return render_template(
                "result.html",
                result=prediction,
                model_name=f"SVM ({kernel} kernel)",
                show_chart=False,          # no donut for SVM
                class_label=class_label
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


# ---------- DECISION TREE SETUP ----------
# load dataset once for Decision Tree playground
df_tree = pd.read_csv("models_and_datasets/logistic_regression_1000.csv")
X = df_tree.drop("Pass_Fail", axis=1)
y = df_tree["Pass_Fail"]


# ---------- DECISION TREE ----------
@app.route("/predict_tree", methods=["GET", "POST"])
def predict_tree():
    if request.method == "POST":
        try:
            # ---- user inputs for prediction ----
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])

            # ---- hyperparameters from form ----
            criterion = request.form.get("criterion", "gini")
            splitter = request.form.get("splitter", "best")

            max_depth_raw = request.form.get("max_depth", "")
            max_depth = int(max_depth_raw) if max_depth_raw else None

            min_samples_split = int(request.form.get("min_samples_split", 2))
            min_samples_leaf = int(request.form.get("min_samples_leaf", 1))

            max_features_raw = request.form.get("max_features", "None")
            max_features = None if max_features_raw == "None" else max_features_raw

            # ---- train / evaluate model with chosen params ----
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=42
            )

            deeptree = DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
            deeptree.fit(X_train, y_train)

            y_pred_test = deeptree.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_test)

            # accuracy to %
            acc_percent = accuracy * 100.0

            # simple explanation of how good the accuracy is
            if acc_percent >= 90:
                accuracy_desc = "Excellent performance – the tree is classifying most students correctly."
            elif acc_percent >= 80:
                accuracy_desc = "Good performance – the tree is doing well with your current settings."
            elif acc_percent >= 70:
                accuracy_desc = "Okay performance – could be improved by tuning depth or min samples."
            else:
                accuracy_desc = "Low accuracy – the model is struggling. Try limiting depth or increasing min samples."

            # ---- prediction for the user sample ----
            sample = np.array([[study_hours, attendance]])
            user_pred = deeptree.predict(sample)[0]     # usually 0/1

            # map to Pass / Fail safely
            try:
                pred_int = int(user_pred)
                if pred_int == 1:
                    class_label = "Pass"
                elif pred_int == 0:
                    class_label = "Fail"
                else:
                    class_label = str(user_pred)
            except Exception:
                class_label = str(user_pred)

            # ---- feature importance ----
            importances = deeptree.feature_importances_
            feature_names = ["Study_Hours", "Attendance"]

            feature_importance = [
                {"name": feature_names[i], "value": float(importances[i])}
                for i in range(len(importances))
            ]

            # ---- tree complexity ----
            tree_depth = deeptree.get_depth()
            leaf_count = deeptree.get_n_leaves()

            # ---- top feature ----
            if importances.sum() > 0:
                top_feature = feature_names[int(np.argmax(importances))]
            else:
                top_feature = "Study_Hours / Attendance"

            # ---- ASCII tree ----
            ascii_tree = export_text(deeptree, feature_names=feature_names)

            return render_template(
                "result.html",
                result=user_pred,
                class_label=class_label,
                model_name="Decision Tree",
                accuracy=round(acc_percent, 2),
                accuracy_desc=accuracy_desc,
                params={
                    "criterion": criterion,
                    "splitter": splitter,
                    "max_depth": max_depth_raw or "None",
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features_raw,
                },
                feature_importance=feature_importance,
                tree_depth=tree_depth,
                leaf_count=leaf_count,
                top_feature=top_feature,
                ascii_tree=ascii_tree,
                show_chart=False
            )

        except Exception as e:
            # instead of 500, show error text on your result page
            return render_template(
                "result.html",
                result=f"Error in Decision Tree route: {e}",
                model_name="Decision Tree",
                show_chart=False,
                class_label=None
            )

    # GET → just show the form
    return render_template("predict_tree.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
