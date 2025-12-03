import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
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
def describe_accuracy(acc_percent: float) -> str:
    if acc_percent >= 90:
        return "Excellent performance – the model is classifying most students correctly."
    elif acc_percent >= 80:
        return "Good performance – the model is doing well with your current settings."
    elif acc_percent >= 70:
        return "Okay performance – could be improved by tuning depth or min samples."
    else:
        return "Low accuracy – the model is struggling. Try limiting depth or increasing min samples."


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

@app.route("/predict_rf", methods=["GET", "POST"])
def predict_rf():
    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])

            # hyperparameters
            n_estimators = int(request.form.get("n_estimators", 100))
            criterion = request.form.get("criterion", "gini")
            max_depth_raw = request.form.get("max_depth", "")
            max_depth = int(max_depth_raw) if max_depth_raw else None

            min_samples_split = int(request.form.get("min_samples_split", 2))
            min_samples_leaf = int(request.form.get("min_samples_leaf", 1))

            max_features_raw = request.form.get("max_features", "sqrt")
            max_features = None if max_features_raw == "None" else max_features_raw

            # train / test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=42
            )

            forest = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
            forest.fit(X_train, y_train)

            # accuracy
            y_pred_test = forest.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)
            acc_percent = acc * 100.0
            accuracy_desc = describe_accuracy(acc_percent)

            # prediction for this student
            sample = np.array([[study_hours, attendance]])
            user_pred = forest.predict(sample)[0]
            label = "Pass" if int(user_pred) == 1 else "Fail"

            # feature importance (RF has it directly)
            importances = forest.feature_importances_
            feature_names = ["Study_Hours", "Attendance"]
            feature_importance = [
                {"name": feature_names[i], "value": float(importances[i])}
                for i in range(len(importances))
            ]

            return render_template(
                "result.html",
                result=user_pred,
                class_label=label,
                model_name="Random Forest",
                accuracy=round(acc_percent, 2),
                accuracy_desc=accuracy_desc,
                params={
                    "n_estimators": n_estimators,
                    "criterion": criterion,
                    "max_depth": max_depth_raw or "None",
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features_raw,
                },
                feature_importance=feature_importance,
                show_chart=False
            )

        except Exception as e:
            return render_template(
                "result.html",
                result=f"Error in Random Forest route: {e}",
                model_name="Random Forest",
                show_chart=False,
                class_label=None
            )

    return render_template("predict_rf.html")

@app.route("/predict_bagging", methods=["GET", "POST"])
def predict_bagging():
    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])

            # base tree hyperparams
            dt_max_depth_raw = request.form.get("dt_max_depth", "")
            dt_max_depth = int(dt_max_depth_raw) if dt_max_depth_raw else None
            dt_min_samples_split = int(request.form.get("dt_min_samples_split", 2))
            dt_min_samples_leaf = int(request.form.get("dt_min_samples_leaf", 1))

            # bagging hyperparams
            n_estimators = int(request.form.get("n_estimators", 50))
            max_samples = int(request.form.get("bag_max_samples", 100))
            bag_max_features = int(request.form.get("bag_max_features", 2))
            bootstrap_str = request.form.get("bootstrap", "True")
            bootstrap = True if bootstrap_str == "True" else False

            # train / test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=42
            )

            base_tree = DecisionTreeClassifier(
                criterion="gini",
                max_depth=dt_max_depth,
                min_samples_split=dt_min_samples_split,
                min_samples_leaf=dt_min_samples_leaf,
                random_state=42
            )

            bag = BaggingClassifier(
                estimator=base_tree,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=bag_max_features,
                bootstrap=bootstrap,
                random_state=42
            )
            bag.fit(X_train, y_train)

            # accuracy
            y_pred_test = bag.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)
            acc_percent = acc * 100.0
            accuracy_desc = describe_accuracy(acc_percent)

            # prediction
            sample = np.array([[study_hours, attendance]])
            user_pred = bag.predict(sample)[0]
            label = "Pass" if int(user_pred) == 1 else "Fail"

            # feature importance = average of trees (if available)
            feature_names = ["Study_Hours", "Attendance"]
            importances = None
            if hasattr(bag.estimators_[0], "feature_importances_"):
                all_imps = [est.feature_importances_ for est in bag.estimators_]
                importances = np.mean(all_imps, axis=0)

            feature_importance = []
            if importances is not None:
                feature_importance = [
                    {"name": feature_names[i], "value": float(importances[i])}
                    for i in range(len(importances))
                ]

            return render_template(
                "result.html",
                result=user_pred,
                class_label=label,
                model_name="Bagging (Decision Tree)",
                accuracy=round(acc_percent, 2),
                accuracy_desc=accuracy_desc,
                params={
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "bag_max_features": bag_max_features,
                    "bootstrap": bootstrap_str,
                    "dt_max_depth": dt_max_depth_raw or "None",
                    "dt_min_samples_split": dt_min_samples_split,
                    "dt_min_samples_leaf": dt_min_samples_leaf,
                },
                feature_importance=feature_importance if feature_importance else None,
                show_chart=False
            )

        except Exception as e:
            return render_template(
                "result.html",
                result=f"Error in Bagging route: {e}",
                model_name="Bagging",
                show_chart=False,
                class_label=None
            )

    return render_template("predict_bagging.html")

@app.route("/predict_boost", methods=["GET", "POST"])
def predict_boost():
    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])

            # base tree hyperparams
            dt_max_depth_raw = request.form.get("dt_max_depth", "6")
            dt_max_depth = int(dt_max_depth_raw) if dt_max_depth_raw else None
            dt_min_samples_split = int(request.form.get("dt_min_samples_split", 20))
            dt_min_samples_leaf = int(request.form.get("dt_min_samples_leaf", 10))

            # boosting hyperparams
            n_estimators = int(request.form.get("n_estimators", 100))
            learning_rate = float(request.form.get("learning_rate", 1.0))

            # train / test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=42
            )

            base_tree = DecisionTreeClassifier(
                criterion="gini",
                max_depth=dt_max_depth,
                max_features=None,  # keeping simple for 2-feature dataset
                min_samples_split=dt_min_samples_split,
                min_samples_leaf=dt_min_samples_leaf,
                random_state=42
            )

            boost = AdaBoostClassifier(
                estimator=base_tree,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            boost.fit(X_train, y_train)

            # accuracy
            y_pred_test = boost.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)
            acc_percent = acc * 100.0
            accuracy_desc = describe_accuracy(acc_percent)

            # prediction
            sample = np.array([[study_hours, attendance]])
            user_pred = boost.predict(sample)[0]
            label = "Pass" if int(user_pred) == 1 else "Fail"

            # feature importance approx by average of trees
            feature_names = ["Study_Hours", "Attendance"]
            importances = None
            if hasattr(boost.estimators_[0], "feature_importances_"):
                all_imps = [est.feature_importances_ for est in boost.estimators_]
                importances = np.mean(all_imps, axis=0)

            feature_importance = []
            if importances is not None:
                feature_importance = [
                    {"name": feature_names[i], "value": float(importances[i])}
                    for i in range(len(importances))
                ]

            return render_template(
                "result.html",
                result=user_pred,
                class_label=label,
                model_name="AdaBoost (Decision Tree)",
                accuracy=round(acc_percent, 2),
                accuracy_desc=accuracy_desc,
                params={
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "dt_max_depth": dt_max_depth_raw or "None",
                    "dt_min_samples_split": dt_min_samples_split,
                    "dt_min_samples_leaf": dt_min_samples_leaf,
                },
                feature_importance=feature_importance if feature_importance else None,
                show_chart=False
            )

        except Exception as e:
            return render_template(
                "result.html",
                result=f"Error in Boosting route: {e}",
                model_name="Boosting",
                show_chart=False,
                class_label=None
            )

    return render_template("predict_boost.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
