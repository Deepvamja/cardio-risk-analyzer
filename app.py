from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd

app = Flask(__name__)

df = pd.read_csv("dataset/cardio_train.csv", sep=",")
df.columns = df.columns.str.lower()
df["age"] = df["age"] / 365

feature_cols = [
    'id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

X = df[feature_cols]
xmin = X.min()
xmax = X.max()

model_data = np.load("model/model.npy", allow_pickle=True)
W, b = model_data

def scale_input(values):
    return (values - xmin) / (xmax - xmin)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        id_val = float(request.form["id"])
        age = float(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = float(request.form["ap_hi"])
        ap_lo = float(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        features = np.array([
            id_val, age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active
        ])

        scaled = scale_input(features)

        prob = sigmoid(np.dot(W, scaled) + b)
        prediction = "High Risk" if prob >= 0.5 else "Low Risk"
        prob_percent = round(float(prob) * 100, 2)

        return render_template("result.html",
                               prediction=prediction,
                               probability=f"{prob_percent}%")

    except Exception as e:
        return f"Error: {e}"


@app.route("/dashboard")
def dashboard():
    charts = {
        "confusion": "charts/confusion_matrix.png",
        "roc": "charts/roc_curve.png",
        "pr": "charts/pr_curve.png",
        "acc": "charts/accuracy_curve.png",
        "loss": "charts/loss_curve.png"
    }

    
    summary = df[feature_cols].describe().round(2).to_html(
        classes="table table-bordered table-sm"
    )

    return render_template("dashboard.html", charts=charts, summary=summary)


def compute_accuracy():
   
    X = df[feature_cols].values
    y = df["cardio"].values

 
    X_scaled = (X - xmin.values) / (xmax.values - xmin.values)

    logits = np.dot(X_scaled, W) + b
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    accuracy = (preds == y).mean()
    return round(float(accuracy * 100), 2)


@app.route('/charts/<path:filename>')
def chart_files(filename):
    return send_from_directory("charts", filename)




if __name__ == "__main__":
     print("Model Accuracy:", compute_accuracy(), "%")
     app.run(host="0.0.0.0", port=5000)
