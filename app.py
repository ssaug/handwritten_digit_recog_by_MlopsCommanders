from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
##import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the saved ensemble model
model_path = "./model/ensemble_model.pkl"
with open(model_path, "rb") as f:
    ensemble_model = pickle.load(f)

# Extract individual models
best_rf = ensemble_model["RandomForest"]
best_svm = ensemble_model["SVM"]
best_dt = ensemble_model["DecisionTree"]

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/best_model_parameters", methods=["GET"])
def get_best_model_parameters():
    best_params = {
        "RandomForest": best_rf.get_params(),
        "SVM": best_svm.get_params(),
        "DecisionTree": best_dt.get_params()
    }
    return jsonify(best_params)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]  # Expecting a list of feature vectors
    X_input = np.array(data)

    # Reshape the flattened array back to 28x28 for visualization
    #sample_image = X_input.reshape(28, 28)

    # Display the image
    #plt.imshow(sample_image, cmap="gray")
    #plt.title("Received MNIST Image for prediction ")
    #plt.axis("off")
    #plt.show()

    # Get predictions from each model
    rf_pred = best_rf.predict_proba(X_input)
    svm_pred = best_svm.predict_proba(X_input)
    dt_pred = best_dt.predict_proba(X_input)

    # Compute ensemble prediction
    ensemble_pred = (rf_pred + svm_pred + dt_pred) / 3
    y_pred = np.argmax(ensemble_pred, axis=1)

    return jsonify({"predictions": y_pred.tolist()})


@app.route("/train", methods=["POST"])
def train():
    data = request.json
    X_train = np.array(data["features"])
    y_train = np.array(data["labels"])

    # Retrain models
    best_rf.fit(X_train, y_train)
    best_svm.fit(X_train, y_train)
    best_dt.fit(X_train, y_train)

    # Save the updated model
    updated_model = {"RandomForest": best_rf, "SVM": best_svm, "DecisionTree": best_dt}
    with open(model_path, "wb") as f:
        pickle.dump(updated_model, f)

    return jsonify({"message": "Model retrained successfully and saved."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)