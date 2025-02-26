import numpy as np
import struct
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle
from scipy.stats import mode
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow experiment
mlflow.set_experiment("MNIST_Ensemble_Classification")

# Load and preprocess the MNIST dataset
def load_mnist_images(filename):
    try:
        with open(filename, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        logger.info(f"Loaded MNIST images from {filename}")
        return data
    except Exception as e:
        logger.error(f"Failed to load MNIST images from {filename}: {e}")
        raise

def load_mnist_labels(filename):
    try:
        with open(filename, 'rb') as f:
            _, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        logger.info(f"Loaded MNIST labels from {filename}")
        return labels
    except Exception as e:
        logger.error(f"Failed to load MNIST labels from {filename}: {e}")
        raise

# File paths
train_images_path = "data/train-images-idx3-ubyte/train-images.idx3-ubyte"
train_labels_path = "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
test_images_path = "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
test_labels_path = "data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"

try:
    # Load data
    X_train = load_mnist_images(train_images_path).reshape(60000, -1) / 255.0
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path).reshape(10000, -1) / 255.0
    y_test = load_mnist_labels(test_labels_path)
    
    # Reduce dataset size for efficient tuning
    X_train_small, y_train_small = X_train[:10000], y_train[:10000]
    
    # Hyperparameter tuning configurations
    rf_params = {"n_estimators": [50, 100, 150], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10]}
    svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    dt_params = {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]}
    
    # Initialize classifiers
    rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    svm_search = RandomizedSearchCV(SVC(probability=True, random_state=42), svm_params, n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    dt_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), dt_params, n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train models with best hyperparameters
        rf_search.fit(X_train_small, y_train_small)
        svm_search.fit(X_train_small, y_train_small)
        dt_search.fit(X_train_small, y_train_small)
    
        # Get best models
        best_rf = rf_search.best_estimator_
        best_svm = svm_search.best_estimator_
        best_dt = dt_search.best_estimator_
    
        # Log best parameters
        mlflow.log_params({"rf_best_params": rf_search.best_params_})
        mlflow.log_params({"svm_best_params": svm_search.best_params_})
        mlflow.log_params({"dt_best_params": dt_search.best_params_})
    
        # Make predictions and generate labels
        rf_pred = best_rf.predict(X_test)
        svm_pred = best_svm.predict(X_test)
        dt_pred = best_dt.predict(X_test)
    
        # Average predictions for ensemble
        # Stack predictions into a 2D array
        predictions = np.vstack((rf_pred, svm_pred, dt_pred)).T  # Shape: (n_samples, n_classifiers)
    
        # Apply majority voting
        y_pred_final = mode(predictions, axis=1).mode.flatten()
    
        # Evaluate performance
        final_accuracy = accuracy_score(y_test, y_pred_final)
        mlflow.log_metric("accuracy", final_accuracy)
    
        # Save models with input examples
        input_example = X_test[:5]
        mlflow.sklearn.log_model(best_rf, "RandomForest", input_example=input_example)
        mlflow.sklearn.log_model(best_svm, "SVM", input_example=input_example)
        mlflow.sklearn.log_model(best_dt, "DecisionTree", input_example=input_example)
    
    # Save the ensemble model as a pickle file
    ensemble_model = {"RandomForest": best_rf, "SVM": best_svm, "DecisionTree": best_dt}
    
    # Save to file
    with open("./model/ensemble_model.pkl", "wb") as f:
        pickle.dump(ensemble_model, f)
    logger.info("Final accuracy: %f", final_accuracy)
    logger.info("Ensemble model saved to: ./model/ensemble_model.pkl")
    logger.info(y_test)
except Exception as e:
    logger.error(f"An error occurred during the execution: {e}")
