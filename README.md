Handwritten Digit Recognition Using Ensemble Method

Project Overview

This project is an implementation of a Handwritten Digit Recognition System using an Ensemble Learning Approach. The system uses multiple machine learning models (Random Forest, SVM, Decision Tree) to classify handwritten digits (0-9). It exposes a REST API for making predictions, retrieving model parameters, and retraining the model.

Features
      
      ✅ Digit Prediction - Predicts handwritten digits from preprocessed images.
      
      ✅ Model Parameter Retrieval - Retrieve hyperparameters of trained models.
      
      ✅ Model Training - Allows retraining with new data.
      
      ✅ REST API Integration - Easily integrate with applications via API.

Tech Stack-

      1.Python 🐍
      
      2.Flask 🌐 (for API development)
      
      3.Scikit-Learn 📊 (for ML models)
      
      4.Pandas & NumPy 🔢 (for data handling)
      
      5.OpenCV & Pillow 🖼️ (for image preprocessing)
      
      6.MLflow 📈 (for experiment tracking)

Installation & Setup

    1️⃣ Clone the Repository

              git clone https://github.com/yourusername/handwritten-digit-recognition.git
              cd handwritten-digit-recognition

    2️⃣ Install Dependencies

              pip install -r requirements.txt

    3️⃣ Run the Flask API

              python flask_mnist_api.py

              The API will start at http://localhost:5000
              
              API Usage

    1️⃣ Get Model Parameters

              Request:

              GET http://localhost:5000/best_model_parameters
              
              Response:
              
              {
                  "RandomForest": { "n_estimators": 100, "max_depth": 10 },
                  "SVM": { "C": 1.0, "kernel": "linear" },
                  "DecisionTree": { "max_depth": 10 }
              }

      2️⃣ Predict a Handwritten Digit

              Request:
              
              POST http://localhost:5000/predict
              Content-Type: application/json
              
              {
                  "features": [[0.0, 0.1, 0.2, ..., 0.9, 1.0, 0.0]]
              }
              
              Response:
              
              {
                  "predictions": [2]
              }

        3️⃣ Train the Model

              Request:
              
              POST http://localhost:5000/train
              Content-Type: application/json
              
              {
                  "features": [
                      [0.1, 0.2, ..., 0.9],
                      [0.5, 0.6, ..., 0.3]
                  ],
                  "labels": [5, 3]
              }
              
              Response:
              
              {
                  "message": "Model retrained successfully and saved."
              }
              
              [Results_experiments_screenshots.docx](https://github.com/user-attachments/files/18733138/Results_experiments_screenshots.docx)
