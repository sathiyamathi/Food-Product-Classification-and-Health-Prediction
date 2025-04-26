# Food-Product-Classification-and-Health-Prediction
This project classifies packaged food products using CNN (InceptionResNet V2) and predicts their healthiness with an ensemble of SVM, Random Forest, and XGBoost models. It uses image and nutritional data, evaluated with Accuracy, Precision, Recall, and F1 Score.
Overview
This project focuses on classifying packaged food products and predicting their healthiness using a combination of image data and nutritional data.

We have designed a dual-path architecture:

For Image Data, a Convolutional Neural Network (CNN) using InceptionResNet V2 is trained to classify food products like Maggi, Lays, Chips, and Oats.

For Nutritional Data, machine learning models such as SVM, Random Forest, and XGBoost are trained and combined using a Voting Classifier to predict if a food product is Healthy or Unhealthy.

Workflow
Load and Preprocess Dataset
Both image and nutritional datasets are loaded and preprocessed for model training.

Dataset Type Decision
Based on the type (Image or Nutritional data), the pipeline branches into two paths.

Model Training

Image Data: Train a CNN using InceptionResNet V2 to classify food products.

Nutritional Data: Train SVM, Random Forest, and XGBoost models and combine predictions using a Voting Classifier for health classification.

Model Evaluation
Both models are evaluated using Accuracy, Precision, Recall, and F1 Score.

Prediction

Product Classification: Use the trained CNN to classify the food product.

Health Prediction: Use the ensemble model to predict whether the food product is healthy or unhealthy.

Technologies Used
Python

TensorFlow / Keras

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

Performance Metrics
Accuracy

Precision

Recall

F1 Score
