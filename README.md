=========================
🧠 AI PROJECT README
=========================
🧠 Image Classification Comparative Study

A comprehensive machine learning project that compares multiple classification algorithms on an image dataset using both traditional and deep learning approaches.

🚀 Overview

This project evaluates and compares the performance of three machine learning models for image classification:

Naive Bayes
Decision Tree
Feedforward Neural Network (MLP with CNN features)

The objective is to analyze trade-offs between accuracy, interpretability, and computational efficiency.

📊 Dataset
Classes:
🫐 Blueberries
🍌 Bananas
🍎 Pomegranates
Total images: 1539
Images per class: 513 (balanced)
Image size: 64 × 64
Format: RGB
🔧 Preprocessing
Image resizing
Background removal (rembg)
Feature extraction
Stratified 5-fold cross-validation
🤖 Models
Naive Bayes
Fast, simple, low memory
Weak for image data
Decision Tree
Interpretable
May overfit
Neural Network (MLP + CNN Features)
Best performance
Slower training
📈 Metrics
Accuracy
Precision
Recall
F1-score
Confusion matrix
🏆 Results
Model	Performance	Speed
Naive Bayes	Low	Very Fast
Decision Tree	Medium	Fast
MLP (CNN)	High	Slow

👉 Best Model: MLP + CNN Features

🌐 Demo

https://mlclassification.streamlit.app/

Username: admin
Password: 1234

🛠️ Tech Stack
Python
Scikit-learn
TensorFlow / Keras
OpenCV
Streamlit
👨‍💻 Authors
Mohammad Omar
Heba Mustafa
