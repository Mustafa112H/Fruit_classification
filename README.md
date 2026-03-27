# 🧠 AI Project — Image Classification Comparative Study

## 📌 Overview
A machine learning project that compares multiple classification algorithms on an image dataset using both traditional and deep learning approaches.

### 🎯 Objective
Analyze trade-offs between:
- Accuracy  
- Interpretability  
- Computational efficiency  

---

## 🧪 Models Used
- Naive Bayes  
- Decision Tree  
- Feedforward Neural Network (MLP with CNN features)  

---

## 📊 Dataset
- **Classes:**
  - 🫐 Blueberries  
  - 🍌 Bananas  
  - 🍎 Pomegranates  
- **Total Images:** 1539  
- **Images per Class:** 513 (balanced)  
- **Image Size:** 64 × 64  
- **Format:** RGB  

---

## 🔧 Preprocessing
- Image resizing  
- Background removal (`rembg`)  
- Feature extraction  
- Stratified 5-fold cross-validation  

---

## 🤖 Model Details

### 🔹 Naive Bayes
- Fast and simple  
- Low memory usage  
- ❌ Weak for image data  

### 🔹 Decision Tree
- Easy to interpret  
- ❌ May overfit  

### 🔹 Neural Network (MLP + CNN Features)
- ✅ Best performance  
- ❌ Slower training  

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🏆 Results

| Model         | Performance | Speed     |
|--------------|------------|----------|
| Naive Bayes   | Low        | Very Fast |
| Decision Tree | Medium     | Fast      |
| MLP (CNN)     | High       | Slow      |

👉 **Best Model:** MLP + CNN Features  

---

## 🌐 Demo
🔗 https://mlclassification.streamlit.app/

### 🔑 Credentials
- Username: `admin`  
- Password: `1234`  

---

## 🛠️ Tech Stack
- Python  
- Scikit-learn  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
