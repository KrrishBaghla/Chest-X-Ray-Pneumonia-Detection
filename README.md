#  Chest X-Ray Pneumonia Detection using Deep Learning

A deep learning web application that detects **pneumonia from chest X-ray images** using a trained CNN model and an interactive **Streamlit interface**.

This project demonstrates the **complete ML pipeline**:
data → training → evaluation → deployment.

---

## 🚀 Live Demo

https://pneumonia--detector.streamlit.app/

---

## 📌 Problem Statement

Pneumonia is a serious lung infection that must be detected **early and accurately**.
Manual diagnosis from X-ray images can be **time-consuming and error-prone**.

This project uses **Deep Learning** to automatically classify:

* **Normal**
* **Pneumonia**

from chest X-ray images.

---

## 🧠 Model & Approach

* Convolutional Neural Network (CNN)
* Image preprocessing & normalization
* Training using labeled chest X-ray dataset
* Performance evaluation using:

  * Accuracy
  * Confusion Matrix
  * Loss curves

The trained model is saved and integrated into a **real-time prediction web app**.

---

## 🖥️ Web Application

Built using **Streamlit** to allow users to:

* Upload chest X-ray image
* Run model prediction instantly
* View classification result in a clean UI

This converts a **research notebook → usable AI product**.

---

## 📂 Project Structure

```
chest-xray-project/
│
├── model/
│   └── best_model.h5
├── app.py
├── notebook.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Run Locally

### 1. Clone repository

```
git clone https://github.com/KrrishBaghla/Chest-X-Ray-Pneumonia-Detection.git
cd chest-xray-project
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run Streamlit app

```
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 📊 Results

*(Add your accuracy, graphs, or screenshots here)*

Example:

* Training Accuracy: **~95%**
* Validation Accuracy: **~92%**

---

## 🎯 Skills Demonstrated

* Deep Learning with **TensorFlow / Keras**
* Medical image classification
* Model evaluation & saving
* **End-to-end ML deployment**
* Streamlit web app development
* Clean project structuring & GitHub usage

---

## 👨‍💻 Author

**Krrish**
B.Tech Mathematics & Computing
NIT Jalandhar

Interested in **AI, Space Tech, and Real-world ML systems**.

---

## ⭐ If you like this project

Give it a **star on GitHub** — it motivates further research-level builds.
