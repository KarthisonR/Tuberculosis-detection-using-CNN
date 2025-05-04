# 🩺 Tuberculosis Detection Using CNN

This project uses a Convolutional Neural Network (CNN) to detect Tuberculosis (TB) from chest X-ray images. The model is trained to differentiate TB-positive and TB-negative patients, aiming to assist in medical diagnosis with high precision and accuracy.

---

## 📁 Repository Contents

- `TB detection notebook.ipynb` – Jupyter Notebook for model training, evaluation, and visualization.
- `final.py` – Full pipeline script with model training, prediction, database storage, and a Streamlit-based frontend.
- `README.md` – Project overview and instructions.
- `LICENSE` – MIT License.

---

## 📊 Dataset Overview

The dataset consists of chest X-ray images categorized into **TB-positive** and **TB-negative** folders. The images are preprocessed (resized, normalized) for consistent training performance.

> *Make sure the dataset structure aligns with Keras image generator requirements (e.g., `train/TB`, `train/Normal`).*

---

## 🧠 About the Model

The CNN architecture consists of:

- **Convolutional Layers**: Extract spatial patterns from X-ray images
- **ReLU Activation**: Introduces non-linearity
- **MaxPooling**: Downsamples feature maps to reduce computation
- **Dropout**: Prevents overfitting
- **Flatten & Dense Layers**: For final decision-making
- **Sigmoid Output Layer**: Binary classification for TB presence

---

## 🔍 Observations

### 1. Accuracy
- **Training Accuracy** steadily increases and reaches nearly **1.00**.
- **Validation Accuracy** also increases and stabilizes around **0.98**.
- ✅ This is a **very strong performance** with minimal gap between training and validation — indicating **good generalization**.

### 2. Loss
- Both **training and validation loss** consistently **decrease**, although the validation loss becomes a bit **noisy** after epoch 6–7.
- 📉 No clear sign of **overfitting** — validation loss remains stable.

### 3. Precision & Recall
- Both **precision and recall** curves for training and validation mirror accuracy trends:
  - Values near **1.00**
  - Slight validation fluctuations are normal
- ✅ Indicates **balanced model performance** across classes.

---

## 🧰 Features in `final.py`

- 🧠 **Model Training**: Defines and trains the CNN model on X-ray images.
- 💾 **Model Saving**: Saves the trained model as `model.h5`.
- 📈 **Evaluation**: Prints accuracy, precision, recall, confusion matrix.
- 🧪 **Single Image Prediction**: Predicts TB presence from user-uploaded X-rays.
- 🧾 **SQLite Patient Database**:
  - Stores name, age, gender, image path, and TB result.
  - Ensures history is saved and reviewable.
- 🌐 **Streamlit Frontend**:
  - Upload X-ray
  - Enter patient details
  - View prediction result live
  - Automatically logs into local database

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KarthisonR/Tuberculosis-detection-using-CNN.git
cd Tuberculosis-detection-using-CNN
