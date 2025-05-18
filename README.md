#  Pneumonia Detection using CNN

##  Project Overview

This project uses a **Convolutional Neural Network (CNN)** to detect **pneumonia** from chest X-ray images. The model classifies images as either **PNEUMONIA** or **NORMAL**, aiming to assist early medical diagnosis using deep learning.

---

##  Dataset

The dataset used is publicly available on Kaggle:  
🔗 [Chest X-Ray Images (Pneumonia) | Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

It contains three folders:
- `train/`
- `val/`
- `test/`

Each of these contains:
- `PNEUMONIA/`
- `NORMAL/`

###  Dataset Summary

**Train:**  
- 1,341 NORMAL  
- 3,875 PNEUMONIA

**Validation:**  
- 8 NORMAL  
- 8 PNEUMONIA

**Test:**  
- 234 NORMAL  
- 390 PNEUMONIA

---

##  Notebook Structure

### 1. **Library Imports**
Libraries used: TensorFlow, Keras, OpenCV, Matplotlib, Seaborn, Scikit-learn.

### 2. **Image Preprocessing**
- Images are converted to grayscale
- Resized to 150x150
- Labeled as 0 (NORMAL) or 1 (PNEUMONIA)

### 3. **Dataset Preparation**
- Data split into training, validation, and test sets
- Normalization and reshaping of images

### 4. **Data Augmentation**
Using `ImageDataGenerator` to perform:
- Rotation
- Zoom
- Flipping
- Translation

### 5. **CNN Architecture**
The model includes:
- Conv2D, MaxPooling2D, BatchNormalization, Dropout
- Flatten and Dense layers
- Sigmoid activation in the final output for binary classification

### 6. **Compilation**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Callback: ReduceLROnPlateau (adaptive learning rate)

### 7. **Model Training**
- Trained on 12 epochs with real-time data augmentation

### 8. **Evaluation**
- Accuracy on test set
- Classification Report (precision, recall, F1-score)
- Confusion Matrix (visualized with a heatmap)

### 9. **Model Saving**
- Model saved in `.keras` format for reuse or deployment

---

##  Results

The final model successfully detects pneumonia from chest X-rays and serves as a potential tool for assisting diagnosis in medical contexts.

---

## Contact

For any questions or support, please contact Pneumodel team at(yousra1kh2@gmail.com)  
