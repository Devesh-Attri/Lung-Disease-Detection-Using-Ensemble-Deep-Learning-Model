# 🫁 Lung Disease Detection Using Ensemble Deep Learning (EfficientNetB3 + DenseNet121)

This project implements an ensemble deep learning model using **EfficientNetB3** and **DenseNet121** to detect and classify lung diseases from **chest X-ray (CXR)** images. It focuses on five classes: **Bacterial Pneumonia**, **Viral Pneumonia**, **COVID-19**, **Tuberculosis**, and **Normal**. The model leverages transfer learning, data augmentation, and fine-tuning to achieve high classification accuracy, aiding in early diagnosis and treatment.

---

## 📌 Project Highlights

- 🧠 **Ensemble Model:** Combines EfficientNetB3 and DenseNet121 for robust feature extraction.
- 📊 **Multi-Class Classification:** 5 disease categories including Normal, Pneumonia variants, COVID-19, and TB.
- 🧪 **Accuracy:** Achieved **~90% test accuracy** and **F1-score of 0.8987**.
- 📉 **Training Techniques:** Uses **ReduceLROnPlateau**, **EarlyStopping**, and **ModelCheckpoint** for efficient training.
- 🧰 **Post-Training Evaluation:** Includes confusion matrix, classification report, and sample predictions.
- 🔬 **Interpretability:** Visualizations help analyze model behavior and misclassifications.

---

## 📁 Dataset

The dataset consists of **chest X-ray images** divided into the following five classes:
- Bacterial Pneumonia
- Viral Pneumonia
- COVID-19
- Tuberculosis
- Normal

**Data Split:**
- Training: 6,054 images  
- Validation: 2,016 images  
- Test: 2,025 images

> Data augmentation techniques such as rotation, zoom, shift, shear, and horizontal flip were applied to increase model generalization.

---

## 🏗️ Methodology

### 🔍 Model Architecture
- **EfficientNetB3** and **DenseNet121** as base models (pretrained on ImageNet).
- Initial layers frozen: 100 (EfficientNetB3), 200 (DenseNet121).
- Extracted features concatenated → Fully connected layers → Softmax output.

### ⚙️ Technical Setup
- Input Size: `224x224x3`
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
- Scheduler: `ReduceLROnPlateau`
- Callbacks: `EarlyStopping`, `ModelCheckpoint`
- Epochs: `50 (initial) + 20 (fine-tuning)`

### 💻 Hardware
- Training was conducted on a **GPU-enabled TensorFlow environment**.

---

## 📈 Results

| Metric        | Before Fine-Tuning | After Fine-Tuning |
|---------------|--------------------|-------------------|
| Accuracy      | 89.98%             | 89.43%            |
| Precision     | 89.84%             | 89.42%            |
| Recall        | 89.98%             | 89.43%            |
| F1 Score      | 89.87%             | 89.27%            |

- **Best performance** observed before full fine-tuning, indicating potential overfitting during extended training.

### 🔍 Confusion Matrix Insights
- **Strong performance** in classifying: COVID-19, Tuberculosis, Normal.
- **Misclassifications** observed between: Bacterial vs Viral Pneumonia — likely due to radiological similarity.

---

## 🧠 Key Learnings & Limitations

### ✔️ What Worked
- Efficient ensemble of CNNs improved classification performance.
- Data augmentation and freezing layers helped reduce overfitting.
- Visualization tools provided interpretability of results.

### ⚠️ Limitations & Future Work
- Misclassification between pneumonia types needs improvement.
- Incorporating **clinical metadata** (age, symptoms) may improve accuracy.
- Investigating **Vision Transformers (ViTs)** for fine-grained pattern detection.

---

## 🚀 Getting Started

### 🔧 Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- scikit-learn
- OpenCV (optional for visualization)

### 📦 Installation

```bash
git clone https://github.com/your-username/lung-disease-detection.git
cd lung-disease-detection
pip install -r requirements.txt
```
---

### 🏃‍♂️ Running the Model
```bash
python train_model.py   # Train the ensemble model
python evaluate.py      # Evaluate and visualize results
```
Ensure the dataset is placed in the correct directory format. Refer to config.py for file paths and model parameters.

---

### 📚 References
- EfficientNet: Rethinking Model Scaling for CNNs
- Densely Connected Convolutional Networks (DenseNet)
- COVIDx Dataset
