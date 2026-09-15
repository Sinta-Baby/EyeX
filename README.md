# 👁️ RetinaSense

**A Deep Learning-Based Multi-Disease Retinal Disease Classification System**

RetinaSense is a deep learning project designed to automatically classify retinal fundus images into multiple eye disease categories using state-of-the-art convolutional neural networks. The project focuses on building an accurate, scalable, and explainable retinal disease screening system.

---

## 🎯 Project Objectives

- Detect multiple retinal diseases from fundus images
- Perform automated retinal disease classification
- Build a robust preprocessing pipeline
- Train a deep learning model using EfficientNet-B3
- Provide explainable AI predictions using Grad-CAM
- Deploy the model through a Streamlit web application

---

## 🩺 Supported Disease Classes

- ✅ Healthy Retina
- ✅ Diabetic Retinopathy (DR)
- ✅ Glaucoma
- ✅ Age-related Macular Degeneration (AMD)

---

## 📂 Dataset

The RetinaSense dataset is built by combining multiple publicly available retinal image datasets:

- ODIR
- APTOS 2019 Blindness Detection
- ARMD Dataset
- Glaucoma Dataset

After preprocessing:

| Split | Images |
|-------|-------:|
| Training | 8,971 |
| Validation | 1,121 |
| Testing | 1,122 |
| **Total** | **11,214** |

---

## 🚀 Features

### ✅ Completed

- Unified dataset preprocessing pipeline
- Metadata generation
- Dataset cleaning
- Train/Validation/Test splitting
- Duplicate filename handling
- Dataset verification

### 🚧 In Progress

- EfficientNet-B3 training
- Model evaluation
- Performance visualization

### 🔮 Planned

- Grad-CAM Explainability
- Streamlit Web Application
- PDF Medical Report Generation
- Prediction Confidence Scores

---

## 🛠️ Technology Stack

- Python
- PyTorch
- OpenCV
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit (Planned)

---

## 📁 Project Structure

```text
RetinaSense/
│
├── core/
├── datasets/
├── models/
├── scripts/
├── utils/
├── checkpoints/
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Current Progress

- ✅ Dataset Collection
- ✅ Dataset Preprocessing
- ✅ Metadata Generation
- ✅ Dataset Splitting
- ✅ Image Copying
- ✅ Dataset Verification
- 🚧 Model Training
- ⏳ Evaluation
- ⏳ Deployment

---
## Installation

Clone the repository

```bash
git clone https://github.com/Sinta-Baby/RetinaSense.git
cd RetinaSense
```

Create a virtual environment

```bash
python -m venv renv
```

Windows

```bash
renv\Scripts\activate
```

Linux / macOS

```bash
source renv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the required retinal datasets and place them in:

```
datasets/raw/
```

Then run:

```bash
python -m scripts.build_dataset
python -m scripts.split_dataset
python -m scripts.copy_images
```

The final dataset will contain:

| Split | Images |
|------|-------:|
| Train | 8971 |
| Validation | 1121 |
| Test | 1122 |
| Total | 11214 |

Verify the dataset:

```bash
python -m scripts.dataset_statistics
```
## 👩‍💻 Developer

**Sinta Baby**

Integrated M.Sc. Computer Science (Data Science)  
Nirmala College, Muvattupuzha  
Mahatma Gandhi University, Kerala, India

---

## 📄 License

This project is developed for academic and research purposes.