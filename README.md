# 👁️ EyeX — AI-Powered Multi-Disease Retinal Screening System

<p align="center">
  <b>Deep Learning • Retinal Image Analysis • Explainable AI • Lesion Segmentation • Clinical Reporting</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-ff4b4b?logo=streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv" />
  <img src="https://img.shields.io/badge/U--Net-Lesion%20Segmentation-green" />
  <img src="https://img.shields.io/badge/Grad--CAM-Explainable%20AI-purple" />
</p>

---

## 📌 Overview

**EyeX** is an AI-assisted retinal screening and analysis system designed to analyze retinal fundus images using deep learning and computer vision techniques.

The system extends a conventional retinal disease classification pipeline into a broader screening workflow by combining:

- 🩺 Retinal disease classification
- 🔍 Image quality assessment
- 🧠 Explainable AI using Grad-CAM
- 🤖 Comparison of multiple deep learning architectures
- 🔬 Diabetic retinopathy lesion segmentation
- 📊 Model agreement analysis
- 📄 Automated clinical-style PDF reporting

EyeX is designed as a **research prototype and AI-assisted screening system**. It is not intended to replace ophthalmologists or provide an autonomous medical diagnosis.

---

# 🎯 Project Aim

The aim of EyeX is to develop a deep-learning-based retinal screening platform capable of extracting multiple forms of information from fundus images instead of producing only a single disease label.

The system combines classification, image analysis, explainability, segmentation, and reporting into a single workflow.

### Current disease classes

EyeX's primary classification model performs **single-label classification** into four categories:

1. Healthy
2. Diabetic Retinopathy (DR)
3. Glaucoma
4. Age-Related Macular Degeneration (AMD)

---

# ✨ Key Features

## 1. 🖼️ Fundus Image Upload

Users can upload a retinal fundus image through the Streamlit interface.

Supported image formats include:

- JPG
- JPEG
- PNG
- BMP
- TIFF

The application displays basic image information such as:

- File name
- Image format
- Resolution

---

## 2. 🔍 Image Quality Assessment

Before AI analysis, EyeX evaluates the quality of the uploaded retinal image.

The current quality assessment considers:

- **Sharpness**
- **Brightness**
- **Contrast**
- **Resolution**

The system produces an overall quality status such as:

- Good
- Fair
- Poor

This helps identify images that may not be suitable for reliable AI analysis.

> Image quality assessment is a preprocessing/screening aid and does not determine whether a patient has a retinal disease.

---

# 🧠 3. Retinal Disease Classification

The original EyeX classification pipeline uses:

### EfficientNet-B3

The EfficientNet-B3 model was adapted for four-class retinal disease classification.

### Classification output

For an uploaded image, the system provides:

- Predicted condition
- Model confidence
- Grad-CAM explanation

The existing EfficientNet-B3 model is preserved as the primary classification baseline.

---

# 🤖 4. Multi-Model Classification Comparison

EyeX also evaluates the uploaded image using three different deep learning architectures:

| Model | Architecture Type |
|---|---|
| EfficientNet-B3 | CNN |
| ConvNeXt-Tiny | Modern CNN |
| Swin-Tiny | Vision Transformer |

The comparison provides:

- Prediction from each model
- Current-image confidence
- Model agreement
- Final comparison prediction

This allows the system to examine whether different architectures produce consistent predictions for the same retinal image.

### Important distinction

The confidence shown for an individual uploaded image is **not the same as dataset-level model accuracy**.

For example:

```text
Current-image confidence:
Swin-Tiny → 100%

does NOT mean:

Swin-Tiny test accuracy → 100%
