# Information Extraction for Facial Skincare Products

> An NLP pipeline for extracting structured information about active ingredients in Vietnamese skincare product descriptions — built as a course project at the University of Economics Ho Chi Minh City (UEH).

---

## Overview

This project implements an end-to-end **Information Extraction (IE)** system for Vietnamese text, targeting product descriptions and reviews of facial skincare products (cleansers, toners, serums, moisturizers, etc.).

The system identifies key entities and their relationships, enabling automated extraction of structured knowledge from unstructured Vietnamese text.

**Course:** Natural Language Processing (25C1INF50907602)  
**Instructor:** Dr. Đặng Ngọc Hoàng Thành  
**Institution:** School of Technology & Design, UEH University  
**Date:** December 2025

---

## Objectives

- Build a labeled Vietnamese dataset of skincare ingredient descriptions
- Perform **Named Entity Recognition (NER)** to identify entities:
  - `NAME` — common name of the ingredient
  - `INCI` — international nomenclature (INCI name)
  - `BENEFITS` — benefits of the ingredient
  - `SKIN_CONCERNS` — skin concerns it addresses
  - `ORIGIN` — source or origin of the ingredient
- Perform **Relation Extraction (RE)** to identify relationships:
  - `has_inci_name`
  - `has_benefits`
  - `targets_skin_concerns`
  - `has_origin`
- Compare multiple ML/DL models for the RE task
- Deploy a full-stack application for annotation and inference

---

## Dataset

- **Source:** Web-crawled descriptions and articles about 40+ commonly used cosmetic active ingredients
- **Size:** 1,182 sentences
- **Annotation tool:** [Label Studio](https://labelstud.io/)
- **Storage:** MongoDB
- **NER entities labeled:** 4,037 total
- **RE pairs:** 2,690 entity pairs with relationships

| Entity Type   | Count | Percentage |
|---------------|-------|------------|
| BENEFITS      | 1,622 | 40.18%     |
| NAME          | 1,109 | 27.47%     |
| SKIN_CONCERNS | 590   | 14.61%     |
| INCI          | 370   | 9.17%      |
| ORIGIN        | 346   | 8.57%      |

| Relation Type         | Count | Percentage |
|-----------------------|-------|------------|
| has_benefits          | 1,586 | 59.18%     |
| targets_skin_concerns | 565   | 21.08%     |
| has_origin            | 346   | 12.91%     |
| has_inci_name         | 183   | 6.83%      |

---

## System Architecture

The system follows a three-layer architecture:

```
User / Browser
      │ HTTP
      ▼
  Frontend (Next.js)
      │ REST API
      ▼
  Backend (FastAPI)
   ┌───┴────────────────┐
   ▼                    ▼
MongoDB              MLflow
(Labeled Data)   (Experiment Tracking)
                        │ Store artifacts
                        ▼
                      MinIO
                 (Models & Artifacts)
```

---

## ⚙️ Pipeline

```
Crawl / Collect Raw Text
        │
Preprocess (clean, split, normalize)
        │
Label Studio (NER + RE Annotation)
        │
MongoDB (Labeled Data Storage)
   ┌────┴────┐
   ▼         ▼
Build NER  Build RE
Dataset    Dataset
   │             │
   ▼             ▼
CRF Model   PhoBERT Vectorization
(NER)            │
           ┌─────┼──────┬────────┐
           ▼     ▼      ▼        ▼
          SVM   LR    RF        MLP
                │
           MLflow + MinIO (tracking & storage)
                │
         Application Layer (NextJS + FastAPI)
```

---

## Models

### NER — Conditional Random Field (CRF)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.88  |
| Precision | 0.78  |
| Recall    | 0.76  |
| F1-score  | 0.77  |

### RE — Feature Extraction

All RE models use **PhoBERT** (`vinai/phobert-base-v2`) as a frozen feature extractor. The feature vector is 3,840-dimensional, combining:
- Sentence-level mean pooling
- Head entity representation
- Tail entity representation
- Absolute difference of entity vectors
- Element-wise product of entity vectors

### RE — Model Comparison

| Model               | Accuracy | Macro F1 | Weighted F1 | Positive-only F1 |
|---------------------|----------|----------|-------------|------------------|
| **SVM** ⭐           | **0.89** | **0.89** | **0.89**    | **0.75**         |
| Logistic Regression | 0.87     | 0.85     | 0.87        | 0.72             |
| MLP                 | 0.84     | 0.80     | 0.84        | 0.67             |
| Random Forest       | 0.79     | 0.76     | 0.80        | 0.70             |

> **SVM** (LinearSVC with balanced class weights) achieved the best overall performance across all metrics.

---

## 🛠️ Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.8+                         |
| NER Model       | CRF (`python-crfsuite`)             |
| RE Encoder      | PhoBERT (`vinai/phobert-base-v2`)   |
| RE Classifiers  | scikit-learn, PyTorch               |
| Data Storage    | MongoDB                             |
| Annotation      | Label Studio                        |
| Experiment Tracking | MLflow                          |
| Artifact Storage | MinIO                              |
| Backend API     | FastAPI                             |
| Frontend        | Next.js                             |
| Dev Tools       | Jupyter Notebook, Google Colab, Git |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/NguyenPhanNhatLan/NLP_IE_N.git
cd NLP_IE_N

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Copy and fill in .env files for both backend and UI

# 5. Run the backend
uvicorn main.app:app --reload

# 6. Run the frontend
npm run dev
```

---

## Project Structure

```
NLP_IE_N/
├── backend/
│   ├── main/
│   │   └── app.py              # FastAPI application entry point
│   ├── ner/                    # NER model training & inference
│   ├── re/                     # RE model training & inference
│   │   ├── vectorize.py        # PhoBERT-based feature extraction
│   │   ├── train_svm.py
│   │   ├── train_lr.py
│   │   ├── train_rf.py
│   │   └── train_mlp.py
│   ├── data/                   # Dataset building scripts
│   └── requirements.txt
├── frontend/                   # Next.js UI
├── docker-compose.yml
└── README.md
```

---

## Team

| Name                   | Student ID   | Contribution                              |
|------------------------|--------------|-------------------------------------------|
| Nguyễn Phan Nhật Lan   | 31231025473  | NER model training, System design & deployment |
| Hồ Ngọc Như Quỳnh      | 31231021273  | Dataset construction, Vectorization       |
| Nguyễn Thị Minh Thư    | 31231025328  | Evaluation methodology, RE (SVM)          |
| Lê Thủy Tiên           | 31231020076  | RE (Random Forest, Logistic Regression, MLP) |

> All team members contributed to data crawling, manual annotation, and report writing.

---

## References

1. Ramchoun et al. (2016). *Multilayer perceptron: Architecture optimization and training.*
2. Rosenblatt (1958). *The Perceptron: A Theory of Statistical Separability in Cognitive Systems.*
3. GeeksforGeeks. *Support Vector Machine (SVM) Algorithm.*
4. Jaiswal, S. (2024). *Multilayer Perceptrons in Machine Learning.* DataCamp.
5. GeeksforGeeks. *Logistic Regression in Machine Learning.*
6. AIcandy. *Random Forest Algorithm: Detailed Explanation and Applications.*
7. GeeksforGeeks. (2024). *Information Extraction in NLP.*
