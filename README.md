<a id="readme-top"></a>

# Akal-Imitasi

<!-- PROJECT LOGO 
 -->
<img width="2510" alt="Ai-Ohto-NoBG" src="https://github.com/user-attachments/assets/952a044b-95b6-422b-871f-c8d98da94187" />


<br />
<div align="center">
  <h3 align="center">Machine Learning from Scratch</h3>

  <p align="center">
    Implementasi algoritma Machine Learning dari scratch untuk klasifikasi dataset mahasiswa
    <br />
    IF3170 - Inteligensi Artifisial
    <br />
    <a href="https://github.com/SayyakuHajime/Akal-Imitasi"><strong>Explore the repository »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SayyakuHajime/Akal-Imitasi/issues">Report Bug</a>
    ·
    <a href="https://github.com/SayyakuHajime/Akal-Imitasi/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#team">Team</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

Tugas Besar 2 IF3170 Inteligensi Artifisial - Implementasi dari scratch tiga algoritma Machine Learning klasik untuk klasifikasi **Student Dropout Prediction**:

### Algoritma yang Diimplementasikan
- **Decision Tree Learning**: ID3, C4.5, CART
- **Logistic Regression**: Multinomial dengan Stochastic Gradient Descent
- **Support Vector Machine**: One-vs-All (OvA), One-vs-One (OvO), DAGSVM


Proyek ini mencakup analisis komprehensif dari preprocessing, feature engineering, hyperparameter tuning, hingga evaluasi performa model. Detail lengkap tersedia di laporan.

### Built With

* Python 3.12
* NumPy 2.3.5
* Pandas 2.3.3
* Matplotlib 3.10.7 & Seaborn 0.13.2
* Scikit-learn 1.7.2
* Jupyter Notebook

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Getting Started

### Prerequisites

- Python 3.8 atau lebih tinggi
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/SayyakuHajime/Akal-Imitasi.git
cd Akal-Imitasi

# Install dependencies
pip install -r requirements.txt
```

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Usage

### Menjalankan Notebook Utama

```bash
jupyter notebook Tubes2_AI_kelompok5_notebook_BEST.ipynb
```

Notebook ini berisi:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Training & Evaluation semua 7 model
- Perbandingan performa custom vs sklearn
- Visualisasi hasil

### Struktur Kode

```python
# Import models dari scratch
from src.dtl import ID3Classifier, C45Classifier, CARTClassifier
from src.logistic_regression import LogisticRegressionMultinomial
from src.svm import SVMOneVsAll, SVMOneVsOne, SVMDAG

# Contoh penggunaan
model = C45Classifier(max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>


<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Project Structure

<details>
  <summary>Click to expand tree structure</summary>

```
Akal-Imitasi/
├── data/
│   ├── train.csv                  # Training data (3096 samples)
│   ├── test.csv                   # Test data (1328 samples)
│   └── submissions/               # Generated submission files
├── src/                           # From-scratch implementations
│   ├── __init__.py
│   ├── dtl.py                     # Decision Tree Learning (ID3, C4.5, CART)
│   ├── logistic_regression.py    # Logistic Regression (SGD)
│   └── svm.py                     # SVM (OvA, OvO, DAGSVM)
├── Tubes2_AI_kelompok5_notebook_BEST.ipynb  # Main notebook
├── feature_engineering.py         # Feature engineering functions
├── requirements.txt
└── README.md
```

</details>

### Key Files

- **`Tubes2_AI_kelompok5_notebook_BEST.ipynb`**: Main notebook dengan semua eksperimen
- **`src/dtl.py`**: Decision Tree implementations (ID3, C4.5, CART)
- **`src/logistic_regression.py`**: Multinomial Logistic Regression
- **`src/svm.py`**: SVM dengan 3 multiclass strategies
- **`feature_engineering.py`**: Custom feature engineering pipeline

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Team

**Kode Kelompok:** 5

| Nama | NIM | Pembagian Tugas |
|------|-----|-----------------|
| M Hazim R. Prajoda | 13523009 | Implementasi DTL scratch, LogReg scratch, SVM scratch, Feature Engineering |
| Orvin Andika Ikhsan A | 13523017 | Laporan, Tugas pendukung |
| Fajar Kurniawan | 13523027 | Laporan Decision Tree Learning, Laporan Cleaning and Preprocessing |
| Darrel Adinarya Sunanda | 13523061 | SVM Model, Laporan |
| Reza Ahmad Syarif | 13523119 | Preprocessing (Handling Class Imbalance dan Feature Selection), Bonus Tree Visualizer, Laporan |

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Acknowledgments

### Dataset
- **Kaggle Competition**: [Student Dropout Prediction](https://www.kaggle.com/competitions/if3170-2024-large-homework-2)

### Key Techniques
- **SMOTE**: Synthetic Minority Over-sampling Technique untuk class balancing
- **RobustScaler**: IQR-based scaling resistant to outliers
- **Filter-based Feature Selection**: Chi-Square, ANOVA F-Score, Mutual Information
- **Consensus Voting**: Ensemble approach untuk feature selection

### References

**Decision Tree Learning:**
- [ID3 Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/iterative-dichotomiser-3-id3-algorithm-from-scratch/)
- [C4.5 Algorithm - Wikipedia](https://en.wikipedia.org/wiki/C4.5_algorithm)
- [CART - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/tree.html)

**Support Vector Machine:**
- [SVM Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)
- [DAGSVM - Machine Learning Mastery](https://machinelearningmastery.com/)

**Logistic Regression:**
- [Logistic Regression - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)
- [Softmax Function - Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)

**Feature Engineering & SMOTE:**
- [SMOTE - imbalanced-learn Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Feature Engineering Guide - Kaggle](https://www.kaggle.com/learn/feature-engineering)

---

<div align="center">
  <p>
    <strong>IF3170 - Inteligensi Artifisial</strong><br>
    Sekolah Teknik Elektro dan Informatika<br>
    Institut Teknologi Bandung<br>
    2025
  </p>
</div>
