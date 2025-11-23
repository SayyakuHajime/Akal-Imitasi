<a id="readme-top"></a>

# Akal-Imitasi

<!-- PROJECT LOGO -->
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
    <li><a href="#features">Features</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#team">Team</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

Tugas Besar 2 IF3170 Inteligensi Artifisial - Implementasi dari scratch tiga algoritma Machine Learning klasik:
- **Decision Tree Learning** (ID3, C4.5, CART)
- **Logistic Regression** (Stochastic Gradient Ascent)
- **Support Vector Machine** (One-vs-All, One-vs-One, DAGSVM)

Proyek ini mencakup perbandingan antara implementasi from-scratch dengan library scikit-learn, analisis dataset mahasiswa, dan submisi ke kompetisi Kaggle.

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
- virtualenv (optional tapi recommended)

### Installation 

**Option 1: Automatic Setup (Recommended)**
```bash
# Clone repository
git clone https://github.com/SayyakuHajime/Akal-Imitasi.git
cd Akal-Imitasi

# Run setup script
./setup_env.sh

# Activate environment
source .venv/bin/activate
```

**Option 2: Manual Setup**
```bash
# Clone repository
git clone https://github.com/SayyakuHajime/Akal-Imitasi.git
cd Akal-Imitasi

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verifikasi Instalasi
```bash
# Check if packages installed correctly
python -c "import numpy, pandas, sklearn; print('All packages installed!')"
```

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Usage

### Menjalankan Notebook

1. **Activate environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Run Jupyter Notebook**
   ```bash
   jupyter notebook Tubes2_AI_kelompok5_notebook.ipynb
   ```

3. **Atau jalankan dari VS Code**
   - Buka `Tubes2_AI_kelompok5_notebook.ipynb`
   - Select kernel: `.venv/bin/python`
   - Run cells

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Features

...

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Project Structure

<details>
  <summary>Tree Structure:</summary>

```
Akal-Imitasi/
├── data/                          # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/                           # Source code (from-scratch implementations)
│   ├── __init__.py
│   ├── dtl.py                     # ID3, C4.5, CART
│   ├── logistic_regression.py    # Logistic Regression (SGA)
│   └── svm.py                     # SVM (OvA, OvO, DAGSVM)
├── doc/                           # Laporan PDF
├── Tubes2_AI_kelompok5_notebook.ipynb  # Main notebook
├── requirements.txt               # Python dependencies
├── setup_env.sh                   # Setup script
└── README.md
```

</details>

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## Team

**Kode Kelompok:** 5

| Nama | NIM |
|------|-----|
| - | - |
| - | - |
| - | - |
| - | - |
| - | - |

<div align="right">

[&nwarr; Back to top](#readme-top)

</div>

## License

Proyek ini saat ini belum memiliki lisensi. Informasi lisensi akan ditambahkan dalam pembaruan mendatang.

## Acknowledgments

### References:

- [ID3 Algorithm](https://www.geeksforgeeks.org/machine-learning/iterative-dichotomiser-3-id3-algorithm-from-scratch/)
- [CART Algorithm](https://www.geeksforgeeks.org/machine-learning/cart-classification-and-regression-tree-in-machine-learning/)
- [Decision Tree Algorithms](https://www.geeksforgeeks.org/machine-learning/decision-tree-algorithms/)
- [Support Vector Machine](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)
- [Logistic Regression](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)

---

<div align="center">
  <p>
    <strong>IF3170 - Inteligensi Artifisial</strong><br>
    Sekolah Teknik Elektro dan Informatika<br>
    Institut Teknologi Bandung<br>
    2025
  </p>
</div>
