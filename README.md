# Sales Prediction ML Engine & Dashboard

A premium, interactive machine learning application that predicts product sales based on advertising budgets spent across different channels (TV, Radio, and Newspaper). This repository provides a complete end-to-end data science workflow—ranging from exploratory data analysis (EDA) and multi-model training/evaluation pipelines to a CLI prediction tool and a modern, glassmorphic Flask web dashboard.

---

## 🚀 Key Features

*   **Comprehensive EDA:** Auto-generated correlation heatmaps, feature distribution plots, and regression path visualization.
*   **Multi-Model Training Pipeline:** Automatically trains and evaluates **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**, selecting and optimizing the best model architecture.
*   **Web Dashboard:** A high-performance, dark-themed Flask web app styled with premium glassmorphism, responsive sliders, real-time ROI/revenue estimation, and interactive metric comparison tabs.
*   **CLI Prediction Engine:** Quick console-based inference using serialized models.

---

## 📁 Repository Structure

```text
sales_prediction/
├── Advertising.csv         # Historical advertising campaign dataset
├── app.py                  # Flask web backend application
├── predict.py              # Command-line prediction tool
├── requirements.txt        # Python package dependencies
├── models/                 # Saved machine learning models & metadata
│   ├── sales_model.pkl     # Retrained best-performing model (Gradient Boosting)
│   └── metrics.json        # Evaluation metrics for comparison
├── src/                    # Data processing & modeling source code
│   ├── eda.py              # Exploratory Data Analysis & visual asset generator
│   └── train.py            # Model training, evaluation & serialization pipeline
└── static/                 # Web dashboard frontend assets
    ├── index.html          # Dashboard HTML structure
    ├── script.js           # Client-side validation, APIs & gauge rendering
    ├── style.css           # Premium styling, layouts, CSS blobs, & animations
    └── plots/              # Visual assets generated during EDA
        ├── correlation_matrix.png
        ├── distributions.png
        └── sales_vs_features.png
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies locally:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries and packages specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📈 Execution Workflow

### Step 1: Run Exploratory Data Analysis (EDA)
Execute the EDA script to inspect data statistics and generate correlation heatmaps and distribution plots:
```bash
python src/eda.py
```
This will print descriptive statistics to your terminal and save three visual reports inside `static/plots/`.

### Step 2: Train & Evaluate Models
Run the training pipeline to evaluate the model architectures and save the best one:
```bash
python src/train.py
```
This script:
1. Loads the historical training dataset (`Advertising.csv`).
2. Splits the data into training (80%) and testing (20%) datasets.
3. Evaluates Linear Regression, Random Forest, and Gradient Boosting Regressors using $R^2$, MAE, MSE, and RMSE.
4. Serializes the best-performing model (`Gradient Boosting Regressor`) to `models/sales_model.pkl` and writes comparison metrics to `models/metrics.json`.

---

## 💻 Running the Services

### Option A: Launch the Web Dashboard
Launch the local development Flask server:
```bash
python app.py
```
After launching, open your browser and navigate to:
```text
 http://127.0.0.1:8000
```
Use the interactive sliders to adjust TV, Radio, and Newspaper budgets to see immediate predicted sales, estimated revenue, ROI metrics, and model comparison statistics.

### Option B: Perform CLI Predictions
Make predictions directly from the command line by passing budget parameters:
```bash
python predict.py --tv 150.0 --radio 25.0 --newspaper 10.0
```
This displays a beautiful text card mapping inputs to predicted sales units:
```text
==================================================
           SALES PREDICTION ENGINE
==================================================
Input Budgets:
  - TV Advertising:        $150.00k
  - Radio Advertising:     $25.00k
  - Newspaper Advertising: $10.00k
--------------------------------------------------
  => Predicted Sales:      16.58 thousand units
==================================================
```

---

## 📊 Evaluation & Metrics
The models are evaluated on $R^2$ Score (Coefficient of Determination), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Below is an overview of representative metrics:

| Model Architecture | $R^2$ Score | MAE | RMSE |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | ~98.31% | 0.6187 | 0.7298 |
| **Random Forest** | ~98.13% | 0.6201 | 0.7686 |
| **Linear Regression** | ~89.94% | 1.4608 | 1.7816 |

*(Note: Run `python src/train.py` to compile the precise local metrics based on your workspace setup)*.
