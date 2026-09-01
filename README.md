# 🩺 Diabetes AutoML Engine & Live Dashboard

A high-performance, lightweight **Automated Machine Learning (AutoML)** pipeline integrated with a responsive web-based user interface. This application automatically preprocesses clinical datasets, trains multiple algorithm variations, fine-tunes hyperparameters, create leaderboard and streams evaluation results down to a live status cockpit.

Here is Link of Live Website, 
automl-diabetes-dashboard-production.up.railway.app


---

## 🚀 Key Performance Indicators (Winning Model)

The machine learning framework automatically evaluated several candidate model architectures across variations of hyperparameter searching. The **RandomForestClassifier** emerged as the champion layout, producing near-perfect metrics on stratified validation loops:

* **F1 Score:** `99.49%`
* **ROC-AUC Score:** `99.91%`
* **Precision:** `99.50%`
* **Recall:** `99.50%`
* **Accuracy:** `99.50%`

*Note: The high classification accuracy is due to the strong boundary characteristics of clinical diagnostic features (like HbA1c and BMI values) present in the source dataset, which tree-based ensembling layouts map with near-perfect reliability.*

---

## 🛠️ Features & Technical Stack

### Core Engine (Backend Machine Learning)
* **Framework:** Scikit-Learn, Pandas, Joblib.
* **Feature Pipeline:** Automated handling of `Numerical` variables (via standard scaling) and `Categorical` arrays (via one-hot encoding), bundled together inside a structural `ColumnTransformer` to prevent data leakage during hyperparameter cross-validation.
* **AutoML Grid Search:** Exhaustive `GridSearchCV` implementing a 5-fold cross-validation routine tracking generalized accuracy across three core algorithm groups:
  * `RandomForestClassifier` (Ensemble Trees)
  * `DecisionTreeClassifier` (Single Structural Tree)
  * `LogisticRegression` (Linear Baseline Classifier with balanced weights)

### Interface Control Room (Frontend UI)
* **Languages:** Vanilla HTML5, CSS3 Grid layouts, and Modern asynchronous JavaScript (ES6+).
* **Live Status Tracking:** Leverages asynchronous `fetch()` API polling loops targeting your back-end server endpoints (`GET /api/automl/status`) to update experiments completed, current pipeline status, and real-time evaluation logs without browser refreshes.
* **Responsive Architecture:** Fully adapting CSS media configurations designed to maintain a clean modular grid on mobile, tablet, and widescreen developer setups.

---

## ⚙️ Local Installation & Launch

Follow these steps to run the pipeline and explore the dashboard interface locally:

### 1. Set Up Your Environment
Ensure Python 3.10+ is installed on your computer. Create a virtual environment and update your package installers:
```bash
# Initialize environmental sandbox
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install essential dependencies
pip install pandas scikit-learn joblib fastapi uvicorn
```

### 2. Execute the AutoML Training Loop
Run the core script to verify parsing structure, clean missing categorical data, evaluate algorithms, and dump metrics to standard output:
```bash
python ml_engine.py
```

---

## 🧠 Future Enhancement Milestones
* [ ] **Feature Importances Plot:** Inject charts into the UI to dynamically graph top driving attributes (such as HbA1c weight distribution).
* [ ] **Interactive Inference Portal:** Add an intake form component allowing doctors or users to input specific patient lab values manually to generate real-time predictions.
* [ ] **Confusion Matrix Visualizers:** Build a 2x2 color-mapped grid component tracking true positives and false negatives dynamically.
﻿
