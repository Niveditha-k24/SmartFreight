# Vendor Invoice Intelligence System

A machine learning project that analyzes historical vendor invoice data and predicts freight costs using regression models.

## Overview

During my internship at Pantone Freight, I observed that waiting for freight quotations from shipping companies could delay responses to clients. This project explores how historical invoice data can be used to generate preliminary freight cost estimates.

The system uses historical vendor invoice records to train and evaluate machine learning models and provides a Streamlit interface for generating predictions.

**Note:** This is an experimental prediction system, not a replacement for actual shipping quotations or live carrier pricing.

## Objectives

- Analyze historical vendor invoice data.
- Explore relationships between quantity, invoice value, and freight costs.
- Train and compare multiple regression models.
- Evaluate model performance using regression metrics.
- Build a user interface for freight cost prediction.

## Technologies Used

- **Language:** Python
- **Data analysis:** Pandas
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Database:** SQLite
- **User interface:** Streamlit
- **Version control:** Git and GitHub

## Dataset

The project uses a historical vendor invoice dataset containing 5,543 records.

Selected features:

| Feature | Description |
|---|---|
| `Quantity` | Quantity recorded in the invoice |
| `Dollars` | Invoice value |
| `Freight` | Actual freight charge used as the prediction target |

The model uses `Quantity` and `Dollars` as input features and predicts `Freight`.

## Machine Learning Workflow

1. Load invoice records from SQLite.
2. Prepare input features and the target variable.
3. Split the data into training and testing sets using an 80:20 ratio.
4. Train three regression models.
5. Evaluate their predictions on the test set.
6. Save the trained models.
7. Load the selected model in the Streamlit application.

## Models and Evaluation

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 24.46 | 124.43 | 0.9700 |
| Decision Tree Regressor | 33.37 | 150.60 | 0.9561 |
| Random Forest Regressor | 27.02 | 130.76 | 0.9669 |

Linear Regression achieved the best results among the three models on the current test split.

These results are specific to this dataset and evaluation setup. R² is not the same as prediction accuracy, and further validation is required before using predictions for real freight quotations.

## Application

The Streamlit application allows users to:

- Enter invoice quantity.
- Enter invoice value.
- Generate a predicted freight cost.
- View the estimate with a disclaimer about its limitations.

## Project Structure

```text
VendorInvoiceProject/
├── app/
│   └── app.py
├── data/
│   └── inventory.db
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── train_model.py
│   └── analyze_predictions.py
├── .gitignore
├── README.md
└── requirements.txt
```

The database and generated model files are local project assets and are not intended to be committed to GitHub. Obtain or prepare the dataset separately before running the project.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Niveditha-k24/Vendor-Invoice-Intelligence-System.git
cd Vendor-Invoice-Intelligence-System
```

### 2. Create a virtual environment

```bash
py -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Prepare the data

Place the SQLite database at:

```text
data/inventory.db
```

Ensure it contains the `vendor_invoice` table with the required columns.

### 5. Train the models

```bash
py src/train_model.py
```

This generates the trained model files in the `models/` directory.

### 6. Launch the application

```bash
py -m streamlit run app/app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Limitations and Future Improvements

- Investigate extreme prediction errors and validate the model on additional data.
- Incorporate relevant shipping information, such as origin, destination, transport mode, and container type, when suitable data is available.
- Compare performance across different vendor and shipment categories.
- Improve input validation and prediction monitoring.
- Evaluate the model on more recent or independent data.

## Author

**Niveditha Karthikeyan**

MCA Student | Machine Learning Project

GitHub: [Niveditha-k24](https://github.com/Niveditha-k24)
