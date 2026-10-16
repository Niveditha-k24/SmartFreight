#  SmartFreight
### Machine Learning-Based Freight Cost Estimation

SmartFreight is a machine learning project that explores how historical vendor invoice data can be used to estimate freight costs based on quantity and invoice value.

The idea was inspired by my internship experience in freight operations, where delays in receiving freight quotations could sometimes affect responses to clients. This project explores whether data-driven estimates can help provide an initial indication of freight costs.

##  Objectives
- Analyze historical vendor invoice data.
- Explore relationships between quantity, invoice value, and freight costs.
- Train and compare different regression models.
- Build a simple interactive interface for freight cost estimation.

##  Technologies Used
- Python
- Pandas
- Scikit-learn
- SQLite
- Joblib
- Streamlit
- Git and GitHub

##  Machine Learning Models
The project trains and evaluates:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R² score.

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 24.46 | 124.43 | 0.9700 |
| Decision Tree | 33.37 | 150.60 | 0.9561 |
| Random Forest | 27.02 | 130.76 | 0.9669 |

Linear Regression performed best on the current test set. These metrics describe performance on the held-out data and do not guarantee the same performance on new invoices.

## Application
The project includes a Streamlit interface where users can enter quantity and invoice value to generate a freight cost estimate.

**Note:** Predictions are experimental estimates based on historical data, not guaranteed shipping quotations.

##  Running the Project

From the project root, install the required dependencies listed in `requirements.txt`, if that file is present and up to date.

Run the training pipeline:

```bash
py src/train_model.py
```

Run the Streamlit application:

```bash
py -m streamlit run app/app.py
```

##  Limitations
- Predictions use only quantity and invoice value.
- Historical patterns may not reflect current freight rates or every shipping scenario.
- Further validation and feature engineering are needed before real-world operational use.

##  Future Improvements
- Explore additional relevant shipping features.
- Investigate large prediction errors.
- Validate performance on additional data.
- Improve the model's generalization to unseen invoices.

##  Author
Niveditha Karthikeyan  
MCA Student

## 🔗 Repository
https://github.com/Niveditha-k24/SmartFreight
