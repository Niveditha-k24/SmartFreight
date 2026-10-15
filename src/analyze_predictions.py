
import joblib
import pandas as pd

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

# 1. Load the data
df = load_vendor_invoice_data("data/inventory.db")

# 2. Prepare input features and target
X, y = prepare_features(df)

# 3. Recreate the original train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

# 4. Load the saved Linear Regression model
model = joblib.load("models/linear_regression.pkl")

# 5. Predict freight costs for the test records
predictions = model.predict(X_test)

# 6. Create a table containing inputs and results
results = X_test.copy()
results["Actual_Freight"] = y_test
results["Predicted_Freight"] = predictions

# 7. Calculate prediction errors
results["Error"] = (
    results["Predicted_Freight"] - results["Actual_Freight"]
)

results["Absolute_Error"] = results["Error"].abs()

# 8. Show the ten records with the largest errors
largest_errors = results.sort_values(
    by="Absolute_Error",
    ascending=False
)

print("\n10 records with the largest errors:")
print(largest_errors.head(10).round(2).to_string())


# 9. Investigate the original invoice details
top_10 = largest_errors.head(10)

columns_to_inspect = [
    "VendorNumber",
    "VendorName",
    "InvoiceDate",
    "PONumber",
    "PODate",
    "PayDate",
    "Quantity",
    "Dollars",
    "Freight",
    "Approval"
]

print("\nOriginal invoice details for the largest errors:")

print(
    df.loc[top_10.index, columns_to_inspect]
    .round(2)
    .to_string()
)

# 10. Measure the contribution of the 10 largest errors

# Square every prediction error
results["Squared_Error"] = results["Error"] ** 2

# Find the total squared error across all test records
total_squared_error = results["Squared_Error"].sum()

# Find the squared error contributed by the 10 worst records
top_10_squared_error = (
    results.nlargest(10, "Absolute_Error")["Squared_Error"].sum()
)

# Calculate their percentage contribution
percentage = (
    top_10_squared_error / total_squared_error
) * 100

print("\nImpact of the 10 largest errors:")
print(f"Total squared error: {total_squared_error:.2f}")
print(f"Top 10 squared error: {top_10_squared_error:.2f}")
print(f"Percentage contributed by top 10: {percentage:.2f}%")

# 11. Investigate freight cost relative to quantity

top_10 = results.nlargest(10, "Absolute_Error").copy()

top_10["Actual_Freight_Per_Unit"] = (
    top_10["Actual_Freight"] / top_10["Quantity"]
)

top_10["Dollars_Per_Unit"] = (
    top_10["Dollars"] / top_10["Quantity"]
)

print("\nFreight per unit for the 10 largest errors:")

print(
    top_10[
        [
            "Quantity",
            "Dollars",
            "Actual_Freight",
            "Predicted_Freight",
            "Actual_Freight_Per_Unit",
            "Dollars_Per_Unit",
            "Absolute_Error"
        ]
    ].round(4).to_string()
)

# Compare with the full dataset
df["Freight_Per_Unit"] = df["Freight"] / df["Quantity"]

print("\nFreight per unit in the full dataset:")
print(df["Freight_Per_Unit"].describe().round(4))

# 12. Inspect the highest freight-per-unit records

df["Freight_Per_Unit"] = df["Freight"] / df["Quantity"]

highest_freight_per_unit = df.nlargest(
    10, "Freight_Per_Unit"
)

columns_to_inspect = [
    "VendorNumber",
    "VendorName",
    "InvoiceDate",
    "Quantity",
    "Dollars",
    "Freight",
    "Freight_Per_Unit"
]

print("\n10 invoices with the highest freight per unit:")

print(
    highest_freight_per_unit[columns_to_inspect]
    .round(4)
    .to_string()
)
