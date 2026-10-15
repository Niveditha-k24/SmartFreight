import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():

    # Database path
    db_path = "data/inventory.db"

    # Create models folder if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    lr_results = evaluate_model(
        lr_model,
        X_test,
        y_test,
        "Linear Regression"
    )

    dt_results = evaluate_model(
        dt_model,
        X_test,
        y_test,
        "Decision Tree"
    )

    rf_results = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    # Save models
    joblib.dump(lr_model, model_dir / "linear_regression.pkl")
    joblib.dump(dt_model, model_dir / "decision_tree.pkl")
    joblib.dump(rf_model, model_dir / "random_forest.pkl")


if __name__ == "__main__":
    main()