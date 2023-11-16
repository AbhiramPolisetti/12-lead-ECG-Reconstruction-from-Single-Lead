# evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, scaler):
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    predicted_values = model.predict(X_test)
    predicted_values = scaler.inverse_transform(predicted_values)

    mse = mean_squared_error(y_test, predicted_values)
    mae = mean_absolute_error(y_test, predicted_values)
    r2 = r2_score(y_test, predicted_values)

    # Print Metrics
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)

    return mse, mae, r2, predicted_values


def visualize_results(actual_values, predicted_values):
    """Visualize actual vs. predicted values."""
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    sns.lineplot(data=actual_values[:500], color="blue", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("ECG Value")
    plt.title("Actual Values")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    sns.lineplot(data=predicted_values[:500], color="blue", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("ECG Value")
    plt.title("Predicted Values")
    plt.show()
