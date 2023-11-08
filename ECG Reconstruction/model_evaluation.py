# model_evaluation.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def evaluate_model(model, X_test, y_test, scaler):
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    predicted_values = model.predict(X_test)
    actual_values = scaler.inverse_transform(y_test)
    predicted_values = scaler.inverse_transform(predicted_values)

    return actual_values, predicted_values


def visualize_results(actual_values, predicted_values):
    time_series = range(len(actual_values))
    df = pd.DataFrame({'Time': time_series, 'Actual': actual_values.flatten(), 'Predicted': predicted_values.flatten()})

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Time', y='Actual', data=df[:500], label='Actual', linewidth=2)
    sns.lineplot(x='Time', y='Predicted', data=df[:500], label='Predicted', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Lead I')
    plt.title('Actual vs. Predicted Lead I')
    plt.legend()
    plt.show()


def calculate_metrics(actual_values, predicted_values):
    r2 = r2_score(actual_values, predicted_values)
    correlation_coefficient, _ = pearsonr(actual_values.flatten(), predicted_values.flatten())

    regression_model = LinearRegression()
    regression_model.fit(actual_values.reshape(-1, 1), predicted_values)
    regression_coefficient = regression_model.coef_[0]

    print(f"R-squared: {r2:.4f}")
    print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
    print(f"Regression Coefficient: {regression_coefficient[0]:.4f}")
