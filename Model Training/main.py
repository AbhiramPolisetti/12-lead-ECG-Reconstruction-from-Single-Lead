# main.py
from data_loading import load_data, preprocess_data
from model import build_model, train_model, save_model
from evaluation import evaluate_model, visualize_results


def pipeline(file_path, target_column):
    # Load Data
    data = load_data(file_path)

    # Preprocess Data
    X_train, X_test, y_train, y_test, scaler,input_shape = preprocess_data(data, target_column)

    # Build Model
    model = build_model(input_shape)

    # Train Model
    trained_model = train_model(model, X_train, y_train, epochs=10, batch_size=32)

    # Save Model
    save_model(trained_model, f'{target_column}_model.h5')

    # Evaluate Model
    mse, mae, r2, predicted_values = evaluate_model(trained_model, X_test, y_test, scaler)

    # Print Metrics
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)

    # Visualize Results
    visualize_results(y_test, predicted_values)


if __name__ == "__main__":
    file_path = 's0002_re.csv'
    target_column = 'i'

    pipeline(file_path, target_column)
