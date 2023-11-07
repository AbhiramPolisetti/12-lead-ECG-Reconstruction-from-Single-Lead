# main.py
from data_loader import load_data, preprocess_data
from model_builder import build_model, train_model
from model_evaluation import evaluate_model, visualize_results, calculate_metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split

def main_workflow(file_path, input_column, target_column, sequence_length):
    input_data, target_data = load_data(file_path, input_column, target_column)
    X, y, scaler = preprocess_data(input_data, target_data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(sequence_length)
    train_model(model, X_train, y_train, epochs=200, batch_size=512)

    actual_values, predicted_values = evaluate_model(model, X_test, y_test, scaler)
    visualize_results(actual_values, predicted_values)
    calculate_metrics(actual_values, predicted_values)


    model_name = f"{target_column}_model.h5"
    model.save(model_name)

if __name__ == '__main__':
    file_path = r'C:\Users\iabhi\PycharmProjects\12 Lead ECG  reconstruction\data\s0001_re.csvs0001_re.csv'
    input_column = 'ii'
    target_column = 'i'
    sequence_length = 1000

    main_workflow(file_path, input_column, target_column, sequence_length)
