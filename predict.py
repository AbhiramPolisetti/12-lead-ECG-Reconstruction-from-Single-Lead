import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_data(data_path, sequence_length=1000):
    df = pd.read_csv(data_path)
    lead_ii_values = df['ii'].values
    
    # Normalize the data
    scaler = MinMaxScaler()
    lead_ii_values = scaler.fit_transform(lead_ii_values.reshape(-1, 1))
    lead_ii_values = np.repeat(lead_ii_values, sequence_length, axis=-1)
    
    return lead_ii_values

def predict_data(model, lead_ii_values):
    predictions = model.predict(lead_ii_values)
    return predictions

def save_predictions(predictions, output_path='lead_i_predictions.csv'):
    #To Create a DataFrame with the predictions
    data = {'Lead I': predictions.ravel()}
    predictions_df = pd.DataFrame(data)
    
    #To  Save predictions to a CSV file without an index column
    predictions_df.to_csv(output_path)

def plot_actual_vs_predicted(actual_values, predicted_values):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual Values', color='blue')
    plt.plot(predicted_values, label='Predicted Values', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #path to the ECG data file
    data_path = '/content/s0001_re.csv'

    #path to the model 
    model_path = '/content/lead_I.h5'

    # Load the  model
    model = load_model(model_path)

    # Preprocess data
    lead_ii_values = preprocess_data(data_path)

    # Make predictions using the model
    lead_i_predictions = predict_data(model, lead_ii_values)

    # Save 'Lead I' predictions to a CSV file
    save_predictions(lead_i_predictions, 'predictions.csv')

    # Visualize actual vs predicted values
    actual_lead_i_values = lead_ii_values.ravel()
    plot_actual_vs_predicted(actual_lead_i_values, lead_i_predictions.ravel())
