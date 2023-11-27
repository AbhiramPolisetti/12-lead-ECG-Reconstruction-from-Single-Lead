from reconstruction_utils import load_data, preprocess_data, build_model, train_model, evaluate_model, save_model,plot_results


def main():
    train_data = load_data(r"C:\Users\iabhi\Desktop\ECG_10\s0287lre.csv")
    test_data = load_data(r"C:\Users\iabhi\Desktop\ECG_10\s0291lre.csv")

    train_lead_ii_data = train_data['ii'].values
    train_lead_i_data = train_data['i'].values

    test_lead_ii_data = test_data['ii'].values
    test_lead_i_data = test_data['i'].values

    sequence_length = 1000

    X_train, y_train = preprocess_data(train_lead_ii_data, train_lead_i_data, sequence_length)
    X_test, y_test = preprocess_data(test_lead_ii_data, test_lead_i_data, sequence_length)

    model = build_model(sequence_length)

    epochs = 200
    batch_size = 512

    train_model(model, X_train, y_train, epochs, batch_size)

    y_pred = evaluate_model(model, X_test, y_test)

    save_model(model, 'model.h5')

    time_series = range(len(y_test))
    plot_results(time_series, y_test, y_pred)


if __name__ == "__main__":
    main()
