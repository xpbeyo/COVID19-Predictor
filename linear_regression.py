import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(lag_vars, lag_time, train=True):
    # Load data from the source csv file and transform into lag observations
    # of the daily confirmed cases (single variable). 
    # Return (X_train, y_train, X_val, y_val) 

    df = pd.read_csv('covid19_sg_clean.csv', sep=',', header=None)
    data = df.values[1:, 1]

    if train:
        data = data[:int(np.round(data.shape[0]*0.8))]      # Use front 80% data for training and validation
    else:
        data = data[int(np.round(data.shape[0]*0.8)):]      # Use tail 20% for testing
    data = data.astype(np.int32)

    num_examples = len(data) - lag_vars - lag_time + 1
    X = np.zeros((num_examples, 1))

    # Append examples
    for i in range(lag_vars):
        X = np.hstack((X, np.reshape(data[i:i+num_examples], (num_examples, 1))))
    X = np.delete(X, 0, 1)        # Remove placeholder zeros

    # Targets
    # y = np.reshape(data[lag_vars+lag_time-1:], (num_examples, 1))
    y = data[lag_vars+lag_time-1:]

    if train:
        split = int(np.round(num_examples*0.7))     # Split the data 7:3 for training and test without shuffling
        return X[:split], y[:split], X[split:], y[split:] 
    
    else:           # Test data 
        return X, y


if __name__ == '__main__':
    
    experiments = []
    for i in range(1, 21):
        for j in range(1, 11):

            # Hyperparameters
            lag_vars = i
            lag_time = j

            X_train, y_train, X_val, y_val = load_data(lag_vars, lag_time)
            reg = LinearRegression().fit(X_train, y_train)
            train_score = reg.score(X_train, y_train)
            val_score = reg.score(X_val, y_val)
            RMSE = mean_squared_error(reg.predict(X_val), y_val, squared=False)

            print(f'lag_vars = {lag_vars} \nlag_time = {lag_time}')
            print(f'Training set score: {train_score}')
            print(f'Val set score: {val_score}')
            print(f'RMSE: {RMSE}\n')

            experiments.append((lag_vars, lag_time, train_score, val_score, RMSE))

    s = sorted(experiments, key=lambda t: t[3])
    r = sorted(experiments, key=lambda t: t[4])
    print(f'The best model based on val_score: \n'
            f'Lag Variables: {s[-1][0]}, Lag Time: {s[-1][1]}, Val Score: {s[-1][3]:.4f}, RMSE: {s[-1][4]:.4f}\n\n')
    print(f'The best model based on RMSE: \n'
            f'Lag Variables: {r[0][0]}, Lag Time: {r[0][1]}, Val Score: {r[0][3]:.4f}, RMSE: {r[0][4]:.4f}\n\n')

    lag_vars_best = r[0][0] 
    lag_time_best = r[0][1]

    # Load data and train model with best set of hyperparameters
    X_train, y_train, _, _ = load_data(lag_vars_best, lag_time_best)
    X_test, y_test = load_data(lag_vars_best, lag_time_best, train=False)
    reg = LinearRegression().fit(X_train, y_train)
    train_score = reg.score(X_train, y_train)

    # Test set results
    test_score = reg.score(X_test, y_test)
    predictions = reg.predict(X_test)
    test_RMSE = mean_squared_error(predictions, y_test, squared=False)
    print(f'Test set results with the best hyperparameters:\n'
            f'Test Score: {test_score}, Test RMSE: {test_RMSE}')
    
    # Plot

    # plt.figure()
    # plt.plot(np.arange(len(y_train)), reg.predict(X_train), 'r', label='Predictions')
    # plt.plot(np.arange(len(y_train)), y_train, 'b', label='Targets')
    # plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(np.arange(len(predictions)), predictions, 'r', label='Predictions')
    plt.plot(np.arange(len(predictions)), y_test, 'b', label='Targets')
    plt.xlabel('Date')
    plt.ylabel('Daily Confirmed Cases')
    plt.legend()
    plt.show()

