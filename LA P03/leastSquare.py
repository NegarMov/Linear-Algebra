from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math 

def choose_training_data(data):
    rows = len(data)
    indices = np.sort(np.random.choice(rows, math.floor(rows * 0.95), False))
    training_data = np.ndarray((len(indices), 2), dtype='float')
    for i in range(len(indices)):
        training_data[i][0] = indices[i]
        training_data[i][1] = data[indices[i]]
    return training_data

def choose_test_data(data, training_data):
    indices = np.setdiff1d(range(len(data)), training_data[:, 0])
    test_data = np.ndarray((len(indices), 2), dtype='float')
    for i in range(len(indices)):
        test_data[i][0] = indices[i]
        test_data[i][1] = data[indices[i]]
    return test_data

def build_linear_matrix(data):
    rows = data.shape[0]
    res = np.ones((rows, 2), dtype='float')
    for i in range(rows):
        res[i, 1] = data[i][0]
    return res

def build_quadratic_matrix(data):
    rows = data.shape[0]
    res = np.ones((rows, 3), dtype='float')
    for i in range(rows):
        res[i, 1] = data[i][0]
        res[i, 2] = data[i][0] ** 2
    return res

def estimate_linear_equation(data):
    X = build_linear_matrix(data)
    Xt = np.transpose(X)
    v = data[:, 1]
    return np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, v))

def estimate_quadratic_equation(data):
    X = build_quadratic_matrix(data)
    Xt = np.transpose(X)
    v = data[:, 1]
    return np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, v))

def log_test_data(test_data, c, mode):
    print(f"\n\n{mode} ESTIMATION RESULT =============================================")
    print ("{:<20} {:<25} {:<25}".format('Real value','Estimated Value','Error'))
    for i in range(test_data.shape[0]):
        index = test_data[i][0]
        real_value = test_data[i][1]
        estimated_value = index * c[1] + c[0] if mode == "LINEAR" else c[2] * (index ** 2) + c[1] * index + c[0]
        print ("{:<20} {:<25} {:<25}".format(real_value, estimated_value, real_value - estimated_value))


if __name__ == "__main__":

    # Extract data
    df = pd.read_csv('covid_cases.csv')
    data = df[input("> Country's name: ")]
    rows = len(data)
    training_data = choose_training_data(data)
    test_data = choose_test_data(data, training_data)

    # Plot estimated line
    c = estimate_linear_equation(training_data)
    x = np.linspace(0, rows, rows)
    y = x * c[1] + c[0]
    plt.plot(x, y)
    # Plot test data
    plt.scatter(test_data[:, 0], test_data[:, 1], marker = '.', c = 'green')
    # Plot actual data
    plt.plot(data, linewidth = '0.5')
    plt.legend(['Estimated Linear Polynomial', 'Test Data', 'Actual Data'])
    plt.show()
    # Log test data info
    log_test_data(test_data, c, "LINEAR")


    # Plot estimated parabola
    c = estimate_quadratic_equation(training_data)
    y = c[2] * (x ** 2) + c[1] * x + c[0]
    plt.plot(x, y)
    # Plot test data
    plt.scatter(test_data[:, 0], test_data[:, 1], marker = '.', c = 'green')
    # Plot actual data
    plt.plot(data, linewidth = '0.5')
    plt.legend(['Estimated Quadratic Polynomial', 'Test Data', 'Actual Data'])
    plt.show()
    # Log test data info
    log_test_data(test_data, c, "QUADRATIC")