import numpy as np
import time
import matplotlib.pyplot as plt

def generate_data(n, d):
    """
    Generates a synthetic linear regression dataset.
    The data is created using an underlying true linear model:
        y = X w_true + noise

    where:
        - X is an (n x d) matrix of input features,
        - w_true is a randomly chosen weight vector of length d,
        - noise is small random Gaussian noise added to simulate
          measurement errors or natural variability in real data.

    Parameters:
        - n : Number of data points (samples)
        - d : Number of features (dimensions)

    Returns:
        - X : Feature matrix of shape (n, d)
        - y : Target vector of length n
    """
    X = np.random.randn(n, d)
    true_w = np.random.randn(d)
    y = X @ true_w + np.random.randn(n) * 0.1
    return X, y

def f(w, X, y):
    """
    Computes the Mean Squared Error (MSE) loss for a linear model.

    The model predicts outputs using the linear function:
        y_pred = Xw

    The loss function being minimized is:
        f(w) = (1/n) * Σ (y_i - y_pred_i)^2

    This function evaluates how far the predicted values are from the true labels
    by averaging the squared differences over all data points.

    Parameters:
        - w : Weight vector of the linear model
        - X : Input feature matrix (each row is one data point)
        - y : True target values

    Returns:
        Mean Squared Error loss value
    """
    y_pred = X @ w
    return np.mean((y - y_pred) ** 2)

def df(w, X, y):
    """
    Computes the gradient of the Mean Squared Error (MSE) loss with respect to
    the weight vector w.

    Starting from the loss function:
        f(w) = (1/n) * Σ (y_i - X_i w)^2

    Taking the derivative with respect to w and applying the chain rule gives:
        ∂f(w)/∂w = (-2/n) * X^T (y - Xw)

    This gradient points in the direction of steepest increase of the loss.
    Gradient Descent updates the weights in the opposite direction to minimize
    the error.

    Parameters:
        - w : Weight vector of the linear model
        - X : Input feature matrix (each row is one data point)
        - y : True target values

    Returns:
        Gradient of the MSE loss with respect to w
    """
    n = len(y)
    y_pred = X @ w
    return (-2 / n) * X.T @ (y - y_pred)

def gradient_descent(X, y, lr=0.1, tolerance=1e-6, max_epochs=5000):
    """
    Performs Full Batch Gradient Descent to minimize the Mean Squared Error (MSE)
    loss for a linear regression model.

    The algorithm iteratively updates the weight vector using:
        w_{new} = w - lr * ∂f(w)/∂w

    Parameters:
        - X : Input feature matrix (n samples × d features)
        - y : True target values
        - lr : Learning rate (step size in the direction of the negative gradient)
        - tol : Convergence tolerance on change in loss
        - max_epochs : Maximum number of gradient descent iterations

    Returns:
        - Number of iterations taken to converge
        - Final loss value (the MSE)
    """
    n, d = X.shape
    w = np.zeros(d)
    prev_loss = float('inf')

    # Implement your code here
    for epoch in range(max_epochs):
        loss = f(w, X, y)

        if abs(prev_loss - loss) < tolerance:
            return epoch, loss
        prev_loss = loss

        w = w - lr * df(w, X, y)

    return max_epochs, loss

def plot_graphs(x_vals, y_vals, x_label, y_label, title):
    """
    Plots a 2D graph of experimental results.

    Parameters:
        - x_vals : Values on the x-axis (independent variable)
        - y_vals : Values on the y-axis (measured result)
        - x_label : Label for the x-axis
        - y_label : Label for the y-axis
        - title : Title of the plot
    """
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def experiment_1():
    """
    Experiment 1: 
    Evaluates how the number of training examples affects the number of
    gradient descent iterations required for convergence.
    """
    sizes = [100, 500, 1000, 5000, 10000]
    iterations_avg = []
    trials = 30

    for n in sizes:
        trial_iters = []
        for i in range(trials):
            X, y = generate_data(n, d=5)
            iters, _ = gradient_descent(X, y)
            trial_iters.append(iters)

        iterations_avg.append(np.mean(trial_iters))

    plot_graphs(sizes, iterations_avg, "Number of Data Points", "Average Iterations to Converge", f"Iterations to Converge vs Number of Data Points({trials} trials)")

def experiment_2():
    """
    Experiment 2: 
    Evaluates how the number of input features (dimension of the data)
    affects the time required for gradient descent to converge.

    The experiment fixes the number of data points and increases the
    number of features, measuring the total runtime until convergence.
    """
    dimensions = [1, 5, 10, 20, 50, 100]
    times_avg= []
    trials = 30

    # Implement your code here
    n_data = 10000

    for d in dimensions:
        trials_times = []
        for i in range(trials):
            X, y = generate_data(n_data, d)

            time_start = time.time()

            gradient_descent(X, y)

            time_end = time.time()
            trials_times.append(time_end-time_start)

        times_avg.append(np.mean(trials_times))

    plot_graphs(dimensions, times_avg, "Number of Features", "Time to Converge (seconds)", f"Time to Converge vs Number of Features({trials} trials)")

if __name__ == "__main__":
    experiment_1()
    experiment_2()
