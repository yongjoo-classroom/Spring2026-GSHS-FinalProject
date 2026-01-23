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
    epoch=0
    while True:
        loss=f(w,X,y)
        if abs(prev_loss-loss)<tolerance or epoch>=max_epochs:
            break
        epoch+=1
        grad=df(w,X,y)
        grad = np.clip(grad, -1e3, 1e3)
        w-=lr*grad
        prev_loss=loss
    return epoch, loss

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
    iterations = []

    for n in sizes:
        X, y = generate_data(n, d=5)
        iters, _ = gradient_descent(X, y)
        iterations.append(iters)

    plot_graphs(sizes, iterations, "Number of Data Points", "Iterations to Converge", "Iterations to Converge vs Number of Data Points")

def experiment_2():
    """
    Experiment 2: 
    Evaluates how the number of input features (dimension of the data)
    affects the time required for gradient descent to converge.

    The experiment fixes the number of data points and increases the
    number of features, measuring the total runtime until convergence.
    """
    dimensions = [1, 5, 10, 20, 50, 100]
    times = []

    for d in dimensions:
        X,y=generate_data(n=1000,d=d)
        start=time.time()
        result=gradient_descent(X,y)
        end=time.time()
        times.append(end-start)
        

    plot_graphs(dimensions, times, "Number of Features", "Time to Converge (seconds)", "Time to Converge vs Number of Features")
def experiment_3():
    lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    iters = []
    times=[]
    for lr in lrs:
        X,y=generate_data(n=1000,d=10)
        start=time.time()
        iter, loss=gradient_descent(X,y,lr=lr)
        end=time.time()
        iters.append(iter)
        times.append(end-start)
    plot_graphs(lrs, times, "learnig rate", "Time to Converge (seconds)", "Time to Converge vs learnig rate")
    plot_graphs(lrs, iters, "learnig rate", "Iterations to Converge", "Iterations to Converge vs learnig rate")
if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()

