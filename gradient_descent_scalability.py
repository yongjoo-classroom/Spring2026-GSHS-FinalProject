import numpy as np
import time
import matplotlib.pyplot as plt

def generate_data(n, d):
    X = np.random.randn(n, d)
    true_w = np.random.randn(d)
    y = X @ true_w + np.random.randn(n) * 0.1
    return X, y

def f(w, X, y):
    y_pred = X @ w
    return np.mean((y - y_pred) ** 2)

def df(w, X, y):
    n = len(y)
    y_pred = X @ w
    return (-2 / n) * X.T @ (y - y_pred)

def gradient_descent(X, y, lr=0.1, tolerance=1e-6, max_epochs=5000):
    n, d = X.shape
    w = np.zeros(d)
    prev_loss = float("inf")

    for epoch in range(max_epochs):
        loss = f(w, X, y)
        if abs(prev_loss - loss) < tolerance:
            return epoch + 1, loss
        grad = df(w, X, y)
        w = w - lr * grad
        prev_loss = loss

    return max_epochs, loss

def plot_graphs(x, y, xlabel, ylabel, title):
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def experiment_convergence_vs_data_size():
    sizes = [100, 500, 1000, 5000, 10000]
    iterations = []

    for n in sizes:
        X, y = generate_data(n, d=5)
        iters, _ = gradient_descent(X, y)
        iterations.append(iters)

    print("Iterations vs Number of Data Points")
    for n, it in zip(sizes, iterations):
        print(f"(n={n}, iterations={it})")

    plot_graphs(
        sizes,
        iterations,
        "Number of Data Points",
        "Iterations to Converge",
        "Convergence vs Data Size"
    )

def experiment_runtime_vs_num_features():
    dimensions = [1, 5, 10, 20, 50, 100]
    times = []
    n_fixed = 5000

    for d in dimensions:
        X, y = generate_data(n_fixed, d)
        t0 = time.perf_counter()
        gradient_descent(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print("\nRuntime vs Number of Features")
    for d, t in zip(dimensions, times):
        print(f"(d={d}, runtime={t})")

    plot_graphs(
        dimensions,
        times,
        "Number of Features",
        "Runtime (seconds)",
        "Runtime vs Number of Features"
    )

if __name__ == "__main__":
    experiment_convergence_vs_data_size()
    experiment_runtime_vs_num_features()
