# Scalability of Gradient Descent 

In this assignment, you will study how the Gradient Descent algorithm for linear regression scales with:

1. The number of data points (dataset size)  
2. The number of features (data dimensionality)

## Objective

Submit a written report on the scalability of gradient descent using the format provided in `final_report_template.zip`. You can use the provided starter code `gradient_descent_scalability.py` to run the experiments and generate plots that illustrate how the convergence behavior and runtime of gradient descent vary as the problem size increases.

1. **Data Generation**\
The function `generate_data(n, d)` creates a synthetic regression dataset using the below model.

$$
y = X w_{\text{true}} + \text{noise}
$$

where:
- \(n\) = number of data points  
- \(d\) = number of features  

2. **Loss Function**\
The objective function `f(w, X, y)` we aim to minimize is the Mean Squared Error (MSE), defined as:

$$
f(w) = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - X_i w\right)^2
$$

3. **Gradient of the Loss**\
The function `df(w, X, y)` computes the gradient of the loss using the chain rule.

$$
\nabla f(w) = -\frac{2}{n} X^{T}(y - Xw)
$$

This is the direction of steepest increase of the loss. Gradient Descent moves in the opposite direction.

4. **Gradient Descent**
    - You must implement the gradient descent algorithm in the function `gradient_descent(X, y, lr, tol, max_epochs)`.
    - For each step, compute the loss and its gradient and update the weights.
    - Stop when the change in loss is smaller than tolerance.
```
w = w − lr * ∂f(w)/∂w
```
    
5. **Plotting**\
The function `plot_graphs(...)` is provided to visualize results.

## Experiments

1. **Convergence vs Data Size**
    - Measures the number of gradient descent iterations are required to converge, as the number of data points varies.


2. **Runtime vs Number of Features**
    - Should measure the total time taken for gradient descent to converge when the number of features vary.
    - This needs to be implemented.

3. **Note:** You are encouraged to design and include additional experiment beyond those provided to further explore the behavior and scalability of gradient descent.