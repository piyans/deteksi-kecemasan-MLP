# Training model MLP
def train_model(X, Y, hidden_size, num_iterations, learning_rate):
    input_size = X.shape[1]
    output_size = Y.shape[1]
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    for i in range(num_iterations):
        cache = forward_propagation(X, parameters)
        A2 = cache['A2']
        # Backward pass
        dZ2 = A2 - Y.T
        dW2 = (1 / X.shape[0]) * np.dot(dZ2, cache['A1'].T)
        db2 = (1 / X.shape[0]) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(parameters['W2'].T, dZ2) * relu_derivative(cache['Z1'])
        dW1 = (1 / X.shape[0]) * np.dot(dZ1, X)
        db1 = (1 / X.shape[0]) * np.sum(dZ1, axis=1, keepdims=True)
        # Update parameter
        parameters['W1'] -= learning_rate * dW1
        parameters['b1'] -= learning_rate * db1
        parameters['W2'] -= learning_rate * dW2
        parameters['b2'] -= learning_rate * db2
        if i % 100 == 0:
            cost = compute_cost(A2, Y)
            print(f"Iteration {i}, Cost: {cost}")
    return parameters

