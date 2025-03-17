# Prediksi
def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    predictions = np.argmax(cache['A2'], axis=0)
    return predictions

# Menghitung biaya
def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = - (1 / m) * np.sum(Y.T * np.log(A2))
    return cost
