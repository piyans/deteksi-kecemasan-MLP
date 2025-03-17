# Prediksi
def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    predictions = np.argmax(cache['A2'], axis=0)
    return predictions

