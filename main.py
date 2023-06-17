import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, hidden_size, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size),
        'b2': np.zeros((1, output_size))
    }
    return weights


def forward_propagation(X, weights):
    hidden_layer_input = np.dot(X, weights['W1']) + weights['b1']
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights['W2']) + weights['b2']
    output_layer_output = sigmoid(output_layer_input)
    cache = {
        'Z1': hidden_layer_input,
        'A1': hidden_layer_output,
        'Z2': output_layer_input,
        'A2': output_layer_output
    }
    return output_layer_output, cache


def backward_propagation(X, y, weights, cache):
    m = X.shape[0]
    dZ2 = cache['A2'] - y
    dW2 = 1 / m * np.dot(cache['A1'].T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, weights['W2'].T) * (cache['A1'] * (1 - cache['A1']))
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0)
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    return gradients


def update_weights(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    return weights


def calculate_accuracy(predictions, y):
    accuracy = np.mean(predictions == y)
    return accuracy


def train(X, y, hidden_size, output_size, learning_rate, epochs):
    input_size = X.shape[1]
    weights = initialize_weights(input_size, hidden_size, output_size)

    costs = []
    accuracies = []

    for epoch in range(epochs):
        output, cache = forward_propagation(X, weights)
        gradients = backward_propagation(X, y, weights, cache)
        weights = update_weights(weights, gradients, learning_rate)

        predictions = predict(X, weights)
        cost = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        accuracy = calculate_accuracy(predictions, y)
        costs.append(cost)
        accuracies.append(accuracy)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: cost = {cost}, accuracy = {accuracy}")

    return weights, costs, accuracies


def predict(X, weights):
    output, _ = forward_propagation(X, weights)
    predictions = (output > 0.5).astype(int)
    return predictions


# Dane treningowe
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Ustawienia modelu
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 2000

# Trening modelu
trained_weights, costs, accuracies = train(X, y, hidden_size, output_size, learning_rate, epochs)

# Testowanie modelu
predictions = predict(X, trained_weights)
accuracy = calculate_accuracy(predictions, y)
print("Predictions:", predictions)
print("Accuracy:", accuracy)

# Wykres kosztu i dokładności
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(costs)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cost')
ax1.set_title('Cost per Epoch')

ax2.plot(accuracies)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy per Epoch')

plt.tight_layout()
plt.show()
