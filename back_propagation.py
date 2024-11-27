import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Neural Network class for the backpropagation algorithm
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    # Forward propagation
    def forward(self, X):
        self.hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.output

    # Backward propagation
    def backward(self, X, y, learning_rate):
        output_delta = (y - self.output) * sigmoid_derivative(self.output)
        hidden_delta = output_delta.dot(self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(np.square(y - self.output))}")

# Convert words to numerical features
def word_to_features(word):
    vowels = "aeiou"
    length = len(word)
    num_vowels = sum(1 for letter in word if letter in vowels)
    return np.array([length, num_vowels, length - num_vowels])

# Example dataset: Finite words classification (simple binary classification)
words = ["cat", "dog", "elephant", "lion", "giraffe", "hippopotamus", "mouse", "rat"]
labels = [0, 0, 1, 0, 1, 1, 0, 0]  # 0 short words, 1 long words

# Convert words to feature vectors
X = np.array([word_to_features(word) for word in words])
y = np.array(labels).reshape(-1, 1)

# Initialize the neural network
nn = NeuralNetwork(input_size=X.shape[1], hidden_size=4, output_size=1)

# Train the neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the network with a new word
test_word = "217Y1A05C0"
prediction = nn.forward(word_to_features(test_word).reshape(1, -1))
print(f"Prediction for '{test_word}': {prediction[0][0]}")
print(f"Classified as: {'Long word' if prediction[0][0] > 0.5 else 'Short word'}")
