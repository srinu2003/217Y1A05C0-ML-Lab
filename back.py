import numpy as np
# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Neural Network class for the backpropagation algorithm
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):  # Corrected constructor name
        # Initialize weights with random values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Weights initialization
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        # Bias initialization
        self.bias_hidden = np.random.rand(self.hidden_size)
        self.bias_output = np.random.rand(self.output_size)
    # Forward propagation
    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output
    # Backward propagation
    def backward(self, X, y, learning_rate):
        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        # Update the weights and biases using the gradients
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate  # Fixed assignment
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f"Epoch {epoch}, Loss: {loss}")
# Convert words to numerical features
def word_to_features(word):
    # Example feature extraction: [word length, number of vowels, number of consonants]
    vowels = "aeiou"
    length = len(word)
    num_vowels = sum(1 for letter in word if letter in vowels)
    num_consonants = length - num_vowels
    return np.array([length, num_vowels, num_consonants])

# Example dataset: Finite words classification (simple binary classification)
words = ["cat", "dog", "elephant", "lion", "giraffe", "hippopotamus", "mouse", "rat"]
labels = [0, 0, 1, 0, 1, 1, 0, 0]  # 0 short words, 1 long words
# Convert words to feature vectors
X = np.array([word_to_features(word) for word in words])
# Convert labels to a numpy array
y = np.array(labels).reshape(-1, 1)

# Initialize the neural network
input_size = X.shape[1]  # Number of features (3 in our case)
hidden_size = 4  # Arbitrary choice for the hidden layer size
output_size = 1  # Binary classification (short or long)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the network with a new word
test_word = "217Y1A05C0"
test_input = word_to_features(test_word).reshape(1, -1)
prediction = nn.forward(test_input)
print(f"Prediction for '{test_word}': {prediction[0][0]}")
print(f"Classified as: {'Long word' if prediction[0][0] > 0.5 else 'Short word'}")
