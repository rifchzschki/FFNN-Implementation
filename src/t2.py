from FFNN import FFNN
import numpy as np
from sklearn.datasets import fetch_openml

# Initialize the neural network
nn = FFNN(
    N_layer=3, 
    loss='mse', 
    activation=['linear', 'relu', 'sigmoid'], 
    N_neuron_layer=[784, 24, 1],
    weight_method='uniform'
)

# Load MNIST data
mnist = fetch_openml(name='mnist_784', version=1, as_frame=False)  # `as_frame=False` forces NumPy output
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Ensure y is 2D (required for MSE loss)
y = y.reshape(-1, 1)  # Reshape from (70000,) to (70000, 1)

print(X.shape, y.shape)  # Should show (70000, 784) (70000, 1)

# Train the network
nn.fit(X, y, epochs=100, learning_rate=0.01, batch_size=32, verbose=True)

# Predict and visualize
predictions = nn.predict(X)
nn.visualize_selected_layers([0, 1, 2])