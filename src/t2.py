from FFNN import FFNN
import numpy as np
# Example usage:
# Create a network with 2 input units, one hidden layer with 4 units, and 1 output unit
nn = FFNN(N_layer=3, 
          loss='mse', 
          activation=['linear', 'relu', 'sigmoid'], 
          N_neuron_layer=[2, 4, 1], weight_method = 'uniform')
np.random.seed(41)
# Generate some dummy data
X = np.random.rand(100, 2)
y = np.random.rand(100, 1)

# Train the network
nn.fit(X, y, epochs=100, learning_rate=0.01, batch_size=16, verbose=True)

# Make predictions
predictions = nn.predict(X)

# Visualize the network
nn.visualize_network()