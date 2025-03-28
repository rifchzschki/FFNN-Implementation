from FFNN import FFNN
import numpy as np
nn = FFNN(N_layer=3, 
          loss='mse', 
          activation=['linear', 'relu', 'sigmoid'], 
          N_neuron_layer=[2, 4, 1], weight_method = 'uniform')
np.random.seed(41)
X = np.random.rand(100, 2)
y = np.random.rand(100, 1)

nn.fit(X, y, epochs=100, learning_rate=0.01, batch_size=16, verbose=True)

predictions = nn.predict(X)
nn.visualize_selected_layers([0, 1])