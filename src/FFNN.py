import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
np.random.seed(42)
class ActivationFunction:
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x
    
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        sig = ActivationFunction.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class LossFunction:
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    @staticmethod
    def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

class Neuron:
    def __init__(self, id: int):
        self.id = id
        self.output: float = 0.0
        self.grad: float = 0.0
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Neuron):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
    
    def set_output(self, value: float) -> None:
        self.output = value
        
    def get_output(self) -> float:
        return self.output

class Layer:
    def __init__(self, id: int, neurons: list[Neuron], activation: str):
        self.id: int = id
        self.neurons: list[Neuron] = neurons
        self.activation: str = activation
        self.outputs: np.ndarray = np.array([])
        
    def activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'linear':
            return ActivationFunction.linear(x)
        elif self.activation == 'relu':
            return ActivationFunction.relu(x)
        elif self.activation == 'sigmoid':
            return ActivationFunction.sigmoid(x)
        elif self.activation == 'tanh':
            return ActivationFunction.tanh(x)
        elif self.activation == 'softmax':
            return ActivationFunction.softmax(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

class FFNN:
    def __init__(self, N_layer: int, loss: str, activation: list[str], N_neuron_layer: list[int], weight_method: str = 'normal'):
        self.N_layer = N_layer
        self.loss = loss
        self.activation = activation
        self.N_neuron_layer = N_neuron_layer
        self.weight_method = weight_method
        self.neurons: dict[int, Neuron] = {}
        self.layers: list[Layer] = []
        self.weights: dict[tuple[int, int], float] = {}  # (from_neuron_id, to_neuron_id) -> weight
        self.biases: dict[int, float] = {}
        
        self._initialize_network(self.weight_method)
    def initialize_weights(self, method: str = 'normal', scale: float = 1.0) -> None:

        for i in range(self.N_layer - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            
            for neuron1 in current_layer.neurons:
                for neuron2 in next_layer.neurons:
                    
                    if method == 'zero':
                        weight = 0.0
                    elif method == 'uniform':
                        weight = np.random.uniform(-scale, scale)
                    elif method == 'normal':
                        weight = np.random.normal(0, scale)
                    else:
                        raise ValueError(f"Unknown initialization method: {method}")
                    
                    self.weights[(neuron1.id, neuron2.id)] = weight
                    # self.weights[(neuron2.id, neuron1.id)] = weight

        # Initialize biases
        self.biases: dict[int, float] = {}
        for i in range(1, self.N_layer):
            for neuron in self.layers[i].neurons:
                if method == 'zero':
                    self.biases[neuron.id] = 0.0
                elif method == 'uniform':
                    self.biases[neuron.id] = np.random.uniform(-scale, scale)
                elif method == 'normal':
                    self.biases[neuron.id] = np.random.normal(0, scale)

    def _initialize_network(self, weight_method: str):
        """Initialize the network structure"""
        neuron_id = 1
        
        # Create layers and neurons
        for layer_idx in range(self.N_layer):
            neurons_in_layer = []
            for _ in range(self.N_neuron_layer[layer_idx]):
                neuron = Neuron(neuron_id)
                self.neurons[neuron_id] = neuron
                neurons_in_layer.append(neuron)
                neuron_id += 1
            
            activation = self.activation[layer_idx] if layer_idx < len(self.activation) else 'linear'
            self.layers.append(Layer(layer_idx, neurons_in_layer, activation))
        
        # # Initialize weights (only forward connections)
        # for layer_idx in range(self.N_layer - 1):
        #     current_layer = self.layers[layer_idx]
        #     next_layer = self.layers[layer_idx + 1]
            
        #     for current_neuron in current_layer.neurons:
        #         for next_neuron in next_layer.neurons:
        #             # Initialize with random small weights
        #             self.weights[(current_neuron.id, next_neuron.id)] = np.random.randn() * 0.1
        
        # # Initialize biases (except input layer)
        # for layer_idx in range(1, self.N_layer):
        #     self.biases[layer_idx] = np.random.randn() * 0.1
        
        self.initialize_weights(method=weight_method)  # Initialize weights using He initialization
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for a batch of samples"""
        if X.shape[1] != self.N_neuron_layer[0]:
            raise ValueError("Input size doesn't match network input layer size")
        
        for i, neuron in enumerate(self.layers[0].neurons):
            neuron.output = X[:, i]
        
        for layer_idx in range(1, self.N_layer):
            prev_layer = self.layers[layer_idx - 1]
            current_layer = self.layers[layer_idx]
            
            prev_outputs = np.array([neuron.output for neuron in prev_layer.neurons]).T
            
            weighted_sum = np.zeros((X.shape[0], len(current_layer.neurons)))
            for i, current_neuron in enumerate(current_layer.neurons):
                for j, prev_neuron in enumerate(prev_layer.neurons):
                    weight = self.weights.get((prev_neuron.id, current_neuron.id), 0.0)
                    weighted_sum[:, i] += prev_outputs[:, j] * weight
                    weighted_sum[:, i] += self.biases[current_neuron.id] 

            activated = current_layer.activate(weighted_sum)
            for i, neuron in enumerate(current_layer.neurons):
                neuron.output = activated[:, i]
            current_layer.outputs = activated
        
        return self.layers[-1].outputs
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        y_pred = self.forward(X)

        if self.loss == 'mse':
            d_loss = LossFunction.mse_derivative(y, y_pred)
        elif self.loss == 'cross_entropy':
            d_loss = LossFunction.cross_entropy_derivative(y, y_pred)
        else:
            raise ValueError("Only MSE loss is implemented in this version")
        
        deltas = [np.zeros_like(layer.outputs) for layer in self.layers]
        
        output_layer = self.layers[-1]
        if output_layer.activation == 'softmax':
            deltas[-1] = y_pred - y
        else:
            activation_derivative = self._get_activation_derivative(output_layer.activation, output_layer.outputs)
            deltas[-1] = d_loss * activation_derivative
        
        for layer_idx in range(self.N_layer - 2, 0, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            activation_derivative = self._get_activation_derivative(current_layer.activation, current_layer.outputs)

            error = np.zeros_like(current_layer.outputs)

            for i, current_neuron in enumerate(current_layer.neurons):
                for j, next_neuron in enumerate(next_layer.neurons):
                    weight = self.weights.get((current_neuron.id, next_neuron.id), 0.0)
                    error[:, i] += deltas[layer_idx + 1][:, j] * weight
            
            deltas[layer_idx] = error * activation_derivative
        
        for layer_idx in range(self.N_layer - 1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            
            current_outputs = np.array([neuron.output for neuron in current_layer.neurons]).T
            
            for i, current_neuron in enumerate(current_layer.neurons):
                for j, next_neuron in enumerate(next_layer.neurons):
                    grad = np.mean(current_outputs[:, i] * deltas[layer_idx + 1][:, j])
                    self.weights[(current_neuron.id, next_neuron.id)] -= learning_rate * grad
                    self.biases[next_neuron.id] -= learning_rate * grad
                    next_neuron.grad = grad
                    
    
    def _get_activation_derivative(self, activation: str, x: np.ndarray) -> np.ndarray:
        if activation == 'linear':
            return ActivationFunction.linear_derivative(x)
        elif activation == 'relu':
            return ActivationFunction.relu_derivative(x)
        elif activation == 'sigmoid':
            return ActivationFunction.sigmoid_derivative(x)
        elif activation == 'tanh':
            return ActivationFunction.tanh_derivative(x)
        elif activation == 'softmax':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 32, verbose:bool = True) -> None:
        for epoch in range(1,epochs+1):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.backward(X_batch, y_batch, learning_rate)
            
            if verbose and epoch % 10 == 0:
                y_pred = self.forward(X)
                if self.loss == 'mse':
                    loss = LossFunction.mse(y, y_pred)
                elif self.loss == 'cross_entropy':
                    loss = LossFunction.cross_entropy(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def save(self, filename: str) -> None:
        """Save the model to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename: str) -> 'FFNN':
        """Load a model from a file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def visualize_network(self):
        G = nx.DiGraph()
        for layer in self.layers:
            for neuron in layer.neurons:
                G.add_node(neuron.id, layer=layer.id)

        for (from_id, to_id), weight in self.weights.items():
            G.add_edge(from_id, to_id, weight=weight)

        pos = nx.multipartite_layout(G, subset_key="layer")
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=12)

        edge_labels = {(u, v): f"{d['weight']:.2f}" 
                      for u, v, d in G.edges(data=True) }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.1)
        for neuron_id, bias in self.biases.items():
            plt.text(pos[neuron_id][0], pos[neuron_id][1] - 0.15, 
                    f"b={bias:.2f}", 
                    ha='center', fontsize=8, color='red')
        plt.title("Neural Network Architecture")
        plt.axis('off')
        plt.show()