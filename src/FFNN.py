import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time

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
        max_x = np.amax(x,1).reshape(x.shape[0],1)
        e_x = np.exp(x-max_x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunction.softmax(x)
        return s * (1 - s)

    @staticmethod
    def softmax_derivative_jacobian(x: np.ndarray)-> np.ndarray:
        s = ActivationFunction.softmax(x)
        a = np.eye(s.shape[-1])
        temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
        temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
        temp1 = np.einsum('ij,jk->ijk', s, a)
        temp2 = np.einsum('ij,ik->ijk', s, s)
        return temp1-temp2

class LossFunction:
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * np.mean(y_true - y_pred)

    @staticmethod
    def bce(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

    def bce_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean((y_pred-y_true)/(y_pred*(1-y_pred)),axis=0)

    @staticmethod
    def cce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def cce_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true/y_pred)/y_true.shape[0]



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
        self.layers[0].outputs = X 
        assert not np.isnan(X).any(), "Input contains NaN!"
        if np.isnan(self.layers[0].outputs).any():
            raise ValueError("NaN detected in layer 0 outputs.")
        for layer_idx in range(1, self.N_layer):
            prev_layer = self.layers[layer_idx - 1]
            current_layer = self.layers[layer_idx]
            prev_outputs = prev_layer.outputs
            if np.isnan(prev_outputs).any():
                raise ValueError(f"NaN detected in layer {layer_idx - 1} outputs.")
            weight_matrix = np.array([
                [self.weights.get((prev_neuron.id, current_neuron.id), 0.0) for current_neuron in current_layer.neurons]
                for prev_neuron in prev_layer.neurons
            ])
            bias_vector = np.array([self.biases[neuron.id] for neuron in current_layer.neurons])
            weighted_sum = np.dot(prev_outputs, weight_matrix) + bias_vector
            if np.isnan(weighted_sum).any():
                raise ValueError(f"NaN detected in layer {layer_idx} weighted sum.")
            activated = current_layer.activate(weighted_sum)
            current_layer.outputs = activated

        return self.layers[-1].outputs
    

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        y_pred = self.forward(X)

        if self.loss == 'mse':
            d_loss = LossFunction.mse_derivative(y, y_pred)
        elif self.loss == 'cce':
            d_loss = LossFunction.cce_derivative(y, y_pred)
        elif self.loss == 'bce':
            d_loss = LossFunction.bce_derivative(y, y_pred)
        else:
            raise ValueError("Fungsi loss tidak diimplementasi")
        deltas = [np.zeros_like(layer.outputs) for layer in self.layers]

        output_layer = self.layers[-1]
        activation_derivative = self._get_activation_derivative(output_layer.activation, output_layer.outputs)
        deltas[-1] = d_loss * activation_derivative
        for i, neuron in enumerate(output_layer.neurons):
            neuron.grad = np.mean(deltas[-1][:, i])
        for layer_idx in range(self.N_layer - 2, 0, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            activation_derivative = self._get_activation_derivative(current_layer.activation, current_layer.outputs)
            weight_matrix = np.array([
                [self.weights.get((current_neuron.id, next_neuron.id), 0.0) for next_neuron in next_layer.neurons]
                for current_neuron in current_layer.neurons
            ])
            deltas[layer_idx] = np.dot(deltas[layer_idx + 1], weight_matrix.T) * activation_derivative
            for i, neuron in enumerate(current_layer.neurons):
                neuron.grad = np.mean(deltas[layer_idx][:, i]) 
        # Update bobot dan bias
        for layer_idx in range(self.N_layer - 1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            current_outputs = current_layer.outputs

            # Update bobot menggunakan operasi vektor
            weight_updates = learning_rate * np.dot(current_outputs.T, deltas[layer_idx + 1])
            weight_updates = np.clip(weight_updates, -10000.0, 10000.0)
            current_ids = [neuron.id for neuron in current_layer.neurons]
            next_ids = [neuron.id for neuron in next_layer.neurons]

            for i, current_id in enumerate(current_ids):
                for j, next_id in enumerate(next_ids):
                    self.weights[(current_id, next_id)] -= weight_updates[i, j]

            bias_updates = learning_rate * np.mean(deltas[layer_idx + 1], axis=0)
            for j, next_id in enumerate(next_ids):
                self.biases[next_id] -= bias_updates[j]
                    
    
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
            return ActivationFunction.softmax_derivative(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 32, verbose: bool = True) -> None:
        for epoch in range(1, epochs + 1):
            start = time.time()
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            if verbose:
                batch_range = tqdm(range(0, X.shape[0], batch_size), 
                                desc=f"Epoch {epoch}", 
                                leave=False, 
                                unit="batch")
            else:
                batch_range = range(0, X.shape[0], batch_size)

            for i in batch_range:
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch, learning_rate)

            if verbose:
                y_pred = self.forward(X)
                if self.loss == 'mse':
                    loss = LossFunction.mse(y, y_pred)
                elif self.loss == 'cce':
                    loss = LossFunction.cce(y, y_pred)
                elif self.loss == 'bce':
                    loss = LossFunction.bce(y, y_pred)
                print(f"\nEpoch {epoch}, Loss: {loss:.4f}, time: {time.time()-start}")
    
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
    
    def visualize_network(self, show_weights=True, show_biases=True, show_gradients=True):
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

        if show_weights:
            edge_labels = {(u, v): f"{d['weight']:.2f}" 
                        for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=8, label_pos=0.1)

        if show_biases or show_gradients:
            for neuron_id, bias in self.biases.items():
                label_parts = []
                if show_biases:
                    label_parts.append(f"b={bias:.2f}")
                if show_gradients:
                    for layer in self.layers:
                        for neuron in layer.neurons:
                            if neuron.id == neuron_id:
                                label_parts.append(f"gradient={neuron.grad:.2f}")
                                break
                label = "\n".join(label_parts)
                plt.text(pos[neuron_id][0], pos[neuron_id][1] - 0.15, 
                        label, 
                        ha='center', fontsize=8, color='red')

        plt.title("Neural Network Architecture")
        plt.axis('off')
        plt.show()

    def visualize_selected_layers(self, layers_to_show, show_weights=True, show_biases=True, show_gradients=True):
        G = nx.DiGraph()
        for layer in self.layers:
            if layer.id in layers_to_show:
                for neuron in layer.neurons:
                    G.add_node(neuron.id, layer=layer.id)
        for (from_id, to_id), weight in self.weights.items():
            from_layer = None
            to_layer = None

            for layer in self.layers:
                for neuron in layer.neurons:
                    if neuron.id == from_id:
                        from_layer = layer.id
                    if neuron.id == to_id:
                        to_layer = layer.id
            
            if from_layer in layers_to_show and to_layer in layers_to_show:
                G.add_edge(from_id, to_id, weight=weight)

        pos = nx.multipartite_layout(G, subset_key="layer")
        plt.figure(figsize=(12, 8))
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=12)
        if show_weights:
            edge_labels = {(u, v): f"{d['weight']:.2f}" 
                        for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=8, label_pos=0.1)

        if show_biases or show_gradients:
            for neuron_id, bias in self.biases.items():
                if neuron_id in G.nodes:
                    label_parts = []
                    if show_biases:
                        label_parts.append(f"b={bias:.2f}")
                    if show_gradients:
                        for layer in self.layers:
                            for neuron in layer.neurons:
                                if neuron.id == neuron_id:
                                    label_parts.append(f"gradient={neuron.grad:.2f}")
                                    break
                    label = "\n".join(label_parts)
                    plt.text(pos[neuron_id][0], pos[neuron_id][1] - 0.15, 
                            label, 
                            ha='center', fontsize=8, color='red')

        plt.title(f"Neural Network Architecture (Layers: {sorted(layers_to_show)})")
        plt.axis('off')
        plt.show()
    def visualize_network_simple(self, show_weights=True, show_biases=True, show_gradients=True):
        print("="*50)
        print("NEURAL NETWORK ARCHITECTURE")
        print("="*50)
        
        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i+1} ({layer.id}):")
            print("-"*20)
            
            for neuron in layer.neurons:
                print(f"Neuron {neuron.id}:")
                
                if show_biases and neuron.id in self.biases:
                    print(f"  Bias: {self.biases[neuron.id]:.4f}")
                
                if show_gradients:
                    print(f"  Gradient: {neuron.grad:.4f}")
        
        if show_weights:
            print("\nWEIGHTS:")
            print("-"*20)
            for (from_id, to_id), weight in self.weights.items():
                print(f"Connection {from_id} -> {to_id}: {weight:.4f}")
        
        print("\n" + "="*50)