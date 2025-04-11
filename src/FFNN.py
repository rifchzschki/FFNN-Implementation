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
        x = np.clip(x, -500, 500)
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
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunction.softmax(x)
        return s * (1 - s)

class LossFunction:
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred) / y_true.shape[0]

    @staticmethod
    def bce(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

    @staticmethod
    def bce_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

    @staticmethod
    def cce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def cce_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) / y_true.shape[0]

class FFNN:
    def __init__(self, layer_sizes, activations, loss='mse', weight_method='normal', regularization=None, lambda_=0.01):
        """
        Initialize a feed-forward neural network with fully connected layers.
        
        Args:
            layer_sizes: List of integers representing number of neurons in each layer
            activations: List of activation functions for each layer
            loss: Loss function to use
            weight_method: Method for weight initialization
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss = loss
        self.N_layer = len(layer_sizes)
        self.regularization=regularization
        self.lambda_ = lambda_
        self.weights = []
        self.biases = []
        self.outputs = [] 
        self.z_values = [] 

        for i in range(len(layer_sizes) - 1):
            if weight_method == 'zero':
                w = np.zeros((layer_sizes[i], layer_sizes[i+1]))
            elif weight_method == 'uniform':
                w = np.random.uniform(-0.1, 0.1, (layer_sizes[i], layer_sizes[i+1]))
            elif weight_method == 'normal':
                if activations[i] == 'relu':
                    scale = np.sqrt(2.0 / layer_sizes[i])
                else:
                    scale = np.sqrt(1.0 / layer_sizes[i])
                w = np.random.normal(0, scale, (layer_sizes[i], layer_sizes[i+1]))
            else:
                raise ValueError(f"Unknown initialization method: {weight_method}")
                
            b = np.zeros(layer_sizes[i+1])
            
            self.weights.append(w)
            self.biases.append(b)
            
        self.outputs = [np.zeros(size) for size in layer_sizes]
        self.z_values = [np.zeros(size) for size in layer_sizes[1:]]
        
            
    
    def _activate(self, x, activation):
        """Apply activation function to input"""
        if activation == 'linear':
            return ActivationFunction.linear(x)
        elif activation == 'relu':
            return ActivationFunction.relu(x)
        elif activation == 'sigmoid':
            return ActivationFunction.sigmoid(x)
        elif activation == 'tanh':
            return ActivationFunction.tanh(x)
        elif activation == 'softmax':
            return ActivationFunction.softmax(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def _activate_derivative(self, x, activation):
        """Apply derivative of activation function"""
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
        
    def _regularization_loss(self):
        """Calculate regularization loss"""
        if self.regularization is None:
            return 0.0
        reg_loss = 0.0
        for w in self.weights:
            if self.regularization == 'L2':
                reg_loss += np.sum(w ** 2)
            elif self.regularization == 'L1':
                reg_loss += np.sum(np.abs(w))

        return self.lambda_ * reg_loss
    
    def _regularization_gradient(self, weight_matrix):
        """Calculate regularization gradient"""
        if self.regularization is None:
            return 0.0
        if self.regularization == 'L1':
            return self.lambda_ * np.sign(weight_matrix)
        elif self.regularization == 'L2':
            return self.lambda_ * weight_matrix * 2
        return 0.0
    
    def forward(self, X):
        """Forward pass through the network"""
        self.outputs[0] = X
        
        # Forward through each layer
        for i in range(self.N_layer - 1):
            # z = x·W + b
            z = np.dot(self.outputs[i], self.weights[i]) + self.biases[i]
            self.z_values[i] = z
            
            # Apply activation
            self.outputs[i+1] = self._activate(z, self.activations[i])
        
        return self.outputs[-1]
    
    def backward(self, X, y, learning_rate):
        """Backward pass through the network"""
        batch_size = X.shape[0]
        
        y_pred = self.forward(X)

        if self.loss == 'mse':
            d_loss = y_pred - y 
        elif self.loss == 'cce':
            if self.activations[-1] == 'softmax':
                d_loss = y_pred - y
            else:
                d_loss = LossFunction.cce_derivative(y, y_pred)
        elif self.loss == 'bce':
            if self.activations[-1] == 'sigmoid':
                d_loss = y_pred - y
            else:
                d_loss = LossFunction.bce_derivative(y, y_pred)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
        delta = d_loss
        
        for i in range(self.N_layer - 2, -1, -1):
            if i < self.N_layer - 2 or not ((self.loss == 'cce' and self.activations[-1] == 'softmax') or 
                                          (self.loss == 'bce' and self.activations[-1] == 'sigmoid')):
                delta = delta * self._activate_derivative(self.z_values[i], self.activations[i])
            
            # Calculate gradients
            dW = np.dot(self.outputs[i].T, delta) / batch_size
            db = np.mean(delta, axis=0)
            # Regularization
            if self.regularization:
                reg_grad = self._regularization_gradient(self.weights[i])
                dW += reg_grad / batch_size
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Propagate error to previous layer (if not the input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
    
    def fit(self, X, y, epochs=10, learning_rate=0.01, batch_size=32, verbose=True, track_loss=True):
        """
        Train the neural network
        """
        n_samples = X.shape[0]
        loss_history = [] if track_loss else None
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            if verbose:
                batch_range = tqdm(range(0, n_samples, batch_size),
                                desc=f"Epoch {epoch}",
                                leave=False,
                                unit="batch")
            else:
                batch_range = range(0, n_samples, batch_size)

            for i in batch_range:
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.backward(X_batch, y_batch, learning_rate)

            if verbose or track_loss:
                y_pred = self.forward(X)
                
                if self.loss == 'mse':
                    loss = LossFunction.mse(y, y_pred)
                elif self.loss == 'cce':
                    loss = LossFunction.cce(y, y_pred)
                elif self.loss == 'bce':
                    loss = LossFunction.bce(y, y_pred)

                # Regularization loss
                if self.regularization:
                    reg_loss = self._calculate_regularization_loss()
                    total_loss = loss + reg_loss
                    if track_loss:
                        loss_history.append(total_loss)
                    if verbose:
                        print(f"\nEpoch {epoch}, Loss: {loss:.4f}, Reg Loss: {reg_loss:.4f}, "
                              f"Total: {total_loss:.4f}, Time: {time.time() - start_time:.2f}s")
                else:
                    if track_loss:
                        loss_history.append(loss)
                    if verbose:
                        print(f"\nEpoch {epoch}, Loss: {loss:.4f}, Time: {time.time() - start_time:.2f}s")
        
        return loss_history
    
    def predict(self, X):
        """Make predictions for input data"""
        return self.forward(X)
    
    def save(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
            
    def visualize_network(self, show_weights=True):
        """Simple network visualization with weights"""
        G = nx.DiGraph()

        node_positions = {}
        node_count = 0
        
        for layer_idx, size in enumerate(self.layer_sizes):
            for neuron_idx in range(size):
                node_id = f"L{layer_idx}_N{neuron_idx}"
                G.add_node(node_id, layer=layer_idx)
                node_positions[node_id] = (layer_idx, -neuron_idx + size/2)
                node_count += 1

        for layer_idx in range(len(self.layer_sizes) - 1):
            for i in range(self.layer_sizes[layer_idx]):
                for j in range(self.layer_sizes[layer_idx + 1]):
                    from_node = f"L{layer_idx}_N{i}"
                    to_node = f"L{layer_idx+1}_N{j}"
                    weight = self.weights[layer_idx][i, j]
                    G.add_edge(from_node, to_node, weight=weight)
        
        # Plot
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, node_positions, node_size=500, node_color='skyblue')

        if show_weights:
            max_weight = max(abs(w) for _, _, w in G.edges.data('weight'))
            for u, v, d in G.edges(data=True):
                weight = d['weight']
                width = 1 + 3 * abs(weight) / max_weight
                color = 'red' if weight < 0 else 'green'
                nx.draw_networkx_edges(G, node_positions, edgelist=[(u, v)], 
                                      width=width, edge_color=color, arrows=True)
        else:
            nx.draw_networkx_edges(G, node_positions, arrows=True)
            
        nx.draw_networkx_labels(G, node_positions)
        plt.title("Neural Network Architecture")
        plt.axis('off')
        plt.show()

    def visualize_selected_layers(self, layers_to_show, show_weights=True):
        """
        Visualize only selected layers of the network
        """
        G = nx.DiGraph()

        node_positions = {}
        for layer_idx in layers_to_show:
            if layer_idx >= len(self.layer_sizes):
                print(f"Warning: Layer {layer_idx} does not exist in the network")
                continue
                
            size = self.layer_sizes[layer_idx]
            for neuron_idx in range(size):
                node_id = f"L{layer_idx}_N{neuron_idx}"
                G.add_node(node_id, layer=layer_idx)
                node_positions[node_id] = (layer_idx, -neuron_idx + size/2)

        for i in range(len(layers_to_show) - 1):
            current_layer = layers_to_show[i]
            next_layer = layers_to_show[i + 1]

            if next_layer - current_layer == 1:
                for ni in range(self.layer_sizes[current_layer]):
                    for nj in range(self.layer_sizes[next_layer]):
                        from_node = f"L{current_layer}_N{ni}"
                        to_node = f"L{next_layer}_N{nj}"
                        weight = self.weights[current_layer][ni, nj]
                        G.add_edge(from_node, to_node, weight=weight)

        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, node_positions, node_size=500, node_color='skyblue')

        if show_weights:
            if G.edges:
                max_weight = max(abs(w) for _, _, w in G.edges.data('weight'))
                for u, v, d in G.edges(data=True):
                    weight = d['weight']
                    width = 1 + 3 * abs(weight) / max_weight
                    color = 'red' if weight < 0 else 'green'
                    nx.draw_networkx_edges(G, node_positions, edgelist=[(u, v)], 
                                        width=width, edge_color=color, arrows=True)

                    if show_weights:
                        edge_label = {(u, v): f"{weight:.2f}"}
                        nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_label, 
                                                font_size=8, label_pos=0.3)
        else:
            nx.draw_networkx_edges(G, node_positions, arrows=True)

        nx.draw_networkx_labels(G, node_positions)

        for layer_idx in layers_to_show:
            if layer_idx < len(self.layer_sizes):
                plt.text(layer_idx, self.layer_sizes[layer_idx]/2 + 1, 
                        f"Layer {layer_idx}\n({self.activations[layer_idx] if layer_idx < len(self.activations) else 'input'})", 
                        ha='center')
        
        plt.title(f"Neural Network Architecture (Selected Layers: {sorted(layers_to_show)})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    def plot_loss(self, loss_history, title="Training Loss", figsize=(10, 6)):
        """
        Plot the loss history
        """
        plt.figure(figsize=figsize)
        plt.plot(loss_history, 'b-')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



