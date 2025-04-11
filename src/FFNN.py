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
        self.weight_gradients = []
        self.bias_gradients = []
        self.neuron_gradients = []

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
            
            self.weight_gradients.append(np.zeros_like(w))
            self.bias_gradients.append(np.zeros_like(b))
            
        self.outputs = [np.zeros(size) for size in layer_sizes]
        self.z_values = [np.zeros(size) for size in layer_sizes[1:]]
        self.neuron_gradients = [np.zeros(size) for size in layer_sizes]
        
            
    
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
        # Store gradient for output layer
        self.neuron_gradients[-1] = delta.mean(axis=0)
        
        for i in range(self.N_layer - 2, -1, -1):
            if i < self.N_layer - 2 or not ((self.loss == 'cce' and self.activations[-1] == 'softmax') or 
                                        (self.loss == 'bce' and self.activations[-1] == 'sigmoid')):
                delta = delta * self._activate_derivative(self.z_values[i], self.activations[i])
            
            # Calculate gradients
            dW = np.dot(self.outputs[i].T, delta) / batch_size
            db = np.mean(delta, axis=0)
            
            # Store gradients
            self.weight_gradients[i] = dW
            self.bias_gradients[i] = db
            
            # Regularization
            if self.regularization:
                reg_grad = self._regularization_gradient(self.weights[i])
                dW += reg_grad / batch_size
            #Update
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                self.neuron_gradients[i] = np.mean(delta, axis=0)

        if len(self.layer_sizes) > 2:
            self.neuron_gradients[0] = np.mean(delta, axis=0)
    
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
                    reg_loss = self._regularization_loss()
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
            
    def visualize_network(self, show_weights=True, show_gradients=True):
        """Network visualization with weights and gradients"""
        G = nx.DiGraph()

        node_positions = {}
        node_count = 0
        
        for layer_idx, size in enumerate(self.layer_sizes):
            for neuron_idx in range(size):
                node_id = f"L{layer_idx}_N{neuron_idx}"
                gradient = self.neuron_gradients[layer_idx][neuron_idx] if layer_idx < len(self.neuron_gradients) else 0
                G.add_node(node_id, layer=layer_idx, gradient=gradient)
                node_positions[node_id] = (layer_idx, -neuron_idx + size/2)
                node_count += 1

        for layer_idx in range(len(self.layer_sizes) - 1):
            for i in range(self.layer_sizes[layer_idx]):
                for j in range(self.layer_sizes[layer_idx + 1]):
                    from_node = f"L{layer_idx}_N{i}"
                    to_node = f"L{layer_idx+1}_N{j}"
                    weight = self.weights[layer_idx][i, j]
                    weight_gradient = self.weight_gradients[layer_idx][i, j]
                    G.add_edge(from_node, to_node, weight=weight, gradient=weight_gradient)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Color nodes based on gradient
        if show_gradients and any(np.any(g != 0) for g in self.neuron_gradients):
            # Get all gradients flattened for normalization
            all_gradients = [abs(g) for layer_grads in self.neuron_gradients for g in layer_grads]
            max_grad = max(all_gradients) if all_gradients else 1.0
            
            # Create color map for nodes
            node_colors = []
            for node in G.nodes():
                gradient = abs(G.nodes[node].get('gradient', 0))
                intensity = min(gradient / max_grad, 1.0) if max_grad > 0 else 0
                # Use a heat map: white (small gradient) to red (large gradient)
                node_colors.append((1.0, 1.0-intensity, 1.0-intensity))
        else:
            node_colors = ['skyblue'] * len(G.nodes())
        
        nx.draw_networkx_nodes(G, node_positions, node_size=500, node_color=node_colors)

        if show_weights:
            max_weight = max(abs(w) for _, _, w in G.edges.data('weight'))
            max_grad = max(abs(g) for _, _, g in G.edges.data('gradient')) if show_gradients else 1.0
            
            for u, v, d in G.edges(data=True):
                weight = d['weight']
                width = 1 + 3 * abs(weight) / max_weight
                
                # Color based on weight sign and gradient magnitude if showing gradients
                if show_gradients and max_grad > 0:
                    gradient = abs(d.get('gradient', 0))
                    intensity = min(gradient / max_grad, 1.0) if max_grad > 0 else 0
                    # Base color on weight sign, intensity on gradient
                    color = 'darkred' if weight < 0 else 'darkgreen'
                    # Make edges more transparent for small gradients
                    alpha = 0.2 + 0.8 * intensity
                else:
                    color = 'red' if weight < 0 else 'green'
                    alpha = 1.0
                    
                nx.draw_networkx_edges(G, node_positions, edgelist=[(u, v)], 
                                    width=width, edge_color=color, arrows=True, alpha=alpha)
        else:
            nx.draw_networkx_edges(G, node_positions, arrows=True)
            
        nx.draw_networkx_labels(G, node_positions)
        
        # Add legend
        if show_gradients:
            plt.figtext(0.01, 0.01, "Node color: white (small gradient) to red (large gradient)", fontsize=9)
            plt.figtext(0.01, 0.03, "Edge opacity: transparent (small gradient) to solid (large gradient)", fontsize=9)
        
        plt.title("Neural Network Architecture with Gradients")
        plt.axis('off')
        plt.show()

    def visualize_selected_layers(self, layers_to_show, show_weights=True, show_gradients=True):
        """
        Visualize only selected layers of the network with gradient information
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
                gradient = self.neuron_gradients[layer_idx][neuron_idx] if layer_idx < len(self.neuron_gradients) else 0
                G.add_node(node_id, layer=layer_idx, gradient=gradient)
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
                        weight_gradient = self.weight_gradients[current_layer][ni, nj]
                        G.add_edge(from_node, to_node, weight=weight, gradient=weight_gradient)

        plt.figure(figsize=(12, 8))
        
        # Color nodes based on gradient
        if show_gradients and any(np.any(g != 0) for g in self.neuron_gradients):
            # Get all gradients for selected layers
            selected_gradients = []
            for layer_idx in layers_to_show:
                if layer_idx < len(self.neuron_gradients):
                    selected_gradients.extend([abs(g) for g in self.neuron_gradients[layer_idx]])
            
            max_grad = max(selected_gradients) if selected_gradients else 1.0
            
            # Create color map for nodes
            node_colors = []
            for node in G.nodes():
                gradient = abs(G.nodes[node].get('gradient', 0))
                intensity = min(gradient / max_grad, 1.0) if max_grad > 0 else 0
                # Use a heat map: white (small gradient) to red (large gradient)
                node_colors.append((1.0, 1.0-intensity, 1.0-intensity))
        else:
            node_colors = ['skyblue'] * len(G.nodes())
        
        nx.draw_networkx_nodes(G, node_positions, node_size=500, node_color=node_colors)

        if show_weights and G.edges:
            max_weight = max(abs(w) for _, _, w in G.edges.data('weight'))
            max_grad = max(abs(g) for _, _, g in G.edges.data('gradient')) if show_gradients else 1.0
            
            for u, v, d in G.edges(data=True):
                weight = d['weight']
                width = 1 + 3 * abs(weight) / max_weight
                
                # Color based on weight sign and gradient magnitude if showing gradients
                if show_gradients and max_grad > 0:
                    gradient = abs(d.get('gradient', 0))
                    intensity = min(gradient / max_grad, 1.0) if max_grad > 0 else 0
                    # Base color on weight sign, intensity on gradient
                    color = 'darkred' if weight < 0 else 'darkgreen'
                    # Make edges more transparent for small gradients
                    alpha = 0.2 + 0.8 * intensity
                else:
                    color = 'red' if weight < 0 else 'green'
                    alpha = 1.0
                    
                nx.draw_networkx_edges(G, node_positions, edgelist=[(u, v)], 
                                    width=width, edge_color=color, arrows=True, alpha=alpha)

                if show_weights:
                    label = f"{weight:.2f}"
                    if show_gradients:
                        label += f" (g:{d.get('gradient', 0):.2e})"
                    edge_label = {(u, v): label}
                    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_label, 
                                            font_size=8, label_pos=0.3)
        else:
            nx.draw_networkx_edges(G, node_positions, arrows=True)

        # Add node labels with gradients
        node_labels = {}
        for node in G.nodes():
            gradient = G.nodes[node].get('gradient', 0)
            label = node
            if show_gradients:
                label += f"\ng:{gradient:.2e}"
            node_labels[node] = label
        
        nx.draw_networkx_labels(G, node_positions, labels=node_labels)

        for layer_idx in layers_to_show:
            if layer_idx < len(self.layer_sizes):
                plt.text(layer_idx, self.layer_sizes[layer_idx]/2 + 1, 
                        f"Layer {layer_idx}\n({self.activations[layer_idx] if layer_idx < len(self.activations) else 'input'})", 
                        ha='center')
        
        # Add legend
        if show_gradients:
            plt.figtext(0.01, 0.01, "Node color: white (small gradient) to red (large gradient)", fontsize=9)
            plt.figtext(0.01, 0.03, "Edge opacity: transparent (small gradient) to solid (large gradient)", fontsize=9)
        
        plt.title(f"Neural Network Architecture (Selected Layers: {sorted(layers_to_show)}) with Gradients")
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

    def save_to_txt(self, filename):
        """
        Save the neural network parameters and gradients to a text file.
        
        Args:
            filename (str): Path to the output text file
        """
        with open(filename, 'w') as f:
            f.write("Feed-Forward Neural Network Configuration\n")
            f.write("=======================================\n\n")
            
            # architecture
            f.write(f"Architecture: {self.layer_sizes}\n")
            f.write(f"Activation functions: {self.activations}\n")
            f.write(f"Loss function: {self.loss}\n")
            f.write(f"Regularization: {self.regularization}, lambda={self.lambda_}\n\n")
            
            # Write neuron gradients for input layer
            f.write(f"Input Layer (Layer 0) Gradients:\n")
            f.write("-" * 40 + "\n")
            for i, grad in enumerate(self.neuron_gradients[0]):
                f.write(f"  Neuron {i}: {grad:.6e}\n")
            f.write("\n")
            
            # Write weights, biases and gradients for each layer
            for i in range(len(self.weights)):
                from_layer = i
                to_layer = i + 1
                
                f.write(f"Layer {from_layer} -> Layer {to_layer}\n")
                f.write("-" * 40 + "\n")
                
                # Write weights and weight gradients
                f.write(f"Weights (shape: {self.weights[i].shape}):\n")
                f.write(f"  From Layer {from_layer} neurons to Layer {to_layer} neurons:\n")
                for row_idx, row in enumerate(self.weights[i]):
                    w_row = ' '.join([f'{w:.6f}' for w in row])
                    g_row = ' '.join([f'{g:.6e}' for g in self.weight_gradients[i][row_idx]])
                    f.write(f"  Neuron {row_idx} -> {w_row}\n")
                    f.write(f"    Gradients -> {g_row}\n")
                

                f.write(f"\nBiases for Layer {to_layer} neurons:\n")
                b_vals = ' '.join([f'{b:.6f}' for b in self.biases[i]])
                g_vals = ' '.join([f'{g:.6e}' for g in self.bias_gradients[i]])
                f.write(f"  Values: {b_vals}\n")
                f.write(f"  Gradients: {g_vals}\n")
                
                f.write(f"\nNeuron Gradients for Layer {to_layer}:\n")
                for j, grad in enumerate(self.neuron_gradients[to_layer]):
                    f.write(f"  Neuron {j}: {grad:.6e}\n")
                f.write("\n")


