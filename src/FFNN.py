import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
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
        return -2 * (y_true - y_pred)/y_true.shape[0]
    

class Neuron:
    def __init__(self, id: int):
        self.id = id
        self.neighbors: set[Neuron] = set()
        self.weight = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Neuron):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
    
    def add_neighbor(self, neuron: "Neuron") -> None:
        """Menambahkan neuron sebagai tetangga."""
        self.neighbors.add(neuron)
        neuron.neighbors.add(self)

    def print_neighbors(self):
        print(f"Daftar tetangga dari Neuron dengan ID: {self.id}")
        for neuron in self.neighbors:
            print(neuron.id)
    def set_weight(self, weight: float) -> None:
        self.weight = weight

class Layer:
    def __init__(self, id: int, neurons_id: list[int]):
        self.id: int = id
        self.neurons_id: list[int] = neurons_id

    def debug(self):
        print(f"Layer ke-{self.id}")
        for i in range (len(self.neurons_id)):
            print(f"Neuron ID pada Layer ke-{self.id} : {self.neurons_id[i]}")

class FFNN:
    def __init__(self, N_layer: int, loss:str, activation:list[str], N_neuron_layer: list[int]) -> None:
        self.N_layer: int = N_layer
        self.loss: str = loss
        self.activation: list[str] = activation
        self.N_neuron_layer: list[int] = N_neuron_layer
        self.weight: dict[tuple[Neuron, Neuron], float] = {}
        self.gradient: dict[tuple[Neuron, Neuron], float] = {}
        self.bias: list[float] = []
        self.neurons: dict[int, Neuron] = {}
        self.layers: list[Layer] = []
        self.output : list[list[float]] = [] # Diisi ketika forward propagation
        self.input: list[float] = []
        self.predicted_values: list[float] = []
    def set_input_values(self):
        input_layer = self.layers[0]
        print("Masukkan nilai input pada setiap neuron di layer input")
        for neuron_id in input_layer.neurons_id:
            input_weight = float(input(f"Masukkan nilai input pada neuron {neuron_id}: "))
            self.neurons[neuron_id].set_weight(input_weight)
            self.input.append(input_weight)

    def set_predicted_values(self):
        output_layer = self.layers[self.N_layer-1]
        print("Masukkan nilai prediksi pada setiap neuron di layer output")
        for neuron_id in output_layer.neurons_id:
            predicted_weight = float(input(f"Masukkan nilai prediksi pada neuron {neuron_id}: "))
            self.predicted_values.append(predicted_weight)
    def configure(self):
        '''konfigurasi lanjutan'''

        #masukkan layer dan neuron
        idx = 1
        assigned_edges = set()
        for i in range(self.N_layer):
            new_neuronList = []
            for j in range(self.N_neuron_layer[i]):
                newNeuron = Neuron(idx)
                new_neuronList.append(idx)
                self.neurons[idx] = newNeuron
                idx+=1

            newLayer = Layer(i, new_neuronList)
            self.layers.append(newLayer)

        #sambungkan neuron dari layer satu ke layer berikutnya

        for i in range(self.N_layer - 1):
            layer_cur = self.layers[i]
            layer_next = self.layers[i + 1]

            for j in range(len(layer_cur.neurons_id)):
                for k in range(len(layer_next.neurons_id)):
                    neuron1 = self.neurons[layer_cur.neurons_id[j]]
                    neuron2 = self.neurons[layer_next.neurons_id[k]]
                    if (neuron1.id, neuron2.id) not in assigned_edges and (neuron2.id, neuron1.id) not in assigned_edges:
                        weight_value = float(input(f"Masukkan bobot antara neuron {neuron1.id} dan neuron {neuron2.id}: "))
                        self.weight[(neuron1, neuron2)] = weight_value
                        self.weight[(neuron2, neuron1)] = weight_value
                        assigned_edges.add((neuron1.id, neuron2.id))
                    neuron1.add_neighbor(neuron2)

        #masukkan nilai bias dimulai dari hidden layer ke-1 sampai hidden layer terakhir kecuali output layer
        self.bias.append(0)

        for i in range(1,self.N_layer):
            bias_inp = float(input(f"Masukkan nilai bias pada hidden layer ke-{i}: "))
            self.bias.append(bias_inp)



    def __add_neuron(self, id: int) -> Neuron:
        """Menambahkan neuron baru ke dalam graph."""
        if id not in self.neurons:
            self.neurons[id] = Neuron(id)
        return self.neurons[id]

    def add_edge(self, id1: int, id2: int, weight: float) -> None:
        """Menambahkan hubungan antara dua neuron."""
        neuron1 = self.__add_neuron(id1)
        neuron2 = self.__add_neuron(id2)
        self.weight[tuple[neuron1,neuron2]] = weight
        neuron1.add_neighbor(neuron2)

    def forward(self):
        '''Melakukan forward propagation untuk melihat hasil inferensi'''
        layer_output = []
        for layer_idx in range(1, self.N_layer):
            prev_layer = self.layers[layer_idx-1]
            curr_layer = self.layers[layer_idx]

            for curr_neuron_id in curr_layer.neurons_id:
                unactivated_weight = 0

                curr_neuron = self.neurons[curr_neuron_id]
                for prev_neuron_id in prev_layer.neurons_id:
                    prev_neuron = self.neurons[prev_neuron_id]
                    unactivated_weight += (prev_neuron.weight * self.weight[(prev_neuron, curr_neuron)])

                unactivated_weight += self.bias[layer_idx]

                # print(f"Unactivated weight neuron ke-{curr_neuron_id}: {unactivated_weight}")
                activated_weight = 0
                activation_func = self.activation[layer_idx]
                if activation_func == 'linear':
                    activated_weight = ActivationFunction.linear(unactivated_weight)
                elif activation_func == 'relu':
                    activated_weight = ActivationFunction.relu(unactivated_weight)
                elif activation_func == 'sigmoid':
                    activated_weight = ActivationFunction.sigmoid(unactivated_weight)
                elif activation_func == 'tanh':
                    activated_weight = ActivationFunction.tanh(unactivated_weight)
                elif activation_func == 'softmax':
                    activated_weight = ActivationFunction.softmax(unactivated_weight)
                else:
                    raise ValueError("Fungsi aktivasi tidak valid!")
                # print(f"Activated weight neuron ke-{curr_neuron_id}: {activated_weight}")
                curr_neuron.set_weight(float(activated_weight))
                layer_output.append(float(activated_weight))

        self.output.append(layer_output)




    def __update_weight(self, tuple_neuron: tuple[Neuron, Neuron], weight_update: float) -> None:
        '''Melakukan perhitungan update bobot'''
        self.weight[tuple_neuron] -= weight_update
    
    def backprop(self, X: np.ndarray, y: np.ndarray, learning_rate: float, batch_size: int = 32):
        '''Melakukan update bobot dan bias setiap batch size tercapai dengan operasi matriks'''
        num_samples = X.shape[0]
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            X_batch = X[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            
            y_pred = self.forward(X_batch)
            
            if self.loss == 'mse':
                loss_derivative = LossFunction.mse_derivative(y_batch, y_pred)
            else:
                raise ValueError(f"Loss function {self.loss} is not supported")
            
            weight_updates = {key: np.zeros_like(value) for key, value in self.weight.items()}
            bias_updates = {key: np.zeros_like(value) for key, value in self.bias.items()}
            
            current_layer = self.N_layer - 1
            outmatrix = np.array(self.output)
            
            for layer in reversed(self.layers):
                if current_layer == self.N_layer - 1:
                    delta = loss_derivative
                    if self.activation[current_layer] == 'sigmoid':
                        delta *= ActivationFunction.sigmoid_derivative(y_pred)
                    
                    for neuron in layer.neurons_id:
                        xij = outmatrix[current_layer - 1][:, neuron]
                        for neighbor in self.neurons[neuron].neighbors:
                            tuple_neuron = (self.neurons[neighbor.id], self.neurons[neuron])
                            weight_updates[tuple_neuron] += np.sum(delta * xij, axis=0)
                        bias_updates[self.neurons[neuron]] += np.sum(delta, axis=0)
                elif current_layer > 0:
                    delta = ActivationFunction.sigmoid_derivative(outmatrix[current_layer - 1])
                    sum_weighted_gradient = np.zeros_like(delta)
                    
                    for neuron in layer.neurons_id:
                        for neighbor in self.neurons[neuron].neighbors:
                            if neighbor.id in self.layers[current_layer + 1].neurons_id:
                                sum_weighted_gradient[:, neuron] += self.weight[(self.neurons[neuron], self.neurons[neighbor.id])] * delta[:, neighbor]
                        delta[:, neuron] *= sum_weighted_gradient[:, neuron]
                    
                    for neuron in layer.neurons_id:
                        xij = outmatrix[current_layer - 1][:, neuron]
                        for neighbor in self.neurons[neuron].neighbors:
                            if neighbor.id in self.layers[current_layer - 1].neurons_id:
                                tuple_neuron = (self.neurons[neighbor.id], self.neurons[neuron])
                                weight_updates[tuple_neuron] += np.sum(delta * xij, axis=0)
                        bias_updates[self.neurons[neuron]] += np.sum(delta, axis=0)
                
                current_layer -= 1
            
            for tuple_neuron, grad in weight_updates.items():
                self.__update_weight(tuple_neuron, learning_rate * grad / batch_size)
            for neuron, grad in bias_updates.items():
                self.__update_bias(neuron, learning_rate * grad / batch_size)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 32):
        '''Melatih jaringan saraf dengan dataset dalam beberapa epoch'''
        for epoch in range(epochs):
            self.backprop(X, y, learning_rate, batch_size)
            print(f"Epoch {epoch + 1}/{epochs} selesai")

    def predict(self):
        '''Melakukan prediksi dari hasil pelatihan model'''
        pass
    
    def load(self, nama_file: str):
        '''Mengambil data yang sudah disimpan sebelumnya'''
        with open(nama_file, 'rb') as f:
            return pickle.load(f)
    
    def save(self, nama_file: str):
        '''Melakukan prediksi dari hasil pelatihan model'''
        with open(nama_file, 'wb') as f:
            pickle.dump(self, f)

    def debug(self):
        print("Banyak Layer: ", self.N_layer)
        print("Loss Function yang digunakan: ", self.loss)
        print("Activation Function yang digunakan: ")
        for i in range(len(self.activation)):
            print(f"Activation function yang digunakan layer ke-{i} : {self.activation[i]}")
        for i in range(len(self.N_neuron_layer)):
            print(f"Banyak neuron pada layer ke-{i} : {self.N_neuron_layer[i]}")
        for layer in self.layers:
            layer.debug()
        for i in range (1, len(self.neurons)+1):
            self.neurons[i].print_neighbors()
        for i in range (1, len(self.bias)):
            print(f"Bobot bias pada hidden layer ke-{i} : {self.bias[i]}")
        print("\n=== Bobot antar Neuron ===")
        for (neuron1, neuron2), weight in self.weight.items():
            print(f"Neuron {neuron1.id} ↔ Neuron {neuron2.id} : {weight}")

        print("\n=== Bobot input ===")
        input_layer = self.layers[0]

        for neuron_id in input_layer.neurons_id:
            neuron = self.neurons[neuron_id]
            print(f"Neuron {neuron.id} : {neuron.weight}")

        print("\n=== Bobot Setiap Neuron ===")
        for layer_idx, layer in enumerate(self.layers):
            print(f"\nLayer ke-{layer_idx}:")
            for neuron_id in layer.neurons_id:
                neuron = self.neurons[neuron_id]
                print(f"Neuron {neuron.id} : {neuron.weight}")
        print("\n=== Bobot prediksi ===")
        output_layer = self.layers[self.N_layer-1]
        idx = 0
        for neuron_id in output_layer.neurons_id:
            neuron = self.neurons[neuron_id]
            print(f"Neuron {neuron.id} : {neuron.weight}, nilai prediksi: {self.predicted_values[idx]}")
            idx+=1
    def visualize_network(self):
        
        G = nx.DiGraph()
        
        for layer_id, layer in enumerate(self.layers):
            for neuron_id in layer.neurons_id:
                G.add_node(neuron_id, layer=layer_id)
        
        for (neuron1, neuron2), weight in self.weight.items():
            gradient = self.gradient.get((neuron1, neuron2), 0)
            G.add_edge(neuron1.id, neuron2.id, weight=weight, gradient=gradient)
        
        pos = nx.multipartite_layout(G, subset_key="layer")
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        
        edge_labels = {(u, v): f"w:{d['weight']:.2f}\ng:{d['gradient']:.2f}" 
                      for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        plt.title("Struktur FFNN")
        plt.axis('off')
        plt.show()
    
    def plot_weight_distribution(self, layers_to_plot: list[int]):

        plt.figure(figsize=(12, 6))
        
        for layer_idx in layers_to_plot:
            if layer_idx < 0 or layer_idx >= len(self.layers) - 1:
                continue
                
            weights = []
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            for neuron1_id in current_layer.neurons_id:
                for neuron2_id in next_layer.neurons_id:
                    neuron1 = self.neurons[neuron1_id]
                    neuron2 = self.neurons[neuron2_id]
                    if (neuron1, neuron2) in self.weight:
                        weights.append(self.weight[(neuron1, neuron2)])
            
            if weights:
                plt.hist(weights, bins=30, alpha=0.5, 
                         label=f'Layer {layer_idx} to {layer_idx + 1}')
        
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title('Weight Distribution by Layer')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_gradient_distribution(self, layers_to_plot: list[int]):
        plt.figure(figsize=(12, 6))
        
        for layer_idx in layers_to_plot:
            if layer_idx < 0 or layer_idx >= len(self.layers) - 1:
                print(f"Warning: Layer {layer_idx} is out of range. Skipping.")
                continue
                
            gradients = []
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            for neuron1_id in current_layer.neurons_id:
                for neuron2_id in next_layer.neurons_id:
                    neuron1 = self.neurons[neuron1_id]
                    neuron2 = self.neurons[neuron2_id]
                    if (neuron1, neuron2) in self.gradient:
                        gradients.append(self.gradient[(neuron1, neuron2)])
            
            if gradients:
                plt.hist(gradients, bins=30, alpha=0.5, 
                         label=f'Layer {layer_idx} to {layer_idx + 1}')
        
        plt.xlabel('Gradient Value')
        plt.ylabel('Frequency')
        plt.title('Gradient Distribution by Layer')
        plt.legend()
        plt.grid(True)
        plt.show()