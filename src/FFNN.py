import numpy as np
import pickle
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
        self.neighboors: set[Neuron] = set()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Neuron):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
    
    def add_neighboor(self, neuron: "Neuron") -> None:
        """Menambahkan neuron sebagai tetangga."""
        self.neighboors.add(neuron)
        neuron.neighboors.add(self) 
    
class Layer:
    def __init__(self, id: int, neurons_id: list[int]):
        self.id: int = id
        self.neurons_id: list[int] = neurons_id

class FFNN:
    def __init__(self, N_layer: int, loss:str, activation:list[str], N_neuron_layer: list[int]) -> None:
        self.N_layer: int = N_layer
        self.loss: str = loss
        self.activation: list[str] = activation
        self.N_neuron_layer: list[int] = N_neuron_layer
        self.weight: dict[tuple[Neuron, Neuron], float] = {}
        self.gradient: dict[tuple[Neuron, Neuron], float] = {}
        self.bias: list[int] = []
        self.neurons: dict[int, Neuron] = {}
        self.layers: list[Layer] = []
        self.output : list[list[float]] = [] # Diisi ketika forward propagation

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
        neuron1.add_neighboor(neuron2)

    def forward(self):
        '''Melakukan forward propagation untuk melihat hasil inferensi'''
        pass

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
                        for neighbor in self.neurons[neuron].neighboors:
                            tuple_neuron = (self.neurons[neighbor.id], self.neurons[neuron])
                            weight_updates[tuple_neuron] += np.sum(delta * xij, axis=0)
                        bias_updates[self.neurons[neuron]] += np.sum(delta, axis=0)
                else:
                    delta = ActivationFunction.sigmoid_derivative(outmatrix[current_layer - 1])
                    sum_weighted_gradient = np.zeros_like(delta)
                    
                    for neuron in layer.neurons_id:
                        for neighbor in self.neurons[neuron].neighboors:
                            if neighbor.id in self.layers[current_layer + 1].neurons_id:
                                sum_weighted_gradient[:, neuron] += self.weight[(self.neurons[neuron], self.neurons[neighbor.id])] * delta[:, neighbor]
                        delta[:, neuron] *= sum_weighted_gradient[:, neuron]
                    
                    for neuron in layer.neurons_id:
                        xij = outmatrix[current_layer - 1][:, neuron]
                        for neighbor in self.neurons[neuron].neighboors:
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

    def predict():
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
    

     