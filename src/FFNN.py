import numpy as np

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
        self.build()

    def build(self):
        for i in range(self.N_layer):
            for j in range(self.N_neuron_layer[i]):
                
        

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

    def __update_weight(self):
        '''Melakukan perhitungan update bobot'''
        pass
    
    def backprop(self):
        '''Melakukan update bobot dan bias untuk hasil inferensi yang lebih baik'''
        pass

    def fit(self, learning_rate :float, batch_size: int = 32, epochs: int = 100, verbose: bool = True):
        '''Melakukan pelatihan (forward-backward) sebanyak jumlah epochs'''
        pass

    def predict():
        '''Melakukan prediksi dari hasil pelatihan model'''
        pass
    
    def load(self, nama_file: str):
        '''Mengambil data yang sudah disimpan sebelumnya'''
        pass
    
    def save(self, nama_file: str):
        '''Melakukan prediksi dari hasil pelatihan model'''
        pass
    

     