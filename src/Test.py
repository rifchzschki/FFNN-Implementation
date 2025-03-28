from FFNN import ActivationFunction, Layer, Neuron, FFNN

#input banyak layer termasuk input layer dan output layer

layer_count = int(input("Banyak layer (termasuk input dan output layer): "))
if(layer_count < 2):
    raise ValueError("Banyak layer yang diinput harus minimal 2 karena sudah termasuk input layer dan output layer!")

#input banyak neuron di setiap layer
neuron_count_per_layer = []

for i in range(layer_count):
    outStr = "Masukkan banyak neuron di "
    if(i == 0):
        outStr = outStr + "input layer"
    elif (i == layer_count - 1):
        outStr = outStr + "output layer"
    else :
        outStr = outStr + f"hidden layer ke-{i}"
    outStr = outStr + ": "
    neuron_count = int(input(outStr))
    neuron_count_per_layer.append(neuron_count)
    if neuron_count < 1:
        raise ValueError("Banyak neuron invalid, harus input bilangan bulat > 0")
#input urutan fungsi aktivasi dari hidden layer pertama sampai ke layer output
activation_func_sequence = []
activation_func_sequence.append("")
for i in range(1,layer_count):
    outStr = "Masukkan jenis fungsi aktivasi untuk "
    if(i == layer_count - 1):
        outStr = outStr + "output layer "
    else:
        outStr = outStr + f"hidden layer ke-{i} "
    outStr = outStr + "(case insensitive!): "
    valid_activations = ["relu", "tanh", "sigmoid", "linear", "softmax"]
    print("Daftar fungsi aktivasi yang valid: ")
    print("1. ReLU")
    print("2. Sigmoid")
    print("3. Linear")
    print("4. Tangen Hiperbolik")
    print("5. Softmax")
    activation_input = input(outStr).lower()
    activation_func_sequence.append(activation_input)
    if activation_input not in valid_activations:
        raise ValueError("Activation function yang dimasukkan invalid")

#input jenis loss function
print("Daftar fungsi loss yang valid: ")
print("1. MSE / Mean Squared Error (harap input sebagai MSE)")
print("2. CCE / Categorical Cross Entropy (harap input sebagai CCE)")
print("3. BCE / Binary Cross Entropy (harap input sebagai BCE)")

outStr = "Masukkan jenis loss function untuk (case insensitive!): "
valid_losses = ["mse", "cce", "bce"]

loss_input = input(outStr).lower()

if loss_input not in valid_losses:
    raise ValueError("Loss function yang dimasukkan invalid")

ffnn_run = FFNN(N_layer=layer_count, loss=loss_input, activation=activation_func_sequence, N_neuron_layer=neuron_count_per_layer)

ffnn_run.configure()
ffnn_run.set_input_values()
ffnn_run.set_predicted_values()
ffnn_run.debug()