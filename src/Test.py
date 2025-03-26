from FFNN import ActivationFunction, Layer, Neuron, FFNN

ffnn_example = FFNN(4, "SSE",  activation=["relu","sigmoid","linear"], N_neuron_layer=[2,2,2,2])

ffnn_example.debug()


