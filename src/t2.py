import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from FFNN import FFNN

# Load MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale pixel values to [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_val_encoded = encoder.transform(y_val.reshape(-1, 1))

# Neural Network Architecture
nn = FFNN(
    N_layer=3,
    loss='cross_entropy',  # Or 'mse' if you prefer
    activation=['relu', 'relu', 'softmax'],  # Softmax for classification
    N_neuron_layer=[784, 128, 10],  # 10 output neurons for 10 digits
    weight_method='normal'
)

# Train the network
np.random.seed(42)
nn.fit(X_train, y_encoded, epochs=2, learning_rate=0.01, batch_size=64, verbose=True)

# Make predictions
y_pred_encoded = nn.predict(X_val)

# Convert predictions back to original labels
y_pred = encoder.inverse_transform(y_pred_encoded).flatten()

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Example predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_val[i]}, Predicted: {y_pred[i]}")

# Visualization (if your FFNN class has this method)
# nn.visualize_selected_layers([0, 1, 2])