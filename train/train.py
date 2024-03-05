from keras.datasets import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_row = np.reshape(x_train, (-1, 28*28))
x_test_row = np.reshape(x_test, (-1, 28*28))

# Train MLP network
mlp = MLPClassifier(hidden_layer_sizes = (512,), alpha = 0.001)
mlp.fit(x_train_row / 255., y_train)
accuracy_score(y_test, mlp.predict(x_test_row / 255.))

# Save weights and biases to data file
A = np.zeros((784, 512, 4), dtype = np.double, order = 'F')
A[:, :, 0] = mlp.coefs_[0]
A[0:512, 0:10, 1] = mlp.coefs_[1]
A[0:512, 0, 2] = mlp.intercepts_[0]
A[0:10,  0, 3] = mlp.intercepts_[1]

with open("mnist.data", "wb") as binary_file:
    binary_file.write(A.tobytes('F'))
