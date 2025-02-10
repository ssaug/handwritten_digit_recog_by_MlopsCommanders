
import matplotlib.pyplot as plt
import numpy as np
import struct


# Set MLflow experiment
#mlflow.set_experiment("MNIST_Ensemble_Classification")

# Load and preprocess the MNIST dataset
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# File paths
train_images_path = "data/train-images-idx3-ubyte/train-images.idx3-ubyte"
train_labels_path = "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
test_images_path = "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
test_labels_path = "data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"

# Load data
X_train = load_mnist_images(train_images_path).reshape(60000, -1) / 255.0
y_train = load_mnist_labels(train_labels_path)
X_test = load_mnist_images(test_images_path).reshape(10000, -1) / 255.0
y_test = load_mnist_labels(test_labels_path)

# Select a sample image from the test set
sample_image = X_train[0].reshape(28, 28)  # Reshape back to 28x28 for visualization

# Display the image
plt.imshow(sample_image, cmap="gray")
plt.title("Sample MNIST Image")
plt.axis("off")
plt.show()

print(y_train[0])
# Save to txt file
np.savetxt("./data/sample_image_features1.txt", X_train[0], fmt="%1.2f", delimiter=",", newline=",")
