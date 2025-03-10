import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

def preprocess_image(image_path):
    """
    Loads an image, resizes it to 28x28, converts to grayscale, normalizes, and flattens.
    Returns the feature vector as a NumPy array.
    """
    logging.info(f"Starting preprocessing for image: {image_path}")

    # Load image
    orig_image = Image.open(image_path)
    logging.info("Image loaded successfully")

    # Convert to grayscale (if not already)
    img = orig_image.convert("L")
    logging.info("Image converted to grayscale")

    # Resize to 28x28
    img = img.resize((28, 28))
    logging.info("Image resized to 28x28")

    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float32)
    logging.info("Image converted to NumPy array")

    # Normalize pixel values (0 to 1)
    img_array = img_array / 255.0
    logging.info("Image normalized")

    # Flatten to 1D array (length 784)
    img_flattened = img_array.flatten()
    logging.info("Image flattened")

    return orig_image , img_flattened  # Return as NumPy array

# Example usage
image_path = "./data/prediction_sample_image.jpg"  # Replace with your image path
orig_image, features = preprocess_image(image_path)

# Reshape the flattened array back to 28x28 for visualization
sample_image = features.reshape(28, 28)

# Display the image
plt.imshow(orig_image, cmap="gray")
plt.title("Original Image ")
plt.axis("off")
plt.show()

print("Extracted Features Shape:", features.shape)  # Should be (784,)
print("Feature Values:", features)  # Print first 10 pixel values

# Display the image
plt.imshow(sample_image, cmap="gray")
plt.title("Sample MNIST Image")
plt.axis("off")
plt.show()

# Save to txt file
np.savetxt("./data/sample_image_features.txt", features, fmt="%1.2f", delimiter=",", newline=",")
logging.info("Saved features to file")
