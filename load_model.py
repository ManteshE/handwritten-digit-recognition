import cv2
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO for less verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MODEL_PATH = "/content/handwritten-digit-recognition/tf-cnn-model.h5"

def predict_digit(image_path):
    """
    Predict the digit in the given image using the pre-trained model.
    """
    try:
        # Load the model
        logging.debug("Attempting to load the model from: %s", MODEL_PATH)
        model = models.load_model(MODEL_PATH)
        logging.info("Loaded model from disk successfully.")
    except Exception as e:
        logging.error("Failed to load model: %s", e)
        raise

    try:
        # Read the image
        logging.debug("Attempting to read the image from: %s", image_path)
        image = cv2.imread(image_path, 0)  # Read as grayscale
        if image is None:
            raise FileNotFoundError("Image could not be loaded. Check the file path.")

        logging.info("Image loaded successfully. Shape: %s", image.shape)
        
        # Resize and preprocess the image
        logging.debug("Resizing image to 28x28 for the model.")
        image1 = cv2.resize(image, (28, 28))
        logging.info("Image resized. New shape: %s", image1.shape)

        # Prepare image for prediction
        image2 = image1.reshape(1, 28, 28, 1)
        logging.debug("Image reshaped for model input: %s", image2.shape)
        
        # Display the image using matplotlib (instead of cv2.imshow)
        plt.imshow(image1, cmap='gray')
        plt.title("Input Image")
        plt.axis("off")  # Hide axes
        plt.show()

        # Predict the digit
        logging.info("Running prediction on the input image.")
        pred = np.argmax(model.predict(image2), axis=-1)
        logging.info("Prediction successful. Predicted digit: %s", pred[0])

        return pred[0]
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise

def main(image_path):
    """
    Main function to predict the digit in the image provided via CLI.
    """
    logging.debug("Starting main function with image_path: %s", image_path)
    try:
        predicted_digit = predict_digit(image_path)
        logging.info("Predicted Digit: %s", predicted_digit)
        print("Predicted Digit: {}".format(predicted_digit))
    except FileNotFoundError:
        logging.error("[ERROR]: Image not found at %s", image_path)
        print("[ERROR]: Image not found.")
    except Exception as e:
        logging.error("[ERROR]: Unexpected error: %s", e)
        print("[ERROR]: Unexpected error occurred. Check logs for details.")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            logging.error("No image path provided. Usage: python3 script.py <image_path>")
            print("Usage: python3 script.py <image_path>")
        else:
            main(image_path=sys.argv[1])
    except Exception as e:
        logging.error("Script terminated due to an error: %s", e)
