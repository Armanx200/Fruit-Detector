import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
import sys

# Load the trained model and label encoder
model = load_model('fruit_detector_model.h5')
label_encoder = np.load('label_encoder.npy', allow_pickle=True)

def load_and_prepare_image(image_path):
    try:
        image = load_img(image_path, target_size=(100, 100))
        image = img_to_array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict_fruit(image_path):
    image = load_and_prepare_image(image_path)
    if image is None:
        return "Error loading image"
    
    prediction = model.predict(image)
    predicted_label_index = np.argmax(prediction)
    confidence = prediction[0][predicted_label_index]

    if confidence < 0.5:
        return "Not a fruit"
    else:
        predicted_label = label_encoder[predicted_label_index]
        return f"{predicted_label} with {confidence * 100:.2f}% confidence"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_fruit.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_fruit(image_path)
    print(result)
