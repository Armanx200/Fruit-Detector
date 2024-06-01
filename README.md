# 🍎🍊 Fruit Detector 🍌🍇

Welcome to the **Fruit Detector** project! This repository contains a machine learning model that can identify different fruits from images. 🥭🍉

## 📁 Project Structure

Here's an overview of the project directory:

```plaintext
C:\Users\Arman\Desktop\ML\Projects\Fruit-Detector
│
├── Test_File/
│   └── [Test images...]
├── Train_File/
│   └── [Training images...]
│
├── Fruit_Detector.py
├── fruit_detector_model.h5
├── label_encoder.npy
└── predict_fruit.py
```

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Necessary Python packages (use `pip install -r requirements.txt`)

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fruit-detector.git
cd fruit-detector
```

## 📝 Usage

### Training the Model

To train the model, run the `Fruit_Detector.py` script:

```bash
python Fruit_Detector.py
```

This will load images from the `Train_File` directory, train the model, and save it as `fruit_detector_model.h5`. It also saves the label encoder as `label_encoder.npy`.

### Predicting Fruits

To predict the fruit in an image, use the `predict_fruit.py` script:

```bash
python predict_fruit.py <path_to_image>
```

Replace `<path_to_image>` with the path to your image file. For example:

```bash
python predict_fruit.py ./Test_File/apple_01.jpg
```

## 🛠️ Code Overview

### Fruit_Detector.py

This script handles the entire pipeline from loading data, training the model, and saving the model and label encoder. Key steps include:

- **Loading Images**: Reads images and their labels from the `Train_File` and `Test_File` directories.
- **Preprocessing**: Normalizes the images and encodes the labels.
- **Model Building**: Constructs a Convolutional Neural Network (CNN) using Keras.
- **Training**: Trains the model on the training data and evaluates it on the test data.
- **Saving**: Saves the trained model and label encoder for later use.

### predict_fruit.py

This script loads a saved model and label encoder to make predictions on new images. Key steps include:

- **Loading Model**: Loads the trained model and label encoder.
- **Image Preprocessing**: Prepares the image for prediction by resizing and normalizing.
- **Prediction**: Predicts the class of the fruit and prints the result with confidence score.

## 📊 Model Performance

After training, the model achieved an accuracy of **97.27%** on the test data.

## 🏗️ Future Work

- Expand the dataset with more fruit images.
- Improve the model architecture for better accuracy.
- Implement data augmentation to enhance model robustness.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📬 Contact

For questions or comments, please reach out to [kianianarman1@gmail.com](mailto:kianianarman1@gmail.com).

---

*Happy Coding!* 🎉
