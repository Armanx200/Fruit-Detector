import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Directories
train_dir = 'Train_File'
test_dir = 'Test_File'

# Function to load images and labels from filenames
def load_images_and_labels(data_dir):
    images = []
    labels = []
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"{data_dir} is not a valid directory")
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg'):
            label = file_name.split('_')[1].split('.')[0]
            img_path = os.path.join(data_dir, file_name)
            try:
                image = load_img(img_path, target_size=(100, 100))
                image = img_to_array(image) / 255.0
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load data
X_train, y_train = load_images_and_labels(train_dir)
X_test, y_test = load_images_and_labels(test_dir)

# Check if data is loaded correctly
if X_train.size == 0 or y_train.size == 0:
    raise ValueError("No training data loaded. Please check the train directory and its structure.")
if X_test.size == 0 or y_test.size == 0:
    raise ValueError("No testing data loaded. Please check the test directory and its structure.")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_train)), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=10, validation_data=(X_test, y_test_categorical))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Accuracy: {test_accuracy * 100:.2f}%')

# Save the model and label encoder for the next script
model.save('fruit_detector_model.h5')
with open('label_encoder.npy', 'wb') as f:
    np.save(f, label_encoder.classes_)
