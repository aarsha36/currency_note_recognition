import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(data_dir):
    categories = os.listdir(data_dir)
    images, labels = [], []
    label_dict = {cat: idx for idx, cat in enumerate(categories)}

    for cat in categories:
        for img_name in os.listdir(os.path.join(data_dir, cat)):
            img_path = os.path.join(data_dir, cat, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))  # Resize for consistency
            images.append(img)
            labels.append(label_dict[cat])
    
    return np.array(images), np.array(labels), label_dict

data_dir = 'dataset'
X, y, label_dict = load_data(data_dir)
X = X / 255.0  # Normalize
y = to_categorical(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('currency_model.h5')
np.save('label_dict.npy', label_dict)

print("✅ Model trained and saved as currency_model.h5")
