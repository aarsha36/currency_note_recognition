import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# Load the trained model and label dictionary
model = load_model('currency_model.h5')
label_dict = np.load('label_dict.npy', allow_pickle=True).item()
reverse_dict = {v: k for k, v in label_dict.items()}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speaking speed

prev_label = None  # To avoid repeated speaking

print("🎥 Real-time currency note recognition with audio started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Resize and preprocess frame
    img = cv2.resize(frame, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    class_index = np.argmax(pred)
    label = reverse_dict[class_index]

    # Speak only if the prediction changes
    if label != prev_label:
        text = f"This is a {label} rupee note"
        print(f"🔊 {text}")
        engine.say(text)
        engine.runAndWait()
        prev_label = label

    # Show prediction on screen
    cv2.putText(frame, f"Prediction: ₹{label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Currency Note Recognition (with Audio)', frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
