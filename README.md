# 🇮🇳 Real-Time Indian Currency Note Recognition System 🪙📸

This project is a real-time machine learning system that identifies Indian currency notes using image input and gives **text-based predictions** (with optional speech output). Designed for accessibility, the system can assist **visually impaired users** in identifying different currency denominations.

---

## 🧠 Features

- Real-time image-based currency recognition using OpenCV
- Trained ML model (SVM/KNN) for classification
- Predicts denomination such as ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000
- Optional text-to-speech output using `pyttsx3`
- Easy-to-run script with minimal setup

---

## 📂 Project Structure

currency_note_recognition/
│
├── model.py # Trains the currency recognition model
├── predictaudio.py # Loads model and predicts denomination from live camera
└── dataset # (download from drive link provided)

---

##📁 Dataset
💾 Dataset is not included due to size.

Download the Indian currency dataset from this link:
📎https://drive.google.com/drive/folders/1mEbYYZUJwnm2mtVT0qqo1B7vFdOWRmpK?usp=sharing

Unzip it and place it in the project root folder like this:

currency_note_recognition/
└── dataset/
    ├── 10/
    ├── 20/
    ├── 50/
    └── ...

---

##🏋️‍♂️ Training the Model
After placing the dataset, run the training script:

python model.py
This will generate a currency_model.h5 file, which is used for prediction.

---

##🔍 Running Prediction
Run the prediction file using:

python predictaudio.py
Hold the currency note in front of the webcam. The model will display the predicted denomination (and speak it if pyttsx3 is working).

---

##🎥 Demo
✅ Want to see it in action?

📹 Watch the demo video here
📎https://drive.google.com/drive/folders/1mEbYYZUJwnm2mtVT0qqo1B7vFdOWRmpK?usp=sharing

---

##🧰 Requirements
Python 3.6+
OpenCV
Numpy
Scikit-learn
Tensorflow
pyttsx3

---

##💡 Future Improvements
Add voice commands for interaction
Integrate with Raspberry Pi or mobile app
Improve robustness in varying lighting

---

##🙋‍♂️ Author
Developed by [V S Aarsha]
Feel free to reach out for feedback or collaborations!

---

##📄 License
This project is open-source and free to use for educational and research purposes.
