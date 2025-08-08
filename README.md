# Real-Time Sign Language Translator 🤟

![Sign Language Translator Demo](https://i.imgur.com/ag5A1aB.gif)

## 📖 Overview

This project is a real-time sign language translator that uses a computer's webcam to capture hand gestures and translate them into text. The primary goal is to help bridge the communication gap between the deaf and hard-of-hearing community and the hearing world.

This implementation uses computer vision to detect hand landmarks and a trained deep learning model to classify the corresponding sign (e.g., letters of the American Sign Language alphabet).

--- <img width="940" height="470" alt="image" src="https://github.com/user-attachments/assets/bf5ddb8b-3b92-4534-975f-f9a6af584748" />

 
Fig. 1. American Sign Language Alphabets

## ✨ Features

* **Real-Time Translation:** Translates sign language gestures from a live webcam feed with minimal latency.
* **High Accuracy:** Utilizes a robust deep learning model trained on a comprehensive dataset for reliable classification.
* **User-Friendly Interface:** An intuitive display window powered by OpenCV shows the camera feed, detected hand landmarks, and the translated text.
* **Extensible:** The architecture is designed to be easily adaptable for recognizing new signs, words, or even different sign languages.

---

## 🛠️ How It Works & Technology Stack

The translation process is broken down into several key steps:

1.  **Video Capture:** The system uses `OpenCV` to capture video frames from the user's webcam.
2.  **Hand Landmark Detection:** We use Google's `Mediapipe` library to detect the position of the hand and extract the 3D coordinates ($x, y, z$) of its 21 key landmarks. This is highly efficient and works on most standard CPUs.
3.  **Feature Preprocessing:** The raw landmark coordinates are normalized to become independent of the hand's position and scale in the frame. This creates a consistent feature vector for the model.
4.  **Sign Classification:** The processed feature vector is fed into a pre-trained `TensorFlow`/`Keras` neural network model. The model outputs a prediction, classifying the gesture into a specific letter or word.
5.  **Display:** The predicted letter is overlaid onto the video feed, providing immediate feedback to the user.

**Core Technologies:**
* **Python 3.9+**
* **OpenCV:** For video capture and image processing.
* **Mediapipe:** For high-fidelity hand and finger tracking.
* **TensorFlow / Keras:** For building and running the classification model.
* **NumPy:** For numerical operations and data manipulation.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9 or higher
* A webcam connected to your computer
* Git for cloning the repository

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/sign-language-translator.git](https://github.com/your-username/sign-language-translator.git)
    cd sign-language-translator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *The `requirements.txt` file should contain:*
    ```
    opencv-python
    mediapipe
    tensorflow
    numpy
    ```

4.  **Download the pre-trained model:**
    Download the model file (`model.h5`) from the project's [Releases Page](https://github.com/your-username/sign-language-translator/releases) and place it in the `model/` directory.

---

## ▶️ Usage

To start the translator, simply run the main script from your terminal:




 <img width="774" height="463" alt="image" src="https://github.com/user-attachments/assets/e262aa09-89ab-4785-902d-2783bea4b88b" />

Fig. 2. Showing a Finger and testing detection

<img width="949" height="329" alt="image" src="https://github.com/user-attachments/assets/7d378486-266b-48cb-b6ff-e9887cacc24f" />

Fig. 4. Landmarks & Graphs

 
Fig. 4.1 Landmarks

<img width="449" height="357" alt="image" src="https://github.com/user-attachments/assets/6cd5703b-91f1-4e3f-9e35-b3976ccd16f2" />
<img width="453" height="360" alt="image" src="https://github.com/user-attachments/assets/1ffba287-0e54-44f1-b4f6-4d1cee4ca9dd" />

         
Fig. 4.2 Graphs

V.	CONCLUSION AND FUTURE SCOPE
<img width="801" height="450" alt="image" src="https://github.com/user-attachments/assets/25fdc6dd-81a2-456f-ba02-5edc72c998d8" />


Conclusion
Language is more than just words—it is about understanding and connection. The Sign Language Translator is a step towards breaking the barriers that separate the hearing-impaired community from the rest of the world. By leveraging AI and deep learning, we have created a tool that promotes inclusivity and accessibility. As technology advances, we envision a future where sign language translation is as seamless as spoken conversation.
Because everyone deserves to be heard.

REFERENCES
[1] M. Aarthi and P. Vijayalakshmi, "Sign Language to Speech Conversion," International Conference on Recent Trends in Information Technology, 2016.
[2] V. N. T. Truong, C. K. Yang, and Q. V. Tran, "A Translator for American Sign Language to Text and Speech," IEEE Global Conference on Consumer Electronics, 2016.
[3] Anuja and Bindu V. Nair, "A Review on Indian Sign Language Recognition," International Journal of Computer Applications, 2013.
[4] Google AI Gesture Recognizer: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
[5] GitHub - Hand Gesture Recognition using MediaPipe: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
