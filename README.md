
# Indian Sign Language Detection System

A **real-time Indian Sign Language Detection System** that leverages **Random Forest Classifier** and **MediaPipe** to recognize hand gestures and translate them into textual and auditory outputs. This project supports **11 Indian languages** and provides real-time gesture-to-text and gesture-to-speech conversion.

## Features
- **Gesture Recognition**: Detects and classifies Indian sign language gestures in real time.
- **Multi-language Translation**: Translates gestures into 11 Indian languages including Hindi, Bengali, Malayalam, Marathi, Punjabi, Tamil, Telugu, Kannada, Gujarati, Urdu, and English.
- **Text-to-Speech (TTS)**: Provides auditory output for the recognized gestures using Google Text-to-Speech (gTTS).
- **Flask-Based Interface**: Offers a web interface for live gesture recognition and interaction.

## Tech Stack
- **Programming Language**: Python
- **Machine Learning**: Random Forest Classifier
- **Hand Tracking**: MediaPipe
- **Web Framework**: Flask
- **Serialization**: Pickle
- **Text-to-Speech**: gTTS
- **Dataset**: Custom dataset created with OpenCV

## How It Works
1. **Data Collection**: Uses OpenCV to capture hand gesture images and save them in a structured dataset.
2. **Feature Extraction**: Extracts 84-point feature vectors for gestures using MediaPipe's hand landmarks.
3. **Model Training**: Trains a Random Forest Classifier on the feature vectors.
4. **Real-Time Recognition**: Uses the trained model to predict gestures in real-time from webcam feed.
5. **Translation & Speech**: Maps the recognized gestures to corresponding language words and generates audio output.

## Supported Languages
- Hindi
- Bengali
- Malayalam
- Marathi
- Punjabi
- Tamil
- Telugu
- Kannada
- Gujarati
- Urdu
- English

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/OJAS-P-JOSHI/IndianSignLangDetect.git
   cd IndianSignLangDetect
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Usage
- Launch the application by running `app.py`.
- Use your webcam to perform sign language gestures.
- The system will display the recognized text and play the audio output.

## Model Details
- **Algorithm**: Random Forest Classifier
- **Feature Vector**: 84-point hand landmark features for one or two hands.
- **Serialization**: Model and LabelEncoder are saved as `.pkl` files for reuse.

## Future Enhancements
- Extend support for more gestures and languages.
- Improve accuracy using deep learning models.
- Deploy the application on cloud platforms for public access.

## Contributing
Contributions are welcome! Please create a pull request for any suggestions or feature additions.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [MediaPipe](https://mediapipe.dev) for hand tracking.
- [Google Text-to-Speech](https://pypi.org/project/gTTS/) for audio generation.
