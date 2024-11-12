import cv2
import numpy as np
import joblib
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, send_from_directory
from translations import english_dict, hindi_dict, bengali_dict, malayalam_dict, marathi_dict, punjabi_dict, tamil_dict, telugu_dict, kannada_dict, gujarati_dict, urdu_dict
from gtts import gTTS
import os
import uuid
import pygame

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
pygame.mixer.init()

# Load model and label encoder
model_dict = joblib.load('model.pkl')
model = model_dict['model']
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Initialize an empty sentence
sentence = ""

def process_frame(frame, selected_dict):
    global sentence
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = [0] * 84
        hand_idx = 0

        for hand_landmarks in results.multi_hand_landmarks:
            if hand_idx >= 2:
                break
            x_ = []
            y_ = []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            normalized = [(x - min_x) / (max_x - min_x) for x in x_] + [(y - min_y) / (max_y - min_y) for y in y_]
            data_aux[hand_idx * 42:(hand_idx + 1) * 42] = normalized
            hand_idx += 1

        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        predicted_character = selected_dict.get(predicted_label, "Unknown")

        if predicted_character != "Unknown":
            sentence += f"{predicted_character} "  # Append the predicted word to the sentence

        audio_filename = f'{uuid.uuid4()}.mp3'
        audio_file_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        tts = gTTS(predicted_character)
        tts.save(audio_file_path)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        return predicted_character, frame, audio_filename
    else:
        return None, frame, None

@app.route('/')
def index():
    return render_template('index.html', sentence=sentence)

@app.route('/predict', methods=['POST'])
def predict():
    global sentence
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image'].read()
        np_img = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        selected_language = request.form.get('language', 'en')

        selected_dict = {
            'hi': hindi_dict,
            'bn': bengali_dict,
            'pa': punjabi_dict,
            'ml': malayalam_dict,
            'mr': marathi_dict,
            'ta': tamil_dict,
            'te': telugu_dict,
            'kn': kannada_dict,
            'gu': gujarati_dict,
            'ur': urdu_dict
        }.get(selected_language, english_dict)

        prediction, processed_frame, audio_file = process_frame(frame, selected_dict)

        if prediction is None:
            return jsonify({'error': 'No hand landmarks detected or feature length mismatch'}), 400

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()

        response = {
            'prediction': prediction,
            'sentence': sentence.strip(),
            'audio_file': f"/static/{audio_file}" if audio_file else None,
            'image': frame_data.hex()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = ""
    return jsonify({'message': 'Sentence cleared', 'sentence': sentence})

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)