import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room
from keras.models import load_model
from Face_Site import Predict
import datetime

app = Flask(__name__)
socketio = SocketIO(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load models once
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
emotion_model = load_model('FER_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def handle_connect():
    client_id = request.sid  # Use the session ID as a unique identifier
    join_room(client_id)  # Each client joins their own room
    print(f"Client {client_id} connected.")

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    leave_room(client_id)
    print(f"Client {client_id} disconnected.")

@socketio.on('image')
def handle_image(data):
    client_id = request.sid
    np_array = np.frombuffer(data, np.uint8)

    if np_array.size == 0:
        print(f"Client {client_id}: Received empty image data.")
        return

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    result, emotion = Predict(img, face_cascade, eye_cascade, emotion_model, emotion_labels)

    if result is None:
        print(f"Client {client_id}: Predict function returned None.")
        return

    # Log the time and emotion
    with open('emotion_log.txt', 'a') as log_file:
        log_file.write(f"{datetime.datetime.now()} - Client {client_id}: {emotion}\n")

    success, buffer = cv2.imencode('.jpg', result)
    if not success:
        print(f"Client {client_id}: Failed to encode image.")
        return

    processed_image_data = buffer.tobytes()

    socketio.emit('processed_image', processed_image_data, room=client_id)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)