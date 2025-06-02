from collections import deque
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import mediapipe as mp

SEQUENCE_LEN = 30
FPS = 15
frame_buffer = deque(maxlen=SEQUENCE_LEN)
predicted_label = "Waiting..."
last_prediction_time = time.time()
class_names =  ['water','I', 'food', 'want', 'Hello', 'Thanks','please', 'you','help']
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose_model = mp_pose.Pose(static_image_mode=False)
hands_model = mp_hands.Hands(static_image_mode=False)

model_path = 'sign_lang_model_1_best.h5'
sign_lang_model_loaded = load_model(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Check if webcam opened successfully
if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
else:
    print("✅ Webcam opened successfully.")
    print("📷 Starting live sign prediction...")

# Function to extract keypoints from a frame
def extract_frame_landmarks(frame_rgb):
    keypoints = []

    pose_results = pose_model.process(frame_rgb)
    hands_results = hands_model.process(frame_rgb)

    pose_indices = [11, 12, 13, 14, 15, 16]
    if pose_results.pose_landmarks:
        for idx in pose_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * len(pose_indices) * 3)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        if len(hands_results.multi_hand_landmarks) == 1:
            keypoints.extend([0] * 21 * 3)
    else:
        keypoints.extend([0] * 21 * 3 * 2)

    return keypoints if len(keypoints) == 144 else None

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = extract_frame_landmarks(frame_rgb)

    if landmarks:
        frame_buffer.append(landmarks)

    # Predict every 2 seconds if 30 frames available
    if len(frame_buffer) == SEQUENCE_LEN and (time.time() - last_prediction_time) > 2:
        input_data = np.expand_dims(frame_buffer, axis=0)
        prediction = sign_lang_model_loaded.predict(input_data)
        predicted_label = class_names[np.argmax(prediction)]
        print(prediction)
        print(f"✅ Predicted: {predicted_label}")
        last_prediction_time = time.time()

    cv2.putText(frame, f'Sign: {predicted_label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.imshow('Live Sign Language Prediction', frame)
    cv2.namedWindow("SignLanguageWindow", cv2.WINDOW_NORMAL)
    cv2.imshow("SignLanguageWindow", frame)
    cv2.waitKey(1)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
