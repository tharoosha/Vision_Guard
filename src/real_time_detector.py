import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model  # Import the load_model function
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
# Define the necessary variables
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Specify the number of frames of a video that will be fed to the model as one sequence
SEQUENCE_LENGTH = 60

no_sequences = 40

DATA_PATH = "MP_DATA"  # Update this path accordingly

action_list = np.array(["fall_floor", "run", "walk", "shoot_gun", "pullup"])

# Specify the list containing the names of the classes used for training.
CLASSES_LIST = ["fall_floor", "run", "walk", "shoot_gun", "pullup"]


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
    image.flags.writeable = False                  # Image no longer writeable
    results = model.process(image)                # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion
    return image, results


def draw_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


def draw_styled_landmarks(image, results):
    # Define the style for the connections
    red_drawing_spec = mp_drawing.DrawingSpec(
        color=(80, 110, 10), thickness=1, circle_radius=1)
    green_drawing_spec = mp_drawing.DrawingSpec(
        color=(0, 256, 121), thickness=1, circle_radius=1)

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS, red_drawing_spec, green_drawing_spec)


def extract_keypoints(result):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility]
                        for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)  # Placeholder values
    return pose


# New detection variables
model = load_model('action4.h5')  # Load the model from the saved file

colors = [(100, 0, 0), (0, 100, 0), (0, 0, 100), (200, 0, 0), (0, 200, 0)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, action_list[num], [
                    0, 85+num*40], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


# New detection variables
sequence = []
sentence = []
threshold = 0.3

cap = cv2.VideoCapture(0)
# Set  mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detection
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw Landmarks
        draw_styled_landmarks(image, results)

        #  2.Prediction Logic
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:60]

        print(len(sequence))
        if len(sequence) == 60:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            last = action_list[np.argmax(res)]
            print(last)

            # Viz Logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if action_list[np.argmax(res)] != sentence[:-1]:
                        sentence.append(last)
                else:
                    sentence.append(last)
            image = prob_viz(res, action_list, image, colors)

        if len(sentence) > 5:
            sentence = sentence[-1:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
