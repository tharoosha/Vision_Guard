import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities


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


# Specify the height and width each video frame will be resized in our dataset
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the directory containing the UCF50 dataset
DATASET_DIR = "ACTION_DATA"

# specify the list containing the names of the classes used for training.
CLASSES_LIST = ["fall_floor", "run", "walk", "shoot_gun", "pullup"]

# Actions that we try to detect
action_list = np.array(["fall_floor", "run", "walk", "shoot_gun", "pullup"])

# specify the number of frames of a video that will be fed to the model as one sequence
SEQUENCE_LENGTH = 60

no_sequences = 40

DATA_PATH = os.path.join("MP_DATA")


for action in action_list:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            print("created")
        except:
            pass

# Iterate through action classes
for action in CLASSES_LIST:
    action_dir = os.path.join(DATASET_DIR, action)
    if os.path.isdir(action_dir):
        video_files = [f for f in os.listdir(action_dir)]
        count = 0
        for video_filename in video_files:
            if count < no_sequences:

                video_path = os.path.join(action_dir, video_filename)

                # Open the video file
                cap = cv2.VideoCapture(video_path)

                video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                skip_frames_window = max(
                    int(video_frames_count/SEQUENCE_LENGTH), 1)

                if video_frames_count >= SEQUENCE_LENGTH:
                    # Initialize MediaPipe Holistic model
                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        for frame_num in range(SEQUENCE_LENGTH):
                            # set the current frame position of the video
                            cap.set(cv2.CAP_PROP_POS_FRAMES,
                                    frame_num*skip_frames_window)

                            # Read video frame
                            ret, frame = cap.read()
                            if not ret:  # Check if frame was successfully read
                                continue  # Skip processing for this iteration

                            # Make detection
                            image, results = mediapipe_detection(
                                frame, holistic)

                            # Draw landmarks
                            draw_styled_landmarks(image, results)

                            image = cv2.flip(image, 1)

                            # Wait logic
                            if frame_num == 0:
                                cv2.putText(image, 'Starting Collection', (120, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                                cv2.putText(image, "Collecting frames for {} Video Number {}".format(
                                    action, count), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                                cv2.waitKey(2000)
                                print("x")
                            else:
                                cv2.putText(image, "Collecting frames for {} Video Number {}".format(
                                    action, count), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                                print("y")

                            # Export keypoints
                            keypoints = extract_keypoints(results)
                            npy_path = os.path.join("MP_DATA", action, str(
                                count), str(frame_num) + ".npy")
                            try:
                                np.save(npy_path, keypoints)
                                print(f"Saved keypoints to {npy_path}")
                            except Exception as e:
                                print(f"Error saving keypoints: {e}")

                            cv2.imshow('OpenCV Feed', image)

                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

                    cap.release()
                    cv2.destroyAllWindows()
                    count += 1
            else:
                print("Nah")
