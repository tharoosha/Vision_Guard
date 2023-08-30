import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


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
    # mp_drawing.draw_landmarks(image, results.face_landmarks)          # Draw face connections
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)     # Draw lef hand connections
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw right hand connections


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


# Specify the number of frames of a video that will be fed to the model as one sequence
SEQUENCE_LENGTH = 60

no_sequences = 40

DATA_PATH = "MP_DATA"  # Update this path accordingly

action_list = np.array(["fall_floor", "run", "walk", "shoot_gun", "pullup"])

# Specify the list containing the names of the classes used for training.
CLASSES_LIST = ["fall_floor", "run", "walk", "shoot_gun", "pullup"]
# Create a label map
label_map = {label: num for num, label in enumerate(CLASSES_LIST)}

sequences, labels = [], []
for action in CLASSES_LIST:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            res = np.load(os.path.join(DATA_PATH, action,
                          str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        print(len(sequences))

x = np.array(sequences)
y = to_categorical(labels, num_classes=len(CLASSES_LIST)).astype(int)

print("x shape:", x.shape)
print("y shape:", y.shape)

# print(np.array(labels).shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(128, return_sequences=True,
          activation='tanh', input_shape=(60, 132)))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(len(CLASSES_LIST), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


# create an instance of early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

# compile the model and specify loss function, optimizer and metrics values to the model
model.compile(loss='categorical_crossentropy', optimizer='Adam',
              metrics=["categorical_accuracy"])

# Start training the model
lstm_model_training_history = model.fit(x=x_train, y=y_train, epochs=1000, batch_size=4,
                                        shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback])


print(model.summary())

model.save('action4.h5')

yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

# New detection variables

# sequence = []
# sentence = []
# threshold = 0.4

# cap = cv2.VideoCapture(0)
# # Set  mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():

#         # Read feed
#         ret, frame = cap.read()

#         # Make detection
#         image, results = mediapipe_detection(frame, holistic)
#         print(results)

#         # Draw Landmarks
#         draw_styled_landmarks(image, results)

#         #  2.Prediction Logic
#         keypoints = extract_keypoints(results)
#         sequence.insert(0, keypoints)
#         sequence = sequence[:40]

#         print(len(sequence))
#         if len(sequence) == 40:
#             res = model.predict(np.expand_dims(sequence, axis=0))[0]
#             last = action_list[np.argmax(res)]
#             print(last)

#             # Viz Logic
#             if res[np.argmax(res)] > threshold:
#                 if len(sentence) > 0:
#                     if action_list[np.argmax(res)] != sentence[:-1]:
#                         sentence.append(last)
#                 else:
#                     sentence.append(last)

#         if len(sentence) > 5:
#             sentence = sentence[-10:]

#         cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
#         cv2.putText(image, ' '.join(sentence), (3, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         # Show to screen
#         cv2.imshow('OpenCV Feed', image)

#         # Break
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
