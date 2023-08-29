import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Specify the height and width each video frame will be resized in our dataset
IMAGE_HEIGHT, IMAGE_WIDTH = 64,64

# specify the number of frames of a video that will be fed to the model as one sequence
SEQUENCE_LENGTH = 60

# Specify the directory containing the UCF50 dataset
DATASET_DIR = "ACTION_DATA"

# specify the list containing the names of the classes used for training.
CLASSES_LIST = ["fall_floor","run","walk","shoot_gun","hit","pullup"]


def frames_extraction(video_path):
  """
  This function will extract the required frames from a video after resizing and normalizing them
  args:
    video path: the path of the video in the disk, whose frames are to be extracted.
  Returns:
    frames_list: a list containing the resized and normalized frames of the video.
  """

  # declare a list to store video frame
  frames_list = []

  # Read the video file using the video capture object
  video_reader = cv2.VideoCapture(video_path)

  # Get the total number of frames in the video
  video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

  # Calculate the interval after which frames will be added to the list.
  skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

  # Iterate through the Video Frames
  for frame_counter in range(SEQUENCE_LENGTH):

    # set the current frame position of the video
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter*skip_frames_window)

    # Reading the frame position of the video
    success, frame = video_reader.read()

    # check if video frame is not successfully read then break the loop
    if not success:
      break

    # Resize the frame to fixed height and width
    resized_frame = cv2.resize(frame,(IMAGE_HEIGHT,IMAGE_WIDTH))

    # Normalize the resized fram by dividing it with 255 so that each pixel value then lies between 0 and 1
    normalized_frame = resized_frame/255

    # append the normalized frame into the frames list
    frames_list.append(normalized_frame)

  # Release the video capture object
  video_reader.release()

  # Return the frames  list
  return frames_list

def create_dataset():
  """
  This function will extract the data of the selected classes and create the required dataset.
  return:
    features:           A list containing the extracted frames of the videos.
    labels:             A list containing the indexes lf the classes associated with the videos.
    video_files_paths:  A list containing the paths of the videos in the disk.
  """

  # Declared Empty lists to store the features, labels and video path values.
  features = []
  labels = []
  video_files_paths = []

  # iterating through all the classes mentioned in the classes list
  for class_index, class_name in enumerate(CLASSES_LIST):

    # Display the name of the class whose data is being extracted.
    print(f'Extracting Data of Class: {class_name}')

    # get the list of video files present in the specific class name directory
    files_list = os.listdir(os.path.join(DATASET_DIR,class_name))

    # Iterate through all the files present in the files list
    for file_name in files_list:

      # get the complete video path
      video_file_path = os.path.join(DATASET_DIR,class_name,file_name)

      # Extract the frames of the video file.
      frames = frames_extraction(video_file_path)

      # check if the extracted frames are equal to the SEQUENCE_LENGTH sepcified above
      # so ignore the videos having frames less than the SEQUENCE_LENGTH.
      if len(frames) == SEQUENCE_LENGTH:

        # Append the data to their respective lists.
        features.append(frames)
        labels.append(class_index)
        video_files_paths.append(video_file_path)

  # converting the list to numpy arrays
  features = np.asarray(features)
  labels = np.array(labels)

  # return the frames, class index, and video file path
  return features, labels, video_files_paths


# create dataset
features, labels, video_file_path = create_dataset()

# using keras's to_categorical method to convert labels into one-hot-encoded vectors
one_hot_encoded_labels = to_categorical(labels)

# Split the dataset into Train (75%) and test set (25%)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state = seed_constant)

def create_convlstm_model():
  """
  This function will construct the required convlstm model
  Return:
    model: it is the required constructed convlstm model
  """

  # we will use a sequencial model for model construction
  model = Sequential()

  # Define the model Architecture
  #######################################################################################

  model.add(ConvLSTM2D(filters=4,
                       kernel_size=(3,3),
                       activation = 'tanh',
                       data_format='channels_last',
                       recurrent_dropout=0.2,
                       return_sequences=True,
                       input_shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,3)))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format="channels_last"))
  model.add(TimeDistributed(Dropout(0.2)))

  model.add(ConvLSTM2D(filters=8,
                       kernel_size=(3,3),
                       activation = 'tanh',
                       data_format='channels_last',
                       recurrent_dropout=0.2,
                       return_sequences=True,
                       input_shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,3)))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format="channels_last"))
  model.add(TimeDistributed(Dropout(0.2)))

  model.add(ConvLSTM2D(filters=14,
                       kernel_size=(3,3),
                       activation = 'tanh',
                       data_format='channels_last',
                       recurrent_dropout=0.2,
                       return_sequences=True,
                       input_shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,3)))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format="channels_last"))
  model.add(TimeDistributed(Dropout(0.2)))

  model.add(ConvLSTM2D(filters=26,
                       kernel_size=(3,3),
                       activation = 'tanh',
                       data_format='channels_last',
                       recurrent_dropout=0.2,
                       return_sequences=True,
                       input_shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,3)))
  model.add(MaxPooling3D(pool_size=(1,2,2),padding='same',data_format="channels_last"))
  # model.add(TimeDistributed(Dropout(0.2)))

  model.add(Flatten())

  model.add(Dense(len(CLASSES_LIST),activation='softmax'))

  #############################################################################################

  # Display the models summary
  model.summary()

  # Return the constructed convlstm model
  return model

# Construct the required convlstm model
convlstm_model = create_convlstm_model()

# Display the success message
print("Model Created Successfully!")

# plot the structure of the constructed model
plot_model(convlstm_model,to_file = 'convlstm_model_structure_plot.png',show_shapes=True,show_layer_names=True)

# create an instance of early stopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10, mode='min',restore_best_weights=True)

# compile the model and specify loss function, optimizer and metrics values to the model
convlstm_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=["categorical_accuracy"])

# Start training the model
convlstm_model_training_history = convlstm_model.fit(x = features_train,y = labels_train,epochs=10,batch_size=4,shuffle=True,validation_split=0.2,callbacks=[early_stopping_callback])

# Evaluate the test test
model_evaluation_history = convlstm_model.evaluate(features_test,labels_test)

# get the loss and accuracy from model_evaluation_history
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define the string date format
# Get the current Date and time in a datetime object
# convert the datetime object to string according to the style mentioned in date_time_format string
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt,date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
model_file_name = f'convlstm_model__Date_Time_{current_date_time_string}__Loss_{model_evaluation_loss}__Accuracy_{model_evaluation_accuracy}.h5'

# save model
convlstm_model.save(model_file_name)
