import numpy as np
import os
import cv2
import mediapipe as mp
import tensorflow as tf
from keras import *

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process the image and detect landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to visualize the detected landmarks
def draw_landmarks(image, results):
    # Draw face landmarks with modified attributes
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1))

    # Draw left hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=1))

    # Draw right hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1))

# Function to visualize the detected landmarks and return keypoints as arrays
def extract_keypoints(results):
    # Initialize empty arrays for keypoints
    face_keypoints = []
    pose_keypoints = []
    left_hand_keypoints = []
    right_hand_keypoints = []

    # Extract face landmarks
    if results.face_landmarks:
        face_keypoints = np.array([landmark.x for landmark in results.face_landmarks.landmark] +
                                  [landmark.y for landmark in results.face_landmarks.landmark] +
                                  [landmark.z for landmark in results.face_landmarks.landmark])
    else:
        face_keypoints = np.zeros(468 * 3)  # 468 landmarks in face mesh

    # Extract pose landmarks with visibility information
    if results.pose_landmarks:
        pose_keypoints = np.array([landmark.x for landmark in results.pose_landmarks.landmark] +
                                  [landmark.y for landmark in results.pose_landmarks.landmark] +
                                  [landmark.z for landmark in results.pose_landmarks.landmark] +
                                  [landmark.visibility for landmark in results.pose_landmarks.landmark])
    else:
        pose_keypoints = np.zeros((33 * 4))  # 33 landmarks in pose, including visibility

    # Extract left hand landmarks
    if results.left_hand_landmarks:
        left_hand_keypoints = np.array([landmark.x for landmark in results.left_hand_landmarks.landmark] +
                                       [landmark.y for landmark in results.left_hand_landmarks.landmark] +
                                       [landmark.z for landmark in results.left_hand_landmarks.landmark])
    else:
        left_hand_keypoints = np.zeros(21 * 3)  # 21 landmarks in left hand

    # Extract right hand landmarks
    if results.right_hand_landmarks:
        right_hand_keypoints = np.array([landmark.x for landmark in results.right_hand_landmarks.landmark] +
                                        [landmark.y for landmark in results.right_hand_landmarks.landmark] +
                                        [landmark.z for landmark in results.right_hand_landmarks.landmark])
    else:
        right_hand_keypoints = np.zeros(21 * 3)  # 21 landmarks in right hand

    # Return concatenated keypoints array
    return np.concatenate((face_keypoints, pose_keypoints, left_hand_keypoints, right_hand_keypoints))

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Setting up a folder for Data Collection
# ***Setting Data Path*** #
Data_Path = os.path.join("DataSet")
actions = np.array(['Hello', 'Thanks', 'I love You'])
no_sequences = 3
sequence_length = 30




# Process video frames in real-time
with mp_holistic as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            # Initialize variables to store keypoints arrays for each sequence
            all_keypoints = []

            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                # Detect landmarks in the current frame
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks on the frame
                draw_landmarks(image, results)

                # visualizing the process of collection
                if frame_num == 0:
                    cv2.putText(image, 'Starting Collection', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video number {}'.format(action, sequence),
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Exporting & saving key points
                keypoints = extract_keypoints(results)

                # Ensure that the directories exist before saving
                npy_path = os.path.join(Data_Path, action, str(sequence), str(frame_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)

                # Save the keypoints
                np.save(npy_path, keypoints)

                # Append keypoints to the list for this sequence
                all_keypoints.append(keypoints)

                # Display the processed frame
                cv2.imshow('Live Feed', image)

                # Check if 'q' is pressed to quit
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Convert the list of keypoints arrays to a numpy array for this sequence
            all_keypoints_array = np.array(all_keypoints)

            # Print the shape of the resulting array for this sequence
            print("Shape of Keypoints Array for {} Video number {}: {}".format(action, sequence, all_keypoints_array.shape))

# Release webcam resources
cap.release()

# Close all windows
cv2.destroyAllWindows()

# Preprocessing and Label Creation

from sklearn.model_selection import train_test_split


#Labeling assigned actions
label_map= {label:num for num, label in enumerate(actions)}
label_map


# Making the labelled array
sequences, labels = [], []
for action in actions:
    for sequence in range(sequence_length):
        window=[]
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(Data_Path, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

np.array(sequences).shape
np.array(labels).shape
