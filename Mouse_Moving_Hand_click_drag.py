# pip install mediapipe opencv-python pyautogui

import cv2
import mediapipe as mp
import pyautogui
import math
from time import sleep

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Define the region of interest (ROI) within the camera frame
roi_margin = 100  # Adjust this value as needed

# Start video capture
cap = cv2.VideoCapture(0)

flag = False

# Variables to track the initial position of the closed fist
initial_fist_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger MCP (landmark 5)
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * frame.shape[1])
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame.shape[0])

            # Get the coordinates of the thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the 3D Euclidean distance between the thumb tip and index finger tip
            distance_left_click = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2 + (thumb_tip.z - index_tip.z) ** 2)
            # print("LEft:", distance_left_click) 

            # Set a threshold for the distance to detect a left click
            click_threshold = 0.05  # Adjust this value as needed

            # Trigger a left click if the distance is below the threshold
            if distance_left_click < click_threshold and flag == False:
                pyautogui.click()
                # sleep(0.2)

            # Get the coordinates of the index finger tip (landmark 8) and thumb base (landmark 1)
            thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

            # Calculate the 3D Euclidean distance between the index finger tip and thumb base
            distance_right_click = math.sqrt((index_tip.x - thumb_base.x) ** 2 + (index_tip.y - thumb_base.y) ** 2 + (index_tip.z - thumb_base.z) ** 2)
            # print("right:", distance_right_click) 
            # Set a threshold for the distance to detect a right click
            right_click_threshold = 0.05  # Adjust this value as needed

            # Trigger a right click if the distance is below the threshold
            if distance_right_click < right_click_threshold and flag == False:
                pyautogui.rightClick()
                # sleep(0.2)

            # Get the coordinates of the finger tips and bases (excluding the thumb)
            tips = [
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            bases = [
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            ]

            # Check if all finger tips are below their respective bases
            closed_fist = all(tip.y > base.y for tip, base in zip(tips, bases))

            # Trigger a scroll action if a closed fist is detected
            if closed_fist:
                flag = True
                if initial_fist_y is None:
                    initial_fist_y = y  # Set the initial position of the closed fist
                else:
                    # Calculate the movement of the fist relative to its initial position
                    movement_y = y - initial_fist_y
                    pyautogui.scroll(-movement_y * 10)  # Adjust the scroll speed as needed
                    initial_fist_y = y  # Update the initial position

            else:
                flag = False
                initial_fist_y = None  # Reset the initial position if the fist is not closed

            # Map the coordinates within the ROI to the full screen size
            roi_width = frame.shape[1] - 2 * roi_margin
            roi_height = frame.shape[0] - 2 * roi_margin
            screen_x = int((x - roi_margin) * screen_width / roi_width)
            screen_y = int((y - roi_margin) * screen_height / roi_height)

            # Ensure the coordinates are within the screen bounds, one pixel away from the edge
            screen_x = max(1, min(screen_width - 1, screen_x))
            screen_y = max(1, min(screen_height - 1, screen_y))

            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Optionally draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # print(flag)
    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
