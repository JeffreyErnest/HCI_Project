import cv2
import mediapipe as mp
import pygame
import time
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480  # Define the size of the Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw with Nose")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables
nose_position = None
previous_nose_position = None  # Track previous nose position
drawing_points = []  # Store all the points to draw lines

def update_nose_position(result, output_image, timestamp_ms):
    global nose_position, previous_nose_position, drawing_points
    face_landmarks_list = result.face_landmarks
    if len(face_landmarks_list) > 0:
        face_landmarks = face_landmarks_list[0]
        nose_landmark = face_landmarks[4]  # Nose is landmark index 4 (tip of the nose)
        nose_position = (int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))  # Scale to window size

        # Only start drawing if the nose moves a significant amount
        if previous_nose_position and nose_position != previous_nose_position:
            drawing_points.append(nose_position)  # Add the current position to the drawing points list

        previous_nose_position = nose_position  # Update the previous position

# Initialize MediaPipe Face Landmarker with the task file
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./face_landmarker.task'),  # Correct model path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_nose_position)

detector = FaceLandmarker.create_from_options(options)

# Start webcam capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image_rgb))

    # Perform face landmark detection
    frame_timestamp_ms = int(round(time.time() * 1000))
    detection_result = detector.detect_async(mp_image, frame_timestamp_ms)

    # Update Pygame screen
    screen.fill(WHITE)  # Clear screen with white background

    # Draw all points in the drawing_points list
    if drawing_points:
        for i in range(1, len(drawing_points)):
            pygame.draw.line(screen, RED, drawing_points[i - 1], drawing_points[i], 2)  # Draw a red line

    # Check for user quit (closing window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    # Update Pygame display
    pygame.display.update()

    # Show the webcam feed in a window (optional)
    cv2.imshow("MediaPipe Face", image)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
pygame.quit()
