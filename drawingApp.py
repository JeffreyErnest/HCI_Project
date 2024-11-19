#pip install pygame

import cv2
import mediapipe as mp
import pygame
import time
import numpy as np

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480  # Define the size of the Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw with Nose")

# Colors (Expanded Palette)
WHITE = (255, 255, 255)  # White color

colors = [
    (255, 0, 0),           # Red
    (0, 255, 0),           # Green
    (0, 0, 255),           # Blue
    (255, 255, 0),         # Yellow
    (0, 255, 255),         # Cyan
    (255, 0, 255),         # Magenta
    (255, 165, 0),         # Orange
    (128, 0, 128),         # Purple
    (255, 20, 147),        # Deep Pink
    (255, 99, 71),         # Tomato
    (0, 0, 139),           # Dark Blue
    (34, 139, 34),         # Forest Green
    (238, 130, 238),       # Violet
    (75, 0, 130),          # Indigo
    (255, 105, 180),       # Hot Pink
    (0, 255, 127),         # Spring Green
    (255, 69, 0),          # Red-Orange
    (240, 230, 140),       # Khaki
    (0, 191, 255),         # Deep Sky Blue
    (220, 20, 60),         # Crimson
    (255, 228, 196),       # Bisque
    (70, 130, 180),        # Steel Blue
    (255, 222, 173),       # Navajo White
    (250, 128, 114),       # Salmon
    (255, 215, 0),         # Gold
    (255, 140, 0),         # Dark Orange
    (218, 165, 32),        # Goldenrod
    (255, 182, 193),       # Light Pink
    (102, 205, 170),       # Medium Aquamarine
    (255, 240, 245),       # Lavender Blush
    (70, 130, 180),        # Steel Blue
    (186, 85, 211),        # Medium Orchid
    (255, 255, 240),       # Ivory
    (255, 248, 220),       # Cornsilk
    (255, 182, 193),       # Light Pink
    (240, 128, 128),       # Light Coral
    (255, 0, 255),         # Magenta
    (128, 128, 128),       # Gray
    (169, 169, 169),       # Dark Gray
    (0, 0, 0),             # Black
    (255, 255, 255)        # White
]

current_color_index = 0  # Start with the first color (Red)
current_color = colors[current_color_index]

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands  # Add hands module
mp_drawing = mp.solutions.drawing_utils

# Global variables
nose_position = None
previous_nose_position = None  # Track previous nose position
drawing_segments = []  # Store all the drawing segments, each with color
mouth_open = False  # Track the mouth open state
last_mouth_state = False  # To track the transition of mouth open/close
hand_position = None  # Add hand position tracking
previous_hand_position = None
drawing_mode = "nose"  # Add mode switching
drawing_active = False  # Add a new global variable to track drawing state

# Helper function to check if the mouth is open
def is_mouth_open(landmarks):
    # Get the coordinates of the top and bottom lip landmarks
    top_lip = landmarks[13]  # Landmark for the top lip (index 13)
    bottom_lip = landmarks[14]  # Landmark for the bottom lip (index 14)
    
    # Calculate the distance between the top and bottom lip
    mouth_distance = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
    
    # Return True if the mouth distance exceeds a threshold (indicating mouth is open)
    mouth_threshold = 0.03  # Adjust this threshold based on testing
    return mouth_distance > mouth_threshold

# Initialize both MediaPipe modules
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Start webcam capture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Flip it for display
            flipped_image = cv2.flip(image, 1)
            
            # Process both face and hands
            results_face = face_mesh.process(image_rgb)
            results_hands = hands.process(image_rgb)

            # Update Pygame screen
            screen.fill(WHITE)  # Clear screen with white background
            
            # Handle face detection for color changing
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Check mouth state for color changing
                    mouth_open = is_mouth_open(face_landmarks.landmark)
                    
                    if mouth_open and not last_mouth_state:
                        current_color_index = (current_color_index + 1) % len(colors)
                        current_color = colors[current_color_index]
                    
                    last_mouth_state = mouth_open

            # Debug information
            if drawing_mode == "nose":
                if results_face.multi_face_landmarks:
                    print("Face detected")
                else:
                    print("No face detected")
            else:  # hand mode
                if results_hands.multi_hand_landmarks:
                    print("Hand detected")
                else:
                    print("No hand detected")
            
            print(f"Drawing active: {drawing_active}")

            # Move event handling BEFORE the drawing code and remove duplicate event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        drawing_mode = "hand" if drawing_mode == "nose" else "nose"
                        print(f"Switched to {drawing_mode} mode")
                        # Reset positions when switching modes
                        previous_hand_position = None
                        previous_nose_position = None
                    elif event.key == pygame.K_SPACE:
                        drawing_active = True
                        print("Space pressed - Drawing activated")
                elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                    drawing_active = False
                    print("Space released - Drawing deactivated")
                    # Reset positions when stopping drawing
                    previous_hand_position = None
                    previous_nose_position = None

            # Draw all existing segments
            for start_point, end_point, color in drawing_segments:
                pygame.draw.line(screen, color, start_point, end_point, 2)

            # Handle nose mode drawing
            if drawing_mode == "nose" and results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    nose_landmark = face_landmarks.landmark[1]
                    nose_position = (WIDTH - int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))
                    
                    # Always draw cursor dot
                    pygame.draw.circle(screen, current_color, nose_position, 5)
                    
                    # Only add line segments when drawing is active
                    if drawing_active:
                        if previous_nose_position:
                            drawing_segments.append((previous_nose_position, nose_position, current_color))
                        previous_nose_position = nose_position
                    else:
                        previous_nose_position = None

            # Handle hand mode drawing
            if drawing_mode == "hand" and results_hands.multi_hand_landmarks:
                hand_landmarks = results_hands.multi_hand_landmarks[0]
                index_finger = hand_landmarks.landmark[8]
                hand_position = (WIDTH - int(index_finger.x * WIDTH), int(index_finger.y * HEIGHT))
                
                # Always draw cursor dot
                pygame.draw.circle(screen, current_color, hand_position, 5)
                
                # Only add line segments when drawing is active
                if drawing_active:
                    if previous_hand_position:
                        drawing_segments.append((previous_hand_position, hand_position, current_color))
                    previous_hand_position = hand_position
                else:
                    previous_hand_position = None

            # Display the current color on the screen
            font = pygame.font.Font(None, 36)
            color_name = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Orange", "Purple", 
                          "Deep Pink", "Tomato", "Dark Blue", "Forest Green", "Violet", "Indigo", 
                          "Hot Pink", "Spring Green", "Red-Orange", "Khaki", "Deep Sky Blue", "Crimson", 
                          "Bisque", "Steel Blue", "Navajo White", "Salmon", "Gold", "Dark Orange", 
                          "Goldenrod", "Light Pink", "Medium Aquamarine", "Lavender Blush", "Medium Orchid", 
                          "Ivory", "Cornsilk", "Light Coral", "Gray", "Black", "White"][current_color_index]  # Get color name for display
            color_text = font.render(f"Current Color: {color_name}", True, (0, 0, 0))  # Black text
            screen.blit(color_text, (10, 10))

            # Update mode display
            mode_text = font.render(f"Mode: {drawing_mode.capitalize()} Drawing", True, (0, 0, 0))
            screen.blit(mode_text, (10, 50))

            # Update Pygame display
            pygame.display.update()

            # Show the flipped webcam feed in a window
            cv2.imshow("MediaPipe Face", flipped_image)

            # Exit on ESC
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        pygame.quit()
