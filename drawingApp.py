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
mp_drawing = mp.solutions.drawing_utils

# Global variables
nose_position = None
previous_nose_position = None  # Track previous nose position
drawing_segments = []  # Store all the drawing segments, each with color
mouth_open = False  # Track the mouth open state
last_mouth_state = False  # To track the transition of mouth open/close

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

# Initialize MediaPipe Face Mesh with the task file
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # Start webcam capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform face mesh detection
        results = face_mesh.process(image_rgb)

        # Update Pygame screen
        screen.fill(WHITE)  # Clear screen with white background
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the position of the nose (tip of the nose is landmark index 1)
                nose_landmark = face_landmarks.landmark[1]
                nose_position = (int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))

                # Only start drawing if the nose moves a significant amount
                if previous_nose_position and nose_position != previous_nose_position:
                    drawing_segments.append((previous_nose_position, nose_position, current_color))

                previous_nose_position = nose_position  # Update the previous position

                # Check if the mouth is open or closed
                mouth_open = is_mouth_open(face_landmarks.landmark)

                # Detect transition from closed to open mouth (for color cycling)
                if mouth_open and not last_mouth_state:
                    # Cycle to the next color
                    current_color_index = (current_color_index + 1) % len(colors)
                    current_color = colors[current_color_index]
                
                last_mouth_state = mouth_open  # Update last mouth state

        # Draw all the segments with their respective colors
        for start_point, end_point, color in drawing_segments:
            pygame.draw.line(screen, color, start_point, end_point, 2)

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
