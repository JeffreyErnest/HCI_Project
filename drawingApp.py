#pip install pygame

import cv2
import mediapipe as mp
import pygame 
import time
import numpy as np
import math

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
COLOR_WHEEL_RADIUS = 50
selecting_color = False
color_wheel_center = (0, 0)  # This will be updated dynamically
BRUSH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32]  # Different brush sizes in pixels
current_brush_size = 2  # Default brush size
selecting_brush_size = False
brush_wheel_radius = 80  # Slightly smaller than color wheel

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

def draw_color_wheel(center):
    for angle in range(360):
        rad = math.radians(angle)
        color = pygame.Color(0)
        color.hsva = (angle, 100, 100, 100)
        
        start_pos = (
            center[0] + (COLOR_WHEEL_RADIUS - 20) * math.cos(rad),
            center[1] + (COLOR_WHEEL_RADIUS - 20) * math.sin(rad)
        )
        end_pos = (
            center[0] + COLOR_WHEEL_RADIUS * math.cos(rad),
            center[1] + COLOR_WHEEL_RADIUS * math.sin(rad)
        )
        pygame.draw.line(screen, color, start_pos, end_pos, 2)

def get_color_from_wheel(pos, center):
    # Calculate angle and distance from wheel center
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Check if click is within wheel radius
    if distance <= COLOR_WHEEL_RADIUS and distance >= (COLOR_WHEEL_RADIUS - 20):
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
            
        # Convert angle to color
        color = pygame.Color(0)
        color.hsva = (angle, 100, 100, 100)
        return color.r, color.g, color.b
    return None

# Add this function to detect head tilt
def get_head_tilt(landmarks):
    # Get left and right eye landmarks
    left_eye = landmarks[33]  # Adjust index based on your MediaPipe face mesh points
    right_eye = landmarks[263]
    
    # Calculate tilt angle
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = math.degrees(math.atan2(dy, dx))
    
    # Return True if head is tilted left (negative angle beyond threshold)
    return angle < -15  # Adjust threshold as needed

# Add function to draw brush size wheel
def draw_brush_size_wheel(center):
    # Draw background circle
    pygame.draw.circle(screen, (200, 200, 200), center, brush_wheel_radius)
    
    # Calculate positions for each brush size option
    num_sizes = len(BRUSH_SIZES)
    for i, size in enumerate(BRUSH_SIZES):
        angle = (i * (360 / num_sizes)) - 90  # Start from top
        rad = math.radians(angle)
        
        # Calculate position for this size option
        pos_x = center[0] + (brush_wheel_radius * 0.7) * math.cos(rad)
        pos_y = center[1] + (brush_wheel_radius * 0.7) * math.sin(rad)
        
        # Draw circle representing brush size
        pygame.draw.circle(screen, (0, 0, 0), (int(pos_x), int(pos_y)), size // 2)
        
        # Draw selection indicator if this is current size
        if size == current_brush_size:
            pygame.draw.circle(screen, (255, 0, 0), (int(pos_x), int(pos_y)), size // 2 + 2, 2)

def get_brush_size_from_wheel(pos, center):
    # Calculate angle from center to position
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance <= brush_wheel_radius:
        angle = math.degrees(math.atan2(dy, dx)) + 90  # Adjust to match our wheel layout
        if angle < 0:
            angle += 360
            
        # Convert angle to brush size index
        size_index = int((angle / 360) * len(BRUSH_SIZES))
        if size_index >= len(BRUSH_SIZES):
            size_index = 0
            
        return BRUSH_SIZES[size_index]
    return None

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
                    # Get nose position first
                    nose_landmark = face_landmarks.landmark[1]
                    nose_position = (WIDTH - int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))
                    
                    mouth_open = is_mouth_open(face_landmarks.landmark)
                    head_tilted = get_head_tilt(face_landmarks.landmark)
                    
                    if mouth_open and not last_mouth_state:
                        selecting_color = True
                        selecting_brush_size = False
                        color_wheel_center = nose_position
                    elif not mouth_open and last_mouth_state:
                        selecting_color = False
                        selecting_brush_size = False
                    
                    if selecting_color:
                        if head_tilted:
                            selecting_brush_size = True
                            selecting_color = False
                        else:
                            draw_color_wheel(color_wheel_center)
                            new_color = get_color_from_wheel(nose_position, color_wheel_center)
                            if new_color:
                                current_color = new_color
                    
                    if selecting_brush_size:
                        draw_brush_size_wheel(color_wheel_center)
                        new_size = get_brush_size_from_wheel(nose_position, color_wheel_center)
                        if new_size:
                            current_brush_size = new_size
                    
                    last_mouth_state = mouth_open

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
            for start_point, end_point, color, size in drawing_segments:
                # Calculate points between start and end to make smooth line
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Calculate how many circles we need to fill the gap
                steps = max(int(distance / (size/4)), 1)
                
                for i in range(steps + 1):
                    x = start_point[0] + (dx * i / steps)
                    y = start_point[1] + (dy * i / steps)
                    pygame.draw.circle(screen, color, (int(x), int(y)), size//2)

            # Handle nose mode drawing
            if drawing_mode == "nose" and results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    nose_landmark = face_landmarks.landmark[1]
                    nose_position = (WIDTH - int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))
                    
                    # Always draw cursor dot
                    pygame.draw.circle(screen, current_color, nose_position, current_brush_size//2)
                    
                    # Only add line segments when drawing is active
                    if drawing_active:
                        if previous_nose_position:
                            drawing_segments.append((previous_nose_position, nose_position, current_color, current_brush_size))
                        previous_nose_position = nose_position

            # Handle hand mode drawing
            if drawing_mode == "hand" and results_hands.multi_hand_landmarks:
                hand_landmarks = results_hands.multi_hand_landmarks[0]
                index_finger = hand_landmarks.landmark[8]
                hand_position = (WIDTH - int(index_finger.x * WIDTH), int(index_finger.y * HEIGHT))
                
                # Always draw cursor dot
                pygame.draw.circle(screen, current_color, hand_position, current_brush_size//2)
                
                if drawing_active:
                    if previous_hand_position:
                        drawing_segments.append((previous_hand_position, hand_position, current_color, current_brush_size))
                    previous_hand_position = hand_position

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
