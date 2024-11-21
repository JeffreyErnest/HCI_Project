import cv2
import mediapipe as mp
import pygame
import time
import numpy as np

# Initialize Pygame
pygame.init()

# Get screen size for fullscreen
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h  # Define the size of the Pygame window (fullscreen)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE) # create maximized window
pygame.display.set_caption("Draw with Nose")

# Colors (Expanded Palette)
WHITE = (255, 255, 255)  # White color
GREY = (200, 200, 200)
BLACK = (0, 0, 0)

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
brush_sizes = [2, 5, 10, 15]
selected_option = None

current_color_index = 0  # Start with the first color (Red)
current_color = colors[current_color_index]
current_brush_size = brush_sizes[0]  # Default brush size

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Global variables
nose_position = None
previous_nose_position = None  # Track previous nose position
drawing_segments = []  # Store all the drawing segments, each with color and brush size
mouth_open = False  # Track the mouth open state
last_mouth_state = False  # To track the transition of mouth open/close
canChange = True
lastTime = 0
dropdown_open = False
cam_width, cam_height = 275, 160 #Size of webcam feed display
cam_x = WIDTH - cam_width - 10 # 10px margin from right
cam_y = 10 # 10px margin from top

dropdown_rect = pygame.Rect(10, 10, 150, 40)
button_rect = pygame.Rect(450, 50, 50, 50)
color_box_size = 30
dropdown_height = len(colors) * color_box_size + len(brush_sizes) * color_box_size
drawing_segments = []

MAX_ITEMS_PER_ROW = 10  # Max items per row
ITEM_SPACING = 40      # Space between items
ITEM_SIZE = 30 

# Initialize font for rendering text
font = pygame.font.Font(None, 36)

# Function to convert the OpenCV image to a Pygame surface instead for webcam
def to_pygame(image):
    # Convert from BGR to RGB instead of Pygame
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to Pygame surface now
    return pygame.surfarray.make_surface(image_rgb)

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

# Function to check if nose is hovering over a color swatch
def is_hovering_over_color(nose_pos, color_rects):
    for i, color_rect in enumerate(color_rects):
        if color_rect.collidepoint(nose_pos):
            return i  # Return the index of the hovered color
    return -1  # No color hovered

# Function to check if nose is hovering over a brush size
def is_hovering_over_brush_size(nose_pos, brush_size_rects):
    for i, size_rect in enumerate(brush_size_rects):
        if size_rect.collidepoint(nose_pos):
            return i  # Return the index of the hovered brush size
    return -1  # No brush size hovered

# Initialize MediaPipe Face Mesh with the task file
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    current_color = colors[current_color_index]

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flip it for display
        flipped_image = cv2.flip(image, 1)
        
        # Perform face mesh detection
        results = face_mesh.process(image_rgb)

        # Update Pygame screen
        screen.fill(WHITE)  # Clear screen with white background

        # Draw current color in top-left corner
        pygame.draw.rect(screen, current_color, (10, 10, 50, 50))  # Draw color swatch
        color_text = font.render("Current Color", True, BLACK)
        screen.blit(color_text, (70, 10))  # Display label text

        # Draw webcam in fixed top-left corner
        corner_cam = to_pygame(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        screen.blit(pygame.transform.scale(corner_cam, (cam_width, cam_height)), (cam_x, cam_y)) # resize + position
        
        color_rects = []  # List to store the color rectangles for hover checking
        brush_size_rects = []  # List to store the brush size rectangles for hover checking
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the position of the nose (tip of the nose is landmark index 1)
                nose_landmark = face_landmarks.landmark[1]
            
                # Flip the x-coordinate so that the drawing mirrors your motion irl
                nose_position = (WIDTH - int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))

                # Only start drawing if the nose moves a significant amount
                if not dropdown_open and previous_nose_position and np.linalg.norm(np.array(nose_position) - np.array(previous_nose_position)) > 10:
                    drawing_segments.append((previous_nose_position, nose_position, current_color, current_brush_size))

                previous_nose_position = nose_position  # Update the previous position

                # Check if the mouth is open or closed
                mouth_open = is_mouth_open(face_landmarks.landmark)

                if not canChange:
                    currTime = time.time()
                    if currTime - lastTime >= 1:
                        canChange = True

                # Detect transition from closed to open mouth (for color cycling)
                if mouth_open and canChange and not last_mouth_state:
                    dropdown_open = not dropdown_open
                    canChange = False
                    lastTime = time.time()

                # Hover detection logic for colors
                if dropdown_open:
                    pygame.draw.rect(screen, GREY, dropdown_rect)

                    # Draw color swatches and check if nose is hovering over any color
                    for i, color in enumerate(colors):
                        # Calculate row and column for this item
                        row = i // MAX_ITEMS_PER_ROW
                        col = i % MAX_ITEMS_PER_ROW

                        color_rect = pygame.Rect(
                            dropdown_rect.x + col * ITEM_SPACING + 10,
                            dropdown_rect.y + row * ITEM_SPACING + 10,
                            ITEM_SIZE, ITEM_SIZE
                        )
                        pygame.draw.rect(screen, color, color_rect)
                        pygame.draw.rect(screen, BLACK, color_rect, 2)
                        
                        color_rects.append(color_rect)  # Add rect for hover detection

                    # Check if the nose is hovering over any color
                    hovered_index = is_hovering_over_color(nose_position, color_rects)
                    if hovered_index != -1:
                        current_color = colors[hovered_index]  # Change the color to the hovered one

                    # Draw brush size options below the colors
                    num_color_rows = (len(colors) + MAX_ITEMS_PER_ROW - 1) // MAX_ITEMS_PER_ROW
                    brush_offset_y = dropdown_rect.y + num_color_rows * ITEM_SPACING + 10

                    for i, size in enumerate(brush_sizes):
                        size_rect = pygame.Rect(
                            dropdown_rect.x + 10, 
                            brush_offset_y + (i * ITEM_SPACING),  # Adjust position dynamically
                            dropdown_rect.width - 20, ITEM_SIZE   # Brush size height
                        )
                        pygame.draw.rect(screen, WHITE, size_rect)
                        size_text = font.render(f"Size {size}", True, BLACK)
                        screen.blit(size_text, (size_rect.x + 5, size_rect.y + 5))

                        brush_size_rects.append(size_rect)  # Add rect for hover detection

                # Check if the nose is hovering over any brush size
                hovered_brush_size = is_hovering_over_brush_size(nose_position, brush_size_rects)
                if hovered_brush_size != -1:
                    current_brush_size = brush_sizes[hovered_brush_size]  # Change the brush size to the hovered one

                # Draw lines based on previous nose position with their respective sizes
                for segment in drawing_segments:
                    pygame.draw.line(screen, segment[2], segment[0], segment[1], segment[3])  # Use stored brush size for each segment
                
            last_mouth_state = mouth_open  # Update mouth state

            if nose_position:
                pygame.draw.circle(screen, BLACK, nose_position, 5)
        
        # Update the display
        pygame.display.update()

        # Handle quitting the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                quit()
