import cv2
import mediapipe as mp
import pygame  # type: ignore
import time
import numpy as np
import math
from pynput import keyboard
from PIL import ImageGrab

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h  # Window size
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Drawing App")

# Color palette
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 165, 0), (128, 0, 128),
    (255, 20, 147), (255, 99, 71), (0, 0, 139), (34, 139, 34),
    (238, 130, 238), (75, 0, 130), (255, 105, 180), (0, 255, 127),
    (255, 69, 0), (240, 230, 140), (0, 191, 255), (220, 20, 60),
    (255, 228, 196), (70, 130, 180), (255, 222, 173), (250, 128, 114),
    (255, 215, 0), (255, 140, 0), (218, 165, 32), (255, 182, 193),
    (102, 205, 170), (255, 240, 245), (70, 130, 180), (186, 85, 211),
    (255, 255, 240), (255, 248, 220), (255, 182, 193), (240, 128, 128),
    (255, 0, 255), (128, 128, 128), (169, 169, 169), (0, 0, 0),
    (255, 255, 255)
]
brush_sizes = [2, 5, 10, 15]
current_color_index = 0  # Start with Red
current_color = colors[current_color_index]
current_brush_size = brush_sizes[0]  # Default size

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables
nose_position = None
previous_nose_position = None
drawing_segments = []  # Store drawing segments
mouth_open = False
last_mouth_state = False
hand_position = None
previous_hand_position = None
drawing_mode = "nose"  # Drawing mode
drawing_active = False
COLOR_WHEEL_RADIUS = 50
selecting_color = False
color_wheel_center = (0, 0)  # Dynamic center
BRUSH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32]  # Brush sizes
current_brush_size = BRUSH_SIZES[len(BRUSH_SIZES)//2]  # Middle size
selecting_brush_size = False
brush_wheel_radius = 80  # Brush size wheel radius

# Webcam settings
WEBCAM_SCALE = 0.2
WEBCAM_MARGIN = 10
cam_width, cam_height = 275, 160
cam_x = WIDTH - cam_width - 10
cam_y = 10

drawing_history = []  # Store drawing segments
undo_history = []  # Store undone segments
last_mouth_close_time = 0  # Last mouth close time
UNDO_REDO_DELAY = 2  # Delay for undo/redo
is_eraser_mode = False  # Eraser mode flag
ORIGINAL_COLOR = None  # Store original color for eraser
mouth_open_count = 0  # Count of mouth opens

# Save button settings
save_button_width = 150
save_button_height = 50
save_button_color = (0, 128, 255)
save_button_hover_color = (0, 100, 200)
save_button_text_color = (255, 255, 255)
save_button_font = pygame.font.Font(None, 36)

# Check if mouth is open
def is_mouth_open(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    mouth_distance = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
    mouth_threshold = 0.03  # Threshold for mouth open
    return mouth_distance > mouth_threshold

# Draw color wheel
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

# Get color from wheel based on position
def get_color_from_wheel(pos, center):
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance <= COLOR_WHEEL_RADIUS and distance >= (COLOR_WHEEL_RADIUS - 20):
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        color = pygame.Color(0)
        color.hsva = (angle, 100, 100, 100)
        return color.r, color.g, color.b
    elif distance < (COLOR_WHEEL_RADIUS - 20): # inside of radius
        return BLACK
    return None

# Detect head tilt
def get_head_tilt(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = math.degrees(math.atan2(dy, dx))
    
    if angle < -15:  # Left tilt
        return -1
    elif angle > 15:  # Right tilt
        return 1
    return 0

# Draw brush size wheel
def draw_brush_size_wheel(center):
    pygame.draw.circle(screen, (200, 200, 200), center, brush_wheel_radius)
    num_sizes = len(BRUSH_SIZES)
    for i, size in enumerate(BRUSH_SIZES):
        angle = (i * (360 / num_sizes)) - 90
        rad = math.radians(angle)
        pos_x = center[0] + (brush_wheel_radius * 0.7) * math.cos(rad)
        pos_y = center[1] + (brush_wheel_radius * 0.7) * math.sin(rad)
        pygame.draw.circle(screen, (0, 0, 0), (int(pos_x), int(pos_y)), size // 2)
        if size == current_brush_size:
            pygame.draw.circle(screen, (255, 0, 0), (int(pos_x), int(pos_y)), size // 2 + 2, 2)

# Get brush size from wheel based on position
def get_brush_size_from_wheel(pos, center):
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance <= brush_wheel_radius:
        angle = math.degrees(math.atan2(dy, dx)) + 90
        if angle < 0:
            angle += 360
        size_index = int((angle / 360) * len(BRUSH_SIZES))
        if size_index >= len(BRUSH_SIZES):
            size_index = 0
        return BRUSH_SIZES[size_index]
    return None

# Update camera position
def update_cam_position():
    global cam_x, cam_y
    cam_x = WIDTH - cam_width - WEBCAM_MARGIN
    cam_y = WEBCAM_MARGIN

# Convert image from BGR to Pygame surface
def to_pygame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(image_rgb)

# Draw save button
def draw_save_button(x, y):
    button_rect = pygame.Rect(x, y, save_button_width, save_button_height)
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_x, mouse_y):
        pygame.draw.rect(screen, save_button_hover_color, button_rect)
    else:
        pygame.draw.rect(screen, save_button_color, button_rect)
    
    text = save_button_font.render("Save", True, save_button_text_color)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    return button_rect

# Draw exit button
def draw_exit_button(x, y):
    button_rect = pygame.Rect(x, y, save_button_width, save_button_height)
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_x, mouse_y):
        pygame.draw.rect(screen, save_button_hover_color, button_rect)
    else:
        pygame.draw.rect(screen, save_button_color, button_rect)
    
    text = save_button_font.render("Exit", True, save_button_text_color)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    return button_rect

# Display instructions
def display_instructions():
    #create separate surface for popup
    popup_surface = pygame.Surface((WIDTH, HEIGHT))
    popup_surface.fill(GREY)

    #setup fonts
    font = pygame.font.Font("Jefffont-Regular.ttf", 36)
    title_font = pygame.font.Font(None, 50)

    # Text for the instructions
    title_text = title_font.render("Welcome to ClickCanvas!", True, (0, 0, 0))
    instructions = [
        "Instructions:",
        "; Move your nose or hand to draw on the canvas.",
        "      ; Please try drawing slowly as fast movements may not be caught.",
        "; Hold \"Space\" to draw. Let go of \"space\" to stop drawing.",
        "; Press \"Enter\" to switch between nose and hand mode.",
        "; Open your mouth to select a color. Select black from center of wheel.",
        "      ; Tilt right while selecting color to change brush size. Select size from menu.",
        "      ; Tilt left while selecting color to toggle eraser.",
        "; Tilt your head right to undo or tilt left to redo. Do NOT open your mouth.",
        "",
        "Press TAB to start drawing!"
    ]

     # Render instructions line by line
    text_surfaces = [font.render(line, True, (0, 0, 0)) for line in instructions]

    # Positioning
    title_pos = (WIDTH // 2 - title_text.get_width() // 2, 50)
    text_positions = [(50, 150 + i * 40) for i in range(len(text_surfaces))]

    # Draw all text
    popup_surface.blit(title_text, title_pos)
    for text_surface, pos in zip(text_surfaces, text_positions):
        popup_surface.blit(text_surface, pos)

    # Main event loop for the popup
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    running = False  # Exit the popup loop

        # Display the popup surface on the screen
        screen.blit(popup_surface, (0, 0))
        pygame.display.update()

# Show instructions popup
display_instructions()

try:
    # Initialize MediaPipe modules
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                flipped_image = cv2.flip(image, 1)
                
                results_face = face_mesh.process(image_rgb)
                results_hands = hands.process(image_rgb)

                screen.fill(WHITE)  # Clear screen
                pygame.draw.rect(screen, current_color, (10, 10, 50, 50))  # Color swatch
                corner_cam = cv2.resize(image, (cam_width, cam_height))  # Resize feed
                corner_cam_surface = pygame.surfarray.make_surface(corner_cam)

                rotated_x = cam_x - cam_width - WEBCAM_MARGIN 
                rotated_cam = to_pygame(cv2.rotate(corner_cam, cv2.ROTATE_90_COUNTERCLOCKWISE))
                screen.blit(pygame.transform.scale(rotated_cam, (cam_width, cam_height)), (cam_x, cam_y))  # Display

                # Draw save and exit buttons
                save_button_x = WIDTH - save_button_width - 40
                save_button_y = HEIGHT - save_button_height - 40
                exit_button_x = WIDTH - save_button_width - save_button_x
                exit_button_y = HEIGHT - save_button_height - 40  
                
                save_button_rect = draw_save_button(save_button_x, save_button_y)
                exit_button_rect = draw_exit_button(exit_button_x, exit_button_y)

                # Draw existing segments
                for start_point, end_point, color, size in drawing_segments:
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    steps = max(int(distance / (size/4)), 1)
                    
                    for i in range(steps + 1):
                        x = start_point[0] + (dx * i / steps)
                        y = start_point[1] + (dy * i / steps)
                        pygame.draw.circle(screen, color, (int(x), int(y)), size//2)

                # Handle face detection for color changing
                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        nose_landmark = face_landmarks.landmark[1]
                        nose_position = (WIDTH - int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))
                        
                        mouth_open = is_mouth_open(face_landmarks.landmark)
                        head_tilt = get_head_tilt(face_landmarks.landmark)
                        
                        current_time = time.time()
                        
                        if mouth_open and not last_mouth_state:
                            mouth_open_count += 1
                            if mouth_open_count == 1:
                                selecting_color = True
                                selecting_brush_size = False
                                color_wheel_center = nose_position if drawing_mode == "nose" else hand_position
                            elif mouth_open_count == 2:
                                selecting_color = False
                                selecting_brush_size = False
                                mouth_open_count = 0

                        elif not mouth_open and last_mouth_state:
                            if mouth_open_count == 1:
                                last_mouth_close_time = current_time

                        # Handle undo/redo
                        if not mouth_open and (current_time - last_mouth_close_time) > UNDO_REDO_DELAY:
                            if head_tilt == 1 and drawing_segments:
                                last_segment = drawing_segments.pop()
                                undo_history.append(last_segment)
                            elif head_tilt == -1 and undo_history:
                                last_undone = undo_history.pop()
                                drawing_segments.append(last_undone)
                        
                        # Draw wheels
                        if selecting_color:
                            if head_tilt == -1:
                                selecting_brush_size = True
                                selecting_color = False
                            elif head_tilt == 1:
                                is_eraser_mode = not is_eraser_mode
                                current_color = WHITE if is_eraser_mode else ORIGINAL_COLOR
                                selecting_color = False
                            else:
                                draw_color_wheel(color_wheel_center)
                                position_for_wheel = hand_position if drawing_mode == "hand" else nose_position
                                new_color = get_color_from_wheel(position_for_wheel, color_wheel_center)
                                if new_color:
                                    current_color = new_color
                                    is_eraser_mode = False
                        
                        if selecting_brush_size:
                            draw_brush_size_wheel(color_wheel_center)
                            position_for_wheel = hand_position if drawing_mode == "hand" else nose_position
                            new_size = get_brush_size_from_wheel(position_for_wheel, color_wheel_center)
                            if new_size:
                                current_brush_size = new_size
                        
                        last_mouth_state = mouth_open

                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cap.release()
                        pygame.quit()
                        exit()
                    elif event.type == pygame.VIDEORESIZE:
                        WIDTH, HEIGHT = event.w, event.h
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                        update_cam_position()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            drawing_mode = "hand" if drawing_mode == "nose" else "nose"
                            previous_hand_position = None
                            previous_nose_position = None
                        elif event.key == pygame.K_SPACE:
                            if nose_position is not None and save_button_rect.collidepoint(nose_position) and drawing_mode == "nose":
                                screenshot = ImageGrab.grab()
                                screenshot.save("screenshot.png")
                                print("Drawing Was Saved as 'screenshot.png'")
                            elif hand_position is not None and save_button_rect.collidepoint(hand_position) and drawing_mode == "hand":
                                screenshot = ImageGrab.grab()
                                screenshot.save("screenshot.png")
                                print("Drawing Was Saved as 'screenshot.png'")
                            elif nose_position is not None and exit_button_rect.collidepoint(nose_position) and drawing_mode == "nose":
                                cap.release()
                                cv2.destroyAllWindows()
                                pygame.quit()
                            elif hand_position is not None and exit_button_rect.collidepoint(hand_position) and drawing_mode == "hand":
                                cap.release()
                                cv2.destroyAllWindows()
                                pygame.quit()
                            else:
                                drawing_active = True
                    elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                        drawing_active = False
                        previous_hand_position = None
                        previous_nose_position = None

                # Drawing logic
                if drawing_mode == "nose" and results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        nose_landmark = face_landmarks.landmark[1]
                        nose_position = (WIDTH - int(nose_landmark.x * WIDTH), int(nose_landmark.y * HEIGHT))
                        pygame.draw.circle(screen, BLACK, nose_position, current_brush_size//2 + 2)
                        pygame.draw.circle(screen, current_color, nose_position, current_brush_size//2)
                        if drawing_active:
                            if previous_nose_position:
                                new_segment = (previous_nose_position, nose_position, current_color, current_brush_size)
                                drawing_segments.append(new_segment)
                                undo_history.clear()
                            previous_nose_position = nose_position

                if drawing_mode == "hand" and results_hands.multi_hand_landmarks:
                    hand_landmarks = results_hands.multi_hand_landmarks[0]
                    index_finger = hand_landmarks.landmark[8]
                    hand_position = (WIDTH - int(index_finger.x * WIDTH), int(index_finger.y * HEIGHT))
                    pygame.draw.circle(screen, BLACK, hand_position, current_brush_size//2 + 2)
                    pygame.draw.circle(screen, current_color, hand_position, current_brush_size//2)
                    if drawing_active:
                        if previous_hand_position:
                            new_segment = (previous_hand_position, hand_position, current_color, current_brush_size)
                            drawing_segments.append(new_segment)
                            undo_history.clear()
                        previous_hand_position = hand_position

                # Update mode display
                font = pygame.font.Font(None, 36)
                mode_text = font.render(f"Mode: {drawing_mode.capitalize()} {'(Eraser)' if is_eraser_mode else 'Drawing'}", True, (0, 0, 0))
                screen.blit(mode_text, (70, 10))

                # Update Pygame display
                pygame.display.update()

                # Exit on ESC
                if cv2.waitKey(5) & 0xFF == 27:
                    break

except KeyboardInterrupt:
    print("\nProgram closed by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
