#Jeffrey Ernest, 9/28/2024
#Homework 4, Human Computer Interaction, section 01
#Description: This program determines if a user is 
#             nodding or shaking their head
#             yes or no.

# imports
import cv2
import mediapipe as mp
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# used to prevent printing for single head movment
toggle_count = 0 

# used to save previous values to compare to
previous_x4 = 0
previous_y4 = 0

# used as flags for printing only once
head_shake = False  
head_nod = False  

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

RESULT = None

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('face landmarker result: {}'.format(result))
    global RESULT
    RESULT = result

# Create a PoseLandmarker object
base_options = BaseOptions(model_asset_path='face_landmarker.task')
options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

detector = FaceLandmarker.create_from_options(options)

# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    
    frame_timestamp_ms = int(round(time.time()*1000))
    #print("Timestamp:", frame_timestamp_ms)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))

    image = mp_image.numpy_view()
    # detect pose landmarks from image
    detection_result = detector.detect_async(mp_image, frame_timestamp_ms)

    # Draw the pose annotation on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if type(RESULT) is not type(None):
      annotated_image = draw_landmarks_on_image(image, RESULT)
      face_landmarks_list = RESULT.face_landmarks
      if len(face_landmarks_list) > 0:
        
        h, w, _ = image.shape

        face_landmarks = face_landmarks_list[0]
        
      # gets position of nose
      middle_of_nose = face_landmarks[4]
      x4, y4 = int(middle_of_nose.x * w), int(middle_of_nose.y * h)

      if (x4 > previous_x4 + 5): #checks if head has moved left
        if not head_shake:
          toggle_count += 1
          head_shake = True  # Set to True
          if toggle_count == 3: 
            print("No")
            toggle_count = 0 

      if (x4 < previous_x4 - 5): #checks if head has moved right
        if head_shake:
          toggle_count += 1  
          head_shake = False 
          if toggle_count == 3: 
            print("No")
            toggle_count = 0  
        
      if (y4 > previous_y4 + 10): #checks if head has moved up
        if not head_nod:
          toggle_count += 1  
          head_nod = True 
          if toggle_count == 3:  
            print("Yes")
            toggle_count = 0 

      if (y4 < previous_y4 - 10): #checks if head has moved down
        if head_nod:
          toggle_count += 1 
          head_nod = False  
          if toggle_count == 3:  
            print("Yes")
            toggle_count = 0  
        
      # sets previous value before looping back
      previous_x4 = x4
      previous_y4 = y4

    else:
       annotated_image = image

    cv2.imshow('MediaPipe Pose', cv2.flip(annotated_image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()



  