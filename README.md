# ClickCanvas

## Overview
The Drawing App is an interactive application that allows users to draw on a canvas using their nose or hand movements. It utilizes computer vision techniques to detect facial landmarks and hand gestures, enabling a unique drawing experience. The application is built using Python and leverages several libraries for graphics and image processing, as well as tracking movements. 

## Features
- Draw using nose or hand movements.
- Color selection through mouth opening.
- Brush size adjustment by tilting the head.
- Save drawings as images.
- Instruction manual accessible in-app.

## Requirements
Before running the application, ensure you have the following packages installed:

- Python 3.7 or higher
- OpenCV (version 4.5.3)
- Mediapipe (version 0.8.6)
- Pygame (version 2.0.1)
- NumPy (version 1.21.2)
- Pillow (version 8.3.1)
- Pynput (version 1.7.3)

You can install the required packages using pip. Here’s a command to install all the necessary packages:
- pip install opencv-python
- pip install mediapipe
- pip install pygame
- pip install numpy
- pip install Pillow
- pip install pynput


## How to Run the Application
1. Clone the repository to your local machine:
   ```
   git clone https://github.com/JeffreyErnest/HCI_Project
   cd <repository-directory>
   ```

2. Ensure your webcam is connected and accessible.

3. Run the application:
   ```
   python drawingApp.py
   ```

4. Follow the on-screen instructions to start drawing.

## Instructions for Use
- Move your nose or hand to draw on the canvas.
- Hold the "Space" key to draw. Release to stop drawing.
- Press "Enter" to switch between nose and hand modes.
- Open your mouth to select a color from the color wheel. (open mouth again to close and confirm selection)
   - Tilt your head (right) to adjust brush size or toggle eraser mode (left).
- To save your drawing, hover over the 'Save' button and press "Space".
- To quit, hover over the 'Exit' button and press "Space".
- Press "TAB" to view the instruction manual.
- Undo (left) and re-do (right) tilt head while mouth is closed.

## File Structure
- `drawingApp.py`: The main application file containing all the logic for drawing, color selection, and user interaction.
- `face_landmark.task`: A required file for MediaPipe, which must be in the same folder as `drawingApp.py`.
- `Jefffont-Regular.ttf`: The font file used for the instruction/help menu, which must also be in the same folder.

## Acknowledgments
- Thanks to the developers of OpenCV, Mediapipe, and Pygame for their excellent libraries that made this project possible.
