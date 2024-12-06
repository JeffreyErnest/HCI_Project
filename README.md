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

## Instructions for Use:
- Move your nose or hand to draw on the canvas.
  -    Please try drawing slowly as fast movements may not be caught.
- Hold "Space" to draw. Let go of "space" to stop drawing.
- Press "Enter" to switch between nose and hand mode.
- Open and close your mouth to select a color. (black is selected from the center of the wheel.)
  -    To confirm your selection, open and close your mouth again.
  -    Tilt your head right while selecting color to change brush size. Select size from the menu.
  -    Tilt your head left while selecting color to toggle eraser. To get of the erase, you have to close and open your mouth twice.
- Tilt your head left to undo or tilt right to redo while your mouth is closed.
  -    NOTE: You have to tilt your head pretty far; this ensures you don't accidentally undo/redo.
- To save, hover over the 'save' button in the bottom right and hit "space."
- To quit, hover over the 'quit' button in the bottom left and hit "space."
- To re-open the instructions, hover over the 'How To' button and hit "space."

Press TAB to start drawing! (NOTE: that the program might take a second to boot)

## File Structure
- `drawingApp.py`: The main application file containing all the logic for drawing, color selection, and user interaction.
- `face_landmark.task`: A required file for MediaPipe, which must be in the same folder as `drawingApp.py`.

## Acknowledgments
- Thanks to the developers of OpenCV, Mediapipe, and Pygame for their excellent libraries that made this project possible.
