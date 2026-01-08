# Hand Gesture Recognition App

This is a simple Python project that recognizes hand gestures in real time using a webcam.  
It uses MediaPipe to detect hand landmarks and OpenCV to show the live camera feed and matched gestures.

The app learns gestures from images that you place in a folder, then compares them with your hand pose in front of the camera.

---

## What this project does
- Detects a hand using your webcam
- Learns gestures from reference images
- Matches live hand poses with saved gestures
- Shows the detected gesture name on screen
- Displays the matched reference image next to the webcam feed
- Allows capturing gestures live if an image does not contain a detectable hand

---

## Project structure
```

HAND_GESTURE_APP/
│
├── gestures/                  # Images used as gesture references
├── hand_gesture_display.py    # Main Python script
├── run_app.bat                # Run script for Windows
├── requirements.txt           # Required Python packages
├── README.md
└── .gitignore

````

---

## Requirements
- Python 3.9 or higher (recommended)
- A working webcam

Python libraries used:
- opencv-python
- mediapipe
- numpy

---

## How to install and run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd HAND_GESTURE_APP
````

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python hand_gesture_display.py
```

Or simply run:

```bash
run_app.bat
```

Press **q** to quit the program.

---

## How to add your own gestures

1. Take clear images of hand gestures.
2. Put them inside the `gestures/` folder.
3. The image file name becomes the gesture label.

Example:

```
gestures/
├── thumbs_up.jpg
├── peace.png
└── fist.jpg
```

If the program cannot detect a hand in a reference image, it will ask you to show the gesture live and press **SPACE** to capture it.

---

## How gesture matching works (simple explanation)

* MediaPipe detects 21 hand landmarks.
* Landmarks are normalized so position and size don’t matter.
* Live hand landmarks are compared with saved gesture landmarks.
* The closest match under a threshold is selected as the detected gesture.

---

## Customization

You can adjust how strict the gesture matching is by changing the `threshold` value in the code:

```python
match_gesture(..., threshold=2.0)
```

Lower values = stricter matching
Higher values = more flexible matching

---

## Technologies used

* Python
* MediaPipe Hands
* OpenCV
* NumPy

---

## Notes

This project is meant for learning and experimentation with computer vision and hand tracking.
It can be extended with more gestures or integrated into other applications.

---

## Acknowledgements

MediaPipe by Google
OpenCV community

```

---

### ✅ Why this sounds human
- Simple language  
- No marketing buzzwords  
- Explains things like a real developer  
- Looks normal for GitHub / college projects  

If you want, I can:
- Make it **even more casual**
- Make it **more academic**
- Add **screenshots section**
- Adjust it for **resume / internship portfolio**

Just tell me what vibe you want 😄
```
