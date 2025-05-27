# Drone and Bird Detection/custom YOLO model GUI

# Functionalities

1. **Media Selection (Image/Video)**
   - Choose between video or image mode.
   - Adjust file selection based on mode.

2. **File and Directory Selection**
   - Browse and select a video or image file.
   - Choose output directory to save results.
   - Optionally select a custom YOLO weight file.

3. **Detection Parameters**
   - **Frame Interval Slider**: For video mode, controls how frequently frames are processed.
   - **Confidence Threshold Slider**: Sets detection confidence threshold (0.0 to 1.0).

4. **YOLOv8 Detection Logic**
   - Loads a YOLOv8 model.
   - Processes:
     - Video frames at specified intervals.
     - Images directly.
   - Saves annotated images/frames that meet confidence threshold.

5. **Output and Reporting**
   - Saves annotated outputs with timestamped filenames.
   - Creates `detection_report.csv` with frame number, time, and file path.

6. **Preview System (Partially Implemented)**
   - Converts annotated images using `PIL` for preview (GUI display not wired yet).

7. **User Feedback**
   - Status updates shown via message boxes for success, warnings, or errors.

8. **Reset Mechanism**
   - Clears previous detection report and resets the interface.

---

# Packages Used

| Module              | Purpose                                                      |
|---------------------|--------------------------------------------------------------|
| `cv2 (OpenCV)`      | Image/video processing and saving                            |
| `torch`             | Backend support for YOLO model execution                     |
| `ultralytics.YOLO`  | Loads and runs the YOLOv8 model                              |
| `os`                | Filesystem operations                                        |
| `datetime`          | Timestamping files and folders                               |
| `tkinter`           | GUI interface and file dialogs                               |
| `PIL.ImageTk`       | Image conversion for GUI preview                             |
| `csv`               | Write detection results into a CSV file                      |

---

# How to Run

# Requirements

Install dependencies with:

pip install opencv-python-headless ultralytics torch Pillow

> Use `opencv-python-headless` to avoid GUI conflicts.


# Steps

1. Save the script as `detector_gui.py`.
2. Run using:

python detector_gui.py

3. In the GUI:
   - Select Video or Image mode.
   - Browse media file.
   - Choose output directory.
   - Optionally select YOLO weights.
   - Set detection parameters.
   - Click **Submit**.
   - Annotated outputs and CSV report will be saved.

---

*Made for drone bird detection using YOLOv8*
