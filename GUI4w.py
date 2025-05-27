import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import csv

# Default weight file path
DEFAULT_WEIGHT_FILE = r"enter path to your yolo model here"

# Global variables to store selected paths and mode
media_path = None
output_dir = None
weight_file = DEFAULT_WEIGHT_FILE
mode = "video"
frame_interval = 1  # Default to every frame in video mode
confidence_threshold = 0.5  # Default confidence threshold
report_data = []

def select_media():
    global media_path
    if mode == "video":
        media_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        media_label.config(text=f"Selected Video: {media_path}")
    elif mode == "image":
        media_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")])
        media_label.config(text=f"Selected Image: {media_path}")

def select_output_directory():
    global output_dir
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    output_dir_label.config(text=f"Output Directory: {output_dir}")

def select_weight_file():
    global weight_file
    weight_file = filedialog.askopenfilename(title="Select YOLO Weights File (Optional)", filetypes=[("YOLO Weights", "*.pt"), ("All files", "*.*")])
    if weight_file:
        weight_file_label.config(text=f"Selected Weights File: {weight_file}")
    else:
        weight_file = DEFAULT_WEIGHT_FILE
        weight_file_label.config(text="Using Default Weights File")

def set_mode(selected_mode):
    global mode
    mode = selected_mode
    media_label.config(text=f"Select {mode.capitalize()}")
    media_button.config(text=f"Browse {mode.capitalize()}")

    # Toggle frame interval and confidence slider visibility based on mode
    if mode == "video":
        interval_frame.pack(pady=10)  # Make frame interval visible in video mode
        confidence_frame.pack(pady=10)  # Always show confidence frame
    else:
        interval_frame.pack_forget()  # Hide frame interval in image mode
        confidence_frame.pack_forget()  # Confidence slider will also hide in image mode

def update_preview(frame):
    max_preview_width = 800  # Max width for the preview window
    max_preview_height = 600  # Max height for the preview window

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    frame_image.thumbnail((max_preview_width, max_preview_height), Image.LANCZOS)

    frame_tk = ImageTk.PhotoImage(image=frame_image)


def run_detection():
    global report_data, confidence_threshold
    if not media_path or not output_dir:
        messagebox.showwarning("Input Required", "Please select a media file and output directory.")
        return

    # Retrieve current confidence threshold from the slider
    confidence_threshold = confidence_slider.get()

    try:
        model = YOLO(weight_file)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load YOLO model: {e}")
        return

    # Run detection based on selected mode
    if mode == "video":
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamped_dir = os.path.join(output_dir, datetime.now().strftime("output_%Y%m%d_%H%M%S"))
        os.makedirs(timestamped_dir, exist_ok=True)

        frame_count, saved_count = 0, 0
        while cap.isOpened() and saved_count < 500:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = model.predict(source=frame)
                annotated_frame = results[0].plot()

                # Filter detections by confidence threshold
                high_conf_detections = [box for box in results[0].boxes if box.conf >= confidence_threshold]

                if high_conf_detections:
                    milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
                    seconds = int((milliseconds / 1000) % 60)
                    minutes = int((milliseconds / (1000 * 60)) % 60)
                    output_path = os.path.join(timestamped_dir, f"{minutes:02}_{seconds:02}_{frame_count:04}.jpg")
                    cv2.imwrite(output_path, annotated_frame)
                    update_preview(annotated_frame)
                    saved_count += 1

                    report_data.append({"Frame": frame_count, "Time": f"{minutes:02}:{seconds:02}", "Path": output_path})

            frame_count += 1

        cap.release()
        save_report(timestamped_dir)
        reset_gui()
        messagebox.showinfo("Process Complete", f"{saved_count} frames saved in: {timestamped_dir}")

    elif mode == "image":
        try:
            img = cv2.imread(media_path)
            results = model.predict(source=img)
            high_conf_detections = [box for box in results[0].boxes if box.conf >= confidence_threshold]

            if high_conf_detections:
                annotated_img = results[0].plot()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = os.path.join(output_dir, f"annotated_{timestamp}.jpg")
                cv2.imwrite(output_path, annotated_img)
                report_data.append({"Frame": "N/A", "Time": "N/A", "Path": output_path})
                update_preview(annotated_img)
                save_report(output_dir)
                reset_gui()
                messagebox.showinfo("Process Complete", f"Annotated image saved as {output_path}")
            else:
                reset_gui()
                messagebox.showinfo("No Objects Detected", "No objects were detected in the image with the specified confidence.")
        except Exception as e:
            reset_gui()
            messagebox.showerror("Error", f"Could not process image: {e}")

def save_report(directory):
    report_path = os.path.join(directory, "detection_report.csv")
    with open(report_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Frame", "Time", "Path"])
        writer.writeheader()
        writer.writerows(report_data)

def reset_gui():
    report_data.clear()
    submit_button.config(text="Submit")

# Initialize GUI
app = tk.Tk()
app.title("Drone and Bird Detection")
app.geometry("1920x1080")  # Set default screen size to 1920x1080

# Mode selection
mode_frame = tk.LabelFrame(app, text="Mode Selection", padx=10, pady=10)
mode_frame.pack(pady=10)
video_radio = tk.Radiobutton(mode_frame, text="Video", variable=mode, value="video", command=lambda: set_mode("video"))
video_radio.pack(side="left")
video_radio.select()
image_radio = tk.Radiobutton(mode_frame, text="Image", variable=mode, value="image", command=lambda: set_mode("image"))
image_radio.pack(side="left")

# Media selection
media_frame = tk.LabelFrame(app, text="Media Selection", padx=10, pady=10)
media_frame.pack(pady=10)
media_label = tk.Label(media_frame, text="Select Video or Image")
media_label.pack(pady=5)
media_button = tk.Button(media_frame, text="Browse Video", command=select_media)
media_button.pack(pady=5)

# Output directory selection
output_frame = tk.LabelFrame(app, text="Output Directory", padx=10, pady=10)
output_frame.pack(pady=10)
output_dir_label = tk.Label(output_frame, text="Select Output Directory")
output_dir_label.pack(pady=5)
output_dir_button = tk.Button(output_frame, text="Browse Output Directory", command=select_output_directory)
output_dir_button.pack(pady=5)

# Weight file selection
weight_frame = tk.LabelFrame(app, text="Weights File Selection", padx=10, pady=10)
weight_frame.pack(pady=10)
weight_file_label = tk.Label(weight_frame, text="Using Default Weights File")
weight_file_label.pack(pady=5)
weight_file_button = tk.Button(weight_frame, text="Browse Weights File", command=select_weight_file)
weight_file_button.pack(pady=5)

# Frame interval slider (only for video mode)
interval_frame = tk.LabelFrame(app, text="Frame Interval", padx=10, pady=10)
interval_slider = tk.Scale(interval_frame, from_=1, to=10, orient="horizontal", resolution=1)
interval_slider.set(frame_interval)
interval_slider.pack(pady=5)

# Confidence slider
confidence_frame = tk.LabelFrame(app, text="Confidence Threshold", padx=10, pady=10)
confidence_slider = tk.Scale(confidence_frame, from_=0.0, to=1.0, orient="horizontal", resolution=0.01)
confidence_slider.set(confidence_threshold)
confidence_slider.pack(pady=5)

# Submit button
submit_button = tk.Button(app, text="Submit", command=run_detection)
submit_button.pack(pady=20)

# Preview area


# Start the GUI loop
app.mainloop()
