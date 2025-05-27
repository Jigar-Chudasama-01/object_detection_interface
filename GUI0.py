import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Define default weight file path
DEFAULT_WEIGHT_FILE = r"C:\Users\jigar\runs\detect\train2\weights\best.pt"

# Global variables to store selected paths and mode
media_path = None
output_dir = None
weight_file = DEFAULT_WEIGHT_FILE  # Start with the default weight file
mode = "video"  # Default to video mode

def select_media():
    global media_path
    if mode == "video":
        media_path = filedialog.askopenfilename(title="Select Video File",
                                                filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        media_label.config(text=f"Selected Video: {media_path}")
    elif mode == "image":
        media_path = filedialog.askopenfilename(title="Select Image File",
                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")])
        media_label.config(text=f"Selected Image: {media_path}")

def select_output_directory():
    global output_dir
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    output_dir_label.config(text=f"Output Directory: {output_dir}")

def select_weight_file():
    global weight_file
    weight_file = filedialog.askopenfilename(title="Select YOLO Weights File (Optional)",
                                             filetypes=[("YOLO Weights", "*.pt"), ("All files", "*.*")])
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

def run_detection():
    if not media_path or not output_dir:
        messagebox.showwarning("Input Required", "Please select a media file and output directory.")
        return

    try:
        model = YOLO(weight_file)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load YOLO model: {e}")
        return

    if mode == "video":
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            return

        timestamped_dir = os.path.join(output_dir, datetime.now().strftime("output_%Y%m%d_%H%M%S"))
        os.makedirs(timestamped_dir, exist_ok=True)

        frame_count = 0
        saved_count = 0
        while cap.isOpened() and saved_count < 500:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame)
            annotated_frame = results[0].plot()

            # Check if any objects were detected by inspecting the result
            if len(results[0].boxes) > 0:  # If there are bounding boxes
                # Get the timestamp of the current frame in milliseconds
                milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
                seconds = int((milliseconds / 1000) % 60)
                minutes = int((milliseconds / (1000 * 60)) % 60)

                # Use minutes and seconds in the filename
                output_path = os.path.join(timestamped_dir, f"{minutes:02}_{seconds:02}_{frame_count:04}.jpg")
                cv2.imwrite(output_path, annotated_frame)
                saved_count += 1
                print(f"Saved frame {saved_count} at video time {minutes:02}:{seconds:02}")

            frame_count += 1

        cap.release()
        messagebox.showinfo("Process Complete", f"{saved_count} frames saved in: {timestamped_dir}")

    elif mode == "image":
        try:
            img = cv2.imread(media_path)
            results = model.predict(source=img)

            # Check if any objects were detected
            if len(results[0].boxes) > 0:  # If there are bounding boxes
                annotated_img = results[0].plot()

                # Save image with a timestamp format if object is detected
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = os.path.join(output_dir, f"annotated_{timestamp}.jpg")
                cv2.imwrite(output_path, annotated_img)
                messagebox.showinfo("Process Complete", f"Annotated image saved as {output_path}")
            else:
                messagebox.showinfo("No Objects Detected", "No objects were detected in the image.")

        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {e}")


# Initialize GUI
app = tk.Tk()
app.title("Drone and Bird detection from images and videos")
app.geometry("800x500")
app.config(bg="#f0f0f0")  # Set background color

# Custom font for headings and labels
heading_font = ("Arial", 16, "bold")
label_font = ("Arial", 12)

# Mode selection
mode_frame = tk.Frame(app, bg="#f0f0f0")
mode_frame.pack(pady=20)
mode_label = tk.Label(mode_frame, text="Select Mode:", font=heading_font, bg="#f0f0f0")
mode_label.pack(side="left", padx=10)

video_radio = tk.Radiobutton(mode_frame, text="Video", variable=mode, value="video", command=lambda: set_mode("video"),
                             font=label_font, bg="#f0f0f0", activebackground="#cfe2f3")
video_radio.pack(side="left", padx=10)
video_radio.select()

image_radio = tk.Radiobutton(mode_frame, text="Image", variable=mode, value="image", command=lambda: set_mode("image"),
                             font=label_font, bg="#f0f0f0", activebackground="#cfe2f3")
image_radio.pack(side="left", padx=10)

# Media selection button
media_label = tk.Label(app, text="Select Video or Image", font=label_font, bg="#f0f0f0")
media_label.pack(pady=5)
media_button = tk.Button(app, text="Browse Video", command=select_media, font=label_font, bg="#4CAF50", fg="white", relief="flat")
media_button.pack(pady=5)

# Output directory selection button
output_dir_label = tk.Label(app, text="Select Output Directory", font=label_font, bg="#f0f0f0")
output_dir_label.pack(pady=5)
output_dir_button = tk.Button(app, text="Browse Output Directory", command=select_output_directory, font=label_font, bg="#4CAF50", fg="white", relief="flat")
output_dir_button.pack(pady=5)

# Weight file selection button
weight_file_label = tk.Label(app, text="Using Default Weights File", font=label_font, bg="#f0f0f0")
weight_file_label.pack(pady=5)
weight_file_button = tk.Button(app, text="Browse Weights File", command=select_weight_file, font=label_font, bg="#4CAF50", fg="white", relief="flat")
weight_file_button.pack(pady=5)

# Run detection button
submit_button = tk.Button(app, text="Submit", command=run_detection, font=label_font, bg="#4CAF50", fg="white", relief="flat")
submit_button.pack(pady=20)

# Hover effect for buttons (change color when mouse is over)
def on_enter(event):
    event.widget.config(bg="#45a049")

def on_leave(event):
    event.widget.config(bg="#4CAF50")

buttons = [media_button, output_dir_button, weight_file_button, submit_button]
for button in buttons:
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

# Start the GUI loop
app.mainloop()
