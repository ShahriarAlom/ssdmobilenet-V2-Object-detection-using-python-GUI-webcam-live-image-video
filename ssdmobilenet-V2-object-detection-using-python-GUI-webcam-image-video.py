import tkinter as tk
from tkinter import filedialog
import cv2
import tensorflow as tf
import numpy as np
import time
from object_detection.utils import label_map_util

# Load label map and obtain class names and ids
category_index = label_map_util.create_category_index_from_labelmap("/home/bappy/Desktop/Jupyter_notebook/labelmap.pbtxt", use_display_name=True)

# Function to load the model
def load_model(model_path):
    print("Loading saved model...")
    detect_fn = tf.saved_model.load(model_path)
    print("Model Loaded!")
    return detect_fn

# Function to visualize detections on the image
def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1] * w), int(bbox[0] * h)
            xmax, ymax = int(bbox[3] * w), int(bbox[2] * h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {int(score * 100)}%", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Function to perform object detection on the selected image
def detect_objects_from_image(model, image_path):
    image = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(frame_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)

    scores = detections['detection_scores'][0].numpy()
    bboxes = detections['detection_boxes'][0].numpy()
    labels = detections['detection_classes'][0].numpy().astype(int)
    labels = [category_index[n]['name'] for n in labels]

    visualise_on_image(image, bboxes, labels, scores, 0.3)  # You can adjust the threshold

    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to perform object detection on video
def detect_objects_from_video(model, video_path):
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print('Unable to read video / Video ended')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = model(input_tensor)

        scores = detections['detection_scores'][0].numpy()
        bboxes = detections['detection_boxes'][0].numpy()
        labels = detections['detection_classes'][0].numpy().astype(int)
        labels = [category_index[n]['name'] for n in labels]

        visualise_on_image(frame, bboxes, labels, scores, 0.3)  # You can adjust the threshold

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to perform object detection on webcam
def detect_objects_from_webcam(model):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print('Unable to read video / Video ended')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = model(input_tensor)

        scores = detections['detection_scores'][0].numpy()
        bboxes = detections['detection_boxes'][0].numpy()
        labels = detections['detection_classes'][0].numpy().astype(int)
        labels = [category_index[n]['name'] for n in labels]

        visualise_on_image(frame, bboxes, labels, scores, 0.3)  # You can adjust the threshold

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to browse and select the environment file
def browse_environment():
    environment_path = filedialog.askopenfilename(filetypes=[("Python Environment File", "*.yaml")])
    environment_entry.delete(0, tk.END)
    environment_entry.insert(0, environment_path)

# Function to change detection parameters
def change_parameters():
    # Add code to change detection parameters here
    pass

# Function to browse and select the saved model directory
def browse_model():
    model_path = filedialog.askdirectory()
    model_entry.delete(0, tk.END)
    model_entry.insert(0, model_path)

# Function to browse and select the image file
def browse_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    image_entry.delete(0, tk.END)
    image_entry.insert(0, image_path)

# Function to browse and select the video file
def browse_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    video_entry.delete(0, tk.END)
    video_entry.insert(0, video_path)

# Function to perform detection based on the selected source
def perform_detection():
    model_path = model_entry.get()
    detection_source = var.get()

    if not model_path:
        print("Please select the model directory.")
        return

    model = load_model(model_path)

    if detection_source == "Image":
        image_path = image_entry.get()
        if not image_path:
            print("Please select the image file.")
            return
        detect_objects_from_image(model, image_path)

    elif detection_source == "Video":
        video_path = video_entry.get()
        if not video_path:
            print("Please select the video file.")
            return
        detect_objects_from_video(model, video_path)

    elif detection_source == "Webcam":
        detect_objects_from_webcam(model)

# Create the main window
root = tk.Tk()
root.title("Object Detection")

# Add widgets
environment_label = tk.Label(root, text="Select Python Environment File:")
environment_label.pack()
environment_entry = tk.Entry(root)
environment_entry.pack()
environment_button = tk.Button(root, text="Browse", command=browse_environment)
environment_button.pack()

model_label = tk.Label(root, text="Select Saved Model Directory:")
model_label.pack()
model_entry = tk.Entry(root)
model_entry.pack()
model_button = tk.Button(root, text="Browse", command=browse_model)
model_button.pack()

image_button = tk.Button(root, text="Detect From Image", command=perform_detection)
image_button.pack()

video_button = tk.Button(root, text="Detect From Video", command=perform_detection)
video_button.pack()

webcam_button = tk.Button(root, text="Detect From Webcam", command=perform_detection)
webcam_button.pack()

root.mainloop()

labelmap_label = tk.Label(root, text="Select Label Map File:")
labelmap_label.pack()
labelmap_entry = tk.Entry(root)
labelmap_entry.pack()
labelmap_button = tk.Button(root, text="Browse", command=browse_labelmap)
labelmap_button.pack()

# Function to browse and select the label map file
def browse_labelmap():
    labelmap_path = filedialog.askopenfilename(filetypes=[("Label Map Files", "*.pbtxt")])
    labelmap_entry.delete(0, tk.END)
    labelmap_entry.insert(0, labelmap_path)

# Add detection parameter inputs
threshold_label = tk.Label(root, text="Detection Threshold:")
threshold_label.pack()
threshold_entry = tk.Entry(root)
threshold_entry.insert(0, "0.5")  # Default threshold value
threshold_entry.pack()

# Function to perform detection based on the selected source and parameters
def perform_detection():
    model_path = model_entry.get()
    labelmap_path = labelmap_entry.get()
    detection_source = var.get()
    threshold = float(threshold_entry.get())

    if not model_path:
        print("Please select the model directory.")
        return
    if not labelmap_path:
        print("Please select the label map file.")
        return

    model = load_model(model_path)
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

    if detection_source == "Image":
        image_path = image_entry.get()
        if not image_path:
            print("Please select the image file.")
            return
        detect_objects_from_image(model, image_path, category_index, threshold)

    elif detection_source == "Video":
        video_path = video_entry.get()
        if not video_path:
            print("Please select the video file.")
            return
        detect_objects_from_video(model, video_path, category_index, threshold)

    elif detection_source == "Webcam":
        detect_objects_from_webcam(model, category_index, threshold)

# Add detection parameter inputs
threshold_label = tk.Label(root, text="Detection Threshold:")
threshold_label.pack()
threshold_entry = tk.Entry(root)
threshold_entry.insert(0, "0.5")  # Default threshold value
threshold_entry.pack()

root.mainloop()

labelmap_label = tk.Label(root, text="Select Label Map File:")
labelmap_label.pack()
labelmap_entry = tk.Entry(root)
labelmap_entry.pack()
labelmap_button = tk.Button(root, text="Browse", command=browse_labelmap)
labelmap_button.pack()

# Function to browse and select the label map file
def browse_labelmap():
    labelmap_path = filedialog.askopenfilename(filetypes=[("Label Map Files", "*.pbtxt")])
    labelmap_entry.delete(0, tk.END)
    labelmap_entry.insert(0, labelmap_path)

# Add detection parameter inputs
threshold_label = tk.Label(root, text="Detection Threshold:")
threshold_label.pack()
threshold_entry = tk.Entry(root)
threshold_entry.insert(0, "0.5")  # Default threshold value
threshold_entry.pack()

# Function to perform detection based on the selected source and parameters
def perform_detection():
    model_path = model_entry.get()
    labelmap_path = labelmap_entry.get()
    detection_source = var.get()
    threshold = float(threshold_entry.get())

    if not model_path:
        print("Please select the model directory.")
        return
    if not labelmap_path:
        print("Please select the label map file.")
        return

    model = load_model(model_path)
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

    if detection_source == "Image":
        image_path = image_entry.get()
        if not image_path:
            print("Please select the image file.")
            return
        detect_objects_from_image(model, image_path, category_index, threshold)

    elif detection_source == "Video":
        video_path = video_entry.get()
        if not video_path:
            print("Please select the video file.")
            return
        detect_objects_from_video(model, video_path, category_index, threshold)

    elif detection_source == "Webcam":
        detect_objects_from_webcam(model, category_index, threshold)

# Add detection parameter inputs
threshold_label = tk.Label(root, text="Detection Threshold:")
threshold_label.pack()
threshold_entry = tk.Entry(root)
threshold_entry.insert(0, "0.5")  # Default threshold value
threshold_entry.pack()

root.mainloop()
def detect_objects_from_image(model, image_path, category_index, threshold):
    image = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(frame_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)

    scores = detections['detection_scores'][0].numpy()
    bboxes = detections['detection_boxes'][0].numpy()
    labels = detections['detection_classes'][0].numpy().astype(int)
    labels = [category_index[n]['name'] for n in labels]

    visualise_on_image(image, bboxes, labels, scores, threshold)  
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_objects_from_video(model, video_path, category_index, threshold):
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print('Unable to read video / Video ended')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = model(input_tensor)

        scores = detections['detection_scores'][0].numpy()
        bboxes = detections['detection_boxes'][0].numpy()
        labels = detections['detection_classes'][0].numpy().astype(int)
        labels = [category_index[n]['name'] for n in labels]

        visualise_on_image(frame, bboxes, labels, scores, threshold)  
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def detect_objects_from_webcam(model, category_index, threshold):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print('Unable to read video / Video ended')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = model(input_tensor)

        scores = detections['detection_scores'][0].numpy()
        bboxes = detections['detection_boxes'][0].numpy()
        labels = detections['detection_classes'][0].numpy().astype(int)
        labels = [category_index[n]['name'] for n in labels]

        visualise_on_image(frame, bboxes, labels, scores, threshold)  
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()






