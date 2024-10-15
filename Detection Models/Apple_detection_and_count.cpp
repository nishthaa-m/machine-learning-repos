!pip install ultralytics
from ultralytics import YOLO
model = YOLO('yolov8m.pt')           # Load the YOLOv8 model (fine-tuned or pre-trained on apples)
import cv2
from collections import OrderedDict
import numpy as np

# Path to the input video file
video_path = 'path to your video'
cap = cv2.VideoCapture(video_path)

# Get video details (height, width, FPS) for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer to save the result
out = cv2.VideoWriter('output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Class name for apples (this depends on the model's class names, assumed 'apple')
apple_class_name = 'apple'

# Centroid Tracker for tracking apples across frames
class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # Initialize unique ID counter and two dictionaries to store object info
        self.nextObjectID = 0
        self.objects = OrderedDict()   # ID to centroid mapping
        self.disappeared = OrderedDict()  # ID to count of frames disappeared

        self.maxDisappeared = maxDisappeared  # How long an object can disappear before being removed

    def register(self, centroid):
        # Add new apple with a unique ID and reset its disappearance count
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove object from tracking
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        # If no apples detected in this frame, increment disappeared counter for all tracked objects
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # If no objects currently being tracked, register all centroids
        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Calculate distances between new centroids and existing ones
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)

            # Match old objects to new detections based on closest centroids
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # Update existing object with new centroid
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # Register new objects and deregister lost objects
            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# Initialize tracker
tracker = CentroidTracker(maxDisappeared=50)

# Total apples tracked
total_apples = 0

# Loop over each frame in the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Stop if no frames are left

    # Run YOLOv8 object detection on the frame
    results = model(frame)

    # Get detected bounding boxes and filter out only apples
    detections = results[0].boxes

    # List to store centroids of detected apples in this frame
    input_centroids = []

    for box in detections:
        class_id = int(box.cls)  # Get class ID
        class_name = model.names[class_id]  # Get class name for the detected object

        # Only process the detection if it is an apple
        if class_name == apple_class_name:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Compute the centroid of the bounding box
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

    # Update the tracker with detected apple centroids
    objects = tracker.update(input_centroids)

    # Draw bounding boxes and labels on detected apples
    for objectID, centroid in objects.items():
        label = f"Apple {objectID + 1}"  # Unique apple label

        # Draw circle for the centroid and label the apple with increased font size and thickness
        cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 0), -1)
        cv2.putText(frame, label, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display total apple count in the bottom-right corner with increased font size and thickness
    total_apples = tracker.nextObjectID
    total_count_text = f"Total Apples: {total_apples}"
    text_size, _ = cv2.getTextSize(total_count_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    text_x = frame_width - text_size[0] - 10
    text_y = frame_height - 10
    cv2.putText(frame, total_count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Write the frame with detected apples and count to the output video
    out.write(frame)

# Release video capture and writer resources
cap.release()
out.release()
