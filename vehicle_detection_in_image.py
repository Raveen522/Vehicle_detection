import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

# Read an image
image = cv2.imread("input_image.jpg")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

# Point current frame
center_points_cur_frame = []

# Inside the main loop after detecting objects
class_ids, scores, boxes = od.detect(image)

# Define the list of acceptable class labels
acceptable_classes = ["car", "van", "bus", "truck", "motorbike"]

# Filter objects based on class labels
for i in range(len(class_ids)):
    class_id = int(class_ids[i])
    class_name = od.classes[class_id]

    if class_name in acceptable_classes:
        (x, y, w, h) = boxes[i]
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Only at the beginning we compare previous and current frame
if count <= 2:
    for pt in center_points_cur_frame:
        for pt2 in center_points_prev_frame:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < 20:
                tracking_objects[track_id] = pt
                track_id += 1
else:

    tracking_objects_copy = tracking_objects.copy()
    center_points_cur_frame_copy = center_points_cur_frame.copy()

    for object_id, pt2 in tracking_objects_copy.items():
        object_exists = False
        for pt in center_points_cur_frame_copy:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            # Update IDs position
            if distance < 20:
                tracking_objects[object_id] = pt
                object_exists = True
                if pt in center_points_cur_frame:
                    center_points_cur_frame.remove(pt)
                continue

        # Remove IDs lost
        if not object_exists:
            tracking_objects.pop(object_id)

    # Add new IDs found
    for pt in center_points_cur_frame:
        tracking_objects[track_id] = pt
        track_id += 1

for object_id, pt in tracking_objects.items():
    cv2.circle(image, pt, 5, (0, 0, 255), -1)
    cv2.putText(image, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

print("Tracking objects")
print(tracking_objects)

print("CUR FRAME LEFT PTS")
print(center_points_cur_frame)

cv2.imshow("Frame", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
