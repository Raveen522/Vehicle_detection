import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("footages/videos/input_video_01.mp4")



# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Create a binary mask for the region of interest (ROI)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_points = np.array([[550, 300], [880, 300], [1250, 700], [250, 700]])  # Define the polygon points
    cv2.fillPoly(mask, [roi_points], 255)


    # Draw lines for the ROI on the frame
    for i in range(len(roi_points)):
        start_point = tuple(roi_points[i])
        end_point = tuple(roi_points[(i + 1) % len(roi_points)])  # Connect last point to the first
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)


    # Point current frame
    center_points_cur_frame = []

    # Inside the main loop after detecting objects
    class_ids, scores, boxes = od.detect(frame)

    # Define the list of acceptable class labels
    acceptable_classes = ["car", "van", "bus", "truck", "motorbike"]

    # Define the ROI coordinates
    roi_left = 225
    roi_top = 150
    roi_right = 670
    roi_bottom = 375


    # Filter objects based on class labels
    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        class_name = od.classes[class_id]

        if class_name in acceptable_classes:
            (x, y, w, h) = boxes[i]
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            # Check if the object's center is within the ROI
            if mask[cy, cx] == 255:
                center_points_cur_frame.append((cx, cy))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = class_name
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x, y - 20), (x + text_size[0], y), (100, 255, 255), -1)
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


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
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 2, 1, (0, 0, 255), 1)

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)


    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

   

cap.release()
cv2.destroyAllWindows()
