# vehicle_detection_in_image.py file
import cv2
import numpy as np
from vehicle_detector import VehicleDetector

# Loading vehicle detector
vd = VehicleDetector()

# Load the image footage
img_footage = cv2.imread("footages/images/input_image_1120x768.png")

# Define the region of interest (ROI) as a polygon
roi_points = np.array([[350, 300], [630, 300], [1000, 750], [50, 750]], np.int32)

roi_points = roi_points.reshape((-1, 1, 2))

# Create an empty mask of the same size as the image
mask = np.zeros_like(img_footage)

# Fill the ROI polygon with white color (255) on the mask
cv2.fillPoly(mask, [roi_points], (255, 255, 255))

# Apply the mask to the image to extract the ROI
roi_img = cv2.bitwise_and(img_footage, mask)

# Draw lines to outline the ROI
cv2.polylines(img_footage, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

# Get the coordinations of detected vehicles as an array within the ROI
vehicle_boxes = vd.detect_vehicles(roi_img)
vehicle_count = len(vehicle_boxes)

# Draw rectangles for detected vehicles within the ROI
for box in vehicle_boxes:
    x, y, w, h = box
    cv2.rectangle(img_footage, (x, y), (x + w, y + h), (57, 44, 226), 1)

# Put vehicle count that detected
cv2.putText(img_footage, "Vehicles: " + str(vehicle_count), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (31, 31, 31), 2)

# Show the output
cv2.imshow("Vehicles", img_footage)
cv2.waitKey(0)
