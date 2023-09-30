# vehicle_detection_in_image.py file
import cv2
from vehicle_detector import VehicleDetector

# Loading vehicle detector
vd = VehicleDetector()

# Load the image footage
img_footage = cv2.imread("footages/images/input_image_1120x768.png")

# Define the region of interest (ROI) coordinates
roi_left = 350
roi_top = 300
roi_right = 800 # measure from left 
roi_bottom = 700

# Crop the image to the ROI
roi_img = img_footage.copy()  # Create a copy of the original image
# roi_img = roi_img[roi_top:roi_bottom, roi_left:roi_right]
roi_img = roi_img[roi_top:roi_bottom, roi_left:roi_right]

# Draw lines to highlight the ROI
cv2.rectangle(img_footage, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

# Get the coordinations of detected vehicles as an array within the ROI
vehicle_boxes = vd.detect_vehicles(roi_img)
vehicle_count = len(vehicle_boxes)

# Draw rectangles for detected vehicles within the ROI
for box in vehicle_boxes:
    x, y, w, h = box
    # Adjust the coordinates to the original image
    x += roi_left
    y += roi_top
    cv2.rectangle(img_footage, (x, y), (x + w, y + h), (57, 44, 226), 1)

# Put vehicle count that detected
cv2.putText(img_footage, "Vehicles: " + str(vehicle_count), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (31, 31, 31), 2)

# Show the output
cv2.imshow("Vehicles", img_footage)
cv2.waitKey(0)
