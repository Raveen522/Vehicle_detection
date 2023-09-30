import cv2
from vehicle_detector import VehicleDetector

# loading vehicle detector
vd = VehicleDetector()

# load the image footage
# img_footage = cv2.imread("footages/input_image_1120x768.png")
# img_footage = cv2.imread("footages/input_image_02.png")
# img_footage = cv2.imread("footages/input_image_03.png")
img_footage = cv2.imread("footages/input_image_04.png")

# get the coordinations of detected vehicles as an array
vehicle_boxes = vd.detect_vehicles(img_footage) 
vehicle_count = len(vehicle_boxes)

for box in vehicle_boxes:
    x, y, w, h = box # assigning attributes of box

    cv2.rectangle(img_footage, (x, y), (x + w, y + h), (57, 44, 226), 1) # draw a rectangle
    cv2.putText(img_footage, "Vehicles: " + str(vehicle_count), (20, 50), 16, 1, (31, 31, 31), 2) # put vehicle count that detected  


cv2.imshow("Vehicles", img_footage) # show the output
cv2.waitKey(0)
