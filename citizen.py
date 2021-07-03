import cv2

from numpy import array
import numpy as np
from sort import *

# Multiple object tracker
persons_tracker = Sort()


# Counting area
yellow_polygon_pts = np.array([[9, 12], [1186, 13], [1192, 656], [5, 659]])

persons_ids = set()
""" 1. Load the network """
# Load Network
net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")

# Enable GPU CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

colors = np.random.uniform(0, 255, size=(len(classes), 3))


""" 2. We load and display the images and objects detected """

img = cv2.imread("citizens.jpg")
scale_percent=0.50
width = int(img.shape[1]* scale_percent)
height=int(img.shape[0]* scale_percent)
dimension=(width,height)

resized=cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)




 # Cars
persons = []

# Detect Objects
class_ids, scores, boxes = model.detect(resized, nmsThreshold=0.4)
for (class_id, score, box) in zip(class_ids, scores, boxes):
        x, y, w, h = box
        x2 = x + w
        y2 = y + h
        class_name = classes[class_id[0]]
        color = colors[class_id[0]]

        # cv2.putText(img, "{} {}".format(class_name, score), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        # Select only cars
        if class_name == "person":
            persons.append([x, y, x2, y2])


# Update cars tracking
if persons:
        persons_bbs_ids = persons_tracker.update(np.array(persons))
        for person in persons_bbs_ids:
            x, y, x2, y2, id = np.array(person, np.int32)
            cv2.putText(resized, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.rectangle(resized, (x, y), (x2, y2), (0, 255, 0), 3)

# Check if car is inside the Yellow polygon
result = cv2.pointPolygonTest(yellow_polygon_pts, (x, y), False)

            # If car is inside the Polygon, then add the ID to the count
if result > 0:
    persons_ids.add(id)






cv2.polylines(resized, [yellow_polygon_pts], True, (0, 255, 255), 3)

 # Show Cars count
persons_count = len(persons_ids)
cv2.putText(resized, "persons: " + str(persons_count), (10, 30), 0, 1, (0, 0, 0), 2)




# # Show image
# cv2.imshow("Img", img)
# key = cv2.waitKey(0)
# if key == 27:
#     cv2.destroyAllWindows()
print(resized.shape)
cv2.imshow('output', resized)
cv2.imwrite('resized_citizens.jpg', resized)

cv2.waitKey(0)
cv2.destroyAllWindows()