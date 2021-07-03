import cv2
import numpy as np
from sort import *

# Multiple object tracker
persons_tracker = Sort()


# Counting area
#yellow_polygon_pts = np.array([[79, 228], [150, 138], [203, 41], [288, 233]])
yellow_polygon_pts = np.array([[242, 461], [225, 347], [727, 251], [768, 327]])  # plot for the entire room

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

cap = cv2.VideoCapture("Bar left - video1.mp4")

while True:
    # Load Image
    ret, img = cap.read()
    if ret is False:
        break

    # Cars
    persons = []

    # Detect Objects
    class_ids, scores, boxes = model.detect(img, nmsThreshold=0.4)
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
            cv2.putText(img, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 3)

            # Check if car is inside the Yellow polygon
            result = cv2.pointPolygonTest(yellow_polygon_pts, (x, y), False)

            # If car is inside the Polygon, then add the ID to the count
            if result > 0:
                persons_ids.add(id)






    cv2.polylines(img, [yellow_polygon_pts], True, (0, 500, 500), 3)

    # Show Cars count
    persons_count = len(persons_ids)
    cv2.putText(img, "persons: " + str(persons_count), (10, 30), 0, 1, (0, 0, 0), 2)





    # Show image
    cv2.imshow("Img", img)
    key = cv2.waitKey(20)
    if key == 27:
        break
cv2.destroyAllWindows()
