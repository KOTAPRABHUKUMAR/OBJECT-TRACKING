import cv2
import numpy as rect

img = cv2.imread("L4 with out person.jpg")
cap = cv2.VideoCapture('Bar-Video-18-Apr.mp4')


x, y, w, h = 200, 120, 0, 0
x1, y1, w1, h1 = 510, 290, 0, 0
x2, y2, w2, h2 = 530, 410, 0, 0
x3, y3, w3, h3 = 800, 900, 0, 0
x4, y4, w4, h4 = 260, 900, 0, 0

cv2.putText(img, "Left Bar >", (x + int(w / 10), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
cv2.putText(img, "Bar Service >", (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
            2)
cv2.putText(img, "Bar Sitting", (x2 + int(w2 / 10), y2 + int(h2 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
cv2.putText(img, "Right Table >", (x3 + int(w3 / 10), y3 + int(h3 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(img, "< Left Table", (x4 + int(w4 / 10), y4 + int(h4 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Specify the text location and rotation angle
# text_location = (40,90)
# angle = 90

# Draw the text using cv2.putText()
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'left bar', text_location, font, 1, 255, 2)

# Rotate the image using cv2.warpAffine()
# M = cv2.getRotationMatrix2D(text_location, angle, 1)
# out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow('img',out)

leftbar = rect.array([[350, 60], [610, 30], [250, 750], [0, 580], [0, 400]])
barservice = rect.array([[630, 140], [1385, 150], [1385, 250], [620, 340]])
barsitting = rect.array([[500, 370], [1385, 260], [1385, 550], [500, 800]])
righttable = rect.array([[930, 880], [1385, 600], [1385, 940], [880, 940]])
lefttable = rect.array([[0, 600], [250, 780], [250, 940], [0, 940]])

cv2.polylines(img, [leftbar], True, (0, 0, 255), 3)
cv2.polylines(img, [barsitting], True, (0, 0, 255), 3)
cv2.polylines(img, [barservice], True, (0, 0, 255), 3)
cv2.polylines(img, [righttable], True, (0, 0, 255), 3)
cv2.polylines(img, [lefttable], True, (0, 0, 255), 3)

# i = 0
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imwrite('kang' + str(i) + '.jpg', frame)
#     i += 1

cap.release()
cv2.destroyAllWindows()
cv2.imshow("out", img)
cv2.waitKey()
cv2.destroyAllWindows()
