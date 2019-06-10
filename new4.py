# I have used OpenCV library to detect the upper body and then detect the color of it.In this code we have
# pre-trained upper boady classifier downloaded from github in XML file. The upperbody cascade classifier is trained
# using KNN algorithm.After that I have used the masking and morphology process to detect color(RGB).The reason to
# choose KNN because it works very good for classifying Image data without using Deep learning .I have made it to
# detect for RED, BLUE and GREEN only but can be made for any color.The problem I am facing is the accuracy ,
# it is not so accurate but can become more accurate when train on more images.


import cv2
import numpy as np

# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:

    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # definig the range of red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    # defining the Range of Blue color
    blue_lower = np.array([99, 115, 150], np.uint8)
    blue_upper = np.array([110, 255, 255], np.uint8)

    # defining the Range of yellow color
    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    # finding the range of red,blue and yellow color in the image
    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")

    red = cv2.dilate(red, kernal)
    res = cv2.bitwise_and(img, img, mask=red)

    blue = cv2.dilate(blue, kernal)
    res1 = cv2.bitwise_and(img, img, mask=blue)

    yellow = cv2.dilate(yellow, kernal)
    res2 = cv2.bitwise_and(img, img, mask=yellow)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body = upperbody_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    # To draw a rectangle in a face
    #	cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    #	roi_gray = gray[y:y+h, x:x+w]
    #	roi_color = img[y:y+h, x:x+w]

    # Detects eyes of different sizes in the input image
    #	eyes = eye_cascade.detectMultiScale(roi_gray)

    # To draw a rectangle in eyes
    #	for (ex,ey,ew,eh) in eyes:
    #		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    # Display an image in a window
    for (x, y, w, h) in body:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        (contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                a, b, c, d = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (a, b), (a + c, b + d), (0, 0, 255), 2)
                cv2.putText(img, "RED color", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        # Tracking the Blue Color
        (contours, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                a, b, c, d = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (a, b), (a + c, b + d), (255, 0, 0), 2)
                cv2.putText(img, "Blue color", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

        # Tracking the yellow Color
        (contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)
                cv2.putText(img, "yellow  color", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

        # cv2.imshow("Redcolour",red)
        # cv2.imshow("Color Tracking", img)

    cv2.imshow('img', img)
    cv2.imshow("Color Tracking", img)
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()


#Ayush Kumar