# Thug Life Filter
import cv2
import cv2.cv2
import numpy as np
import dlib
from math import hypot

# Capturing Video From Webcam
cap = cv2.VideoCapture(0)

# Thug Life Images xD
glasses_image = cv2.imread("thug_glasses_new.png")
cigerette_image = cv2.imread("thug_cigerette.png")

# Face Detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
          
        # Points / Coordinates
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        center_eye = (landmarks.part(27).x, landmarks.part(27).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        left_mouth = (landmarks.part(9).x, landmarks.part(9).y)
        center_mouth = (landmarks.part(10).x, landmarks.part(10).y)
        right_mouth = (landmarks.part(14).x, landmarks.part(14).y)
        
        eyes_width = int(hypot(left_eye[0] - right_eye[0],
                           left_eye[1] - right_eye[1]) * 1.5)
        eyes_height = int(eyes_width * 0.204)
        
        mouth_width = int(hypot(left_mouth[0] - right_mouth[0],
                           left_mouth[1] - right_mouth[1]))
        mouth_height = int(eyes_width * 0.808)

        
        
        # Eyes Corners
        top_left = (int(center_eye[0] - eyes_width / 2), int(center_eye[1] - eyes_height / 2))
        bottom_right = (int(center_eye[0] + eyes_width / 2), int(center_eye[1] + eyes_height / 2))
        
        # Mouth Corners
        mouth_top_left = (int(center_mouth[0] - mouth_width / 2), int(center_mouth[1] - mouth_height / 2))
        mouth_bottom_right = (int(center_mouth[0] + mouth_width / 2), int(center_mouth[1] + mouth_height / 2))
        
        # To Draw Rectange Around Eyes
        # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
        
        thug_eyes = cv2.resize(glasses_image, (eyes_width, eyes_height))
        thug_eyes_gray = cv2.cvtColor(thug_eyes, cv2.COLOR_BGR2GRAY)
        _, eyes_mask = cv2.threshold(thug_eyes_gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        eyes_area = frame[top_left[1]: top_left[1] + eyes_height,
                          top_left[0]: top_left[0] + eyes_width]
        
        eyes_area_no_eyes = cv2.bitwise_and(eyes_area, eyes_area, mask=eyes_mask)
        
        final_eyes = cv2.add(eyes_area_no_eyes, thug_eyes)
        
        frame[top_left[1]: top_left[1] + eyes_height,
              top_left[0]: top_left[0] + eyes_width] = final_eyes
        
        
        
        thug_mouth = cv2.resize(cigerette_image, (mouth_width, mouth_height))
        thug_mouth_gray = cv2.cvtColor(thug_mouth, cv2.COLOR_BGR2GRAY)
        _, mouth_mask = cv2.threshold(thug_mouth_gray, 25, 255, cv2.THRESH_BINARY_INV)
        
        mouth_area = frame[mouth_top_left[1]: mouth_top_left[1] + mouth_height,
                          mouth_top_left[0]: mouth_top_left[0] + mouth_width]
        
        mouth_area_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
        
        final_mouth = cv2.add(mouth_area_no_mouth, thug_mouth)
        
        frame[mouth_top_left[1]: mouth_top_left[1] + mouth_height,
              mouth_top_left[0]: mouth_top_left[0] + mouth_width] = final_mouth
        
        
        # To Show Area Surrounded By Eyes
        # cv2.imshow("Eyes Area", eyes_area)
    
    # Final Output In a Window
    cv2.namedWindow("Thug Camera", cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty("Thug Camera", cv2.WND_PROP_FULLSCREEN, cv2.cv2.WND_PROP_FULLSCREEN)
    cv2.imshow("Thug Camera", frame)
    
    key = cv2.waitKey(1)
    
    # If User's KeyPress == Esc , Terminate The Program
    if key == 27:
        break