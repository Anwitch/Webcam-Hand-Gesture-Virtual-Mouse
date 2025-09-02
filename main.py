import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
wScr, hScr = pyautogui.size() # Ukuran layar
smoothening = 9 # Faktor perataan (smoothing) gerakan kursor
pX, pY = 0, 0 # Posisi kursor sebelumnya
cX, cY = 0, 0 # Posisi kursor saat ini
is_dragging = False

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img,1)

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hands = hands[0]
        lmList = hands['lmList']
        
        WRIST_POINT = lmList[0]

        INDEX_FINGER_TIP = lmList[8]
        INDEX_6_POINT = lmList[6]

        MIDDLE_FINGER_TIP = lmList[12]
        MIDDLE_10_POINT = lmList[10]

        PINKY_TIP = lmList[20]

        THUMB_TIP = lmList[4]

        x_in6, y_in6 = INDEX_6_POINT[0], INDEX_6_POINT[1]
        x_p, y_p = PINKY_TIP[0], PINKY_TIP[1]

        length_left_click, info, img = detector.findDistance(INDEX_FINGER_TIP[:2], INDEX_6_POINT[:2], img)
        length_right_click, info, img = detector.findDistance(MIDDLE_FINGER_TIP[:2], MIDDLE_10_POINT[:2], img)
        length_scroll, info, img = detector.findDistance(PINKY_TIP[:2], WRIST_POINT[:2], img)
        length_pinch, info, img = detector.findDistance(INDEX_FINGER_TIP[:2], THUMB_TIP[:2], img)

        h, w, _ = img.shape

        frameR = 150

        frame_center_y = h // 2

        cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)

        x_map = np.interp(x_in6, (frameR, w - frameR), (0, wScr))
        y_map = np.interp(y_in6, (frameR, h - frameR), (0, hScr))

        cX = pX + (x_map - pX) / smoothening
        cY = pY + (y_map - pY) / smoothening

        if length_pinch < 13:
            if not is_dragging:
                pyautogui.mouseDown()
                is_dragging= True
            pyautogui.moveTo(cX, cY)
            pX, pY = cX, cY

        elif is_dragging and length_pinch > 13:
            pyautogui.mouseUp()
            is_dragging = False

        elif length_scroll > 100:
            if y_p < frame_center_y: 
                pyautogui.scroll(20)
            else: 
                pyautogui.scroll(-20)

        elif length_left_click < 20:
            pyautogui.click()
            time.sleep(0.5)
        
        elif length_right_click < 20:
            pyautogui.rightClick()
            time.sleep(0.5)

        else:
            pyautogui.moveTo(cX, cY)
            pX, pY = cX, cY
            

    cv2.imshow("Deteksi Tangan", img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows