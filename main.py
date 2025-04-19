from handpoint import HandDetection
import time
import pyautogui
import cv2

lastGestureTime = time.time()
gestureCooldown = 0.5  # detik
handDetection = HandDetection(min_detection_confidence=0.5, min_tracking_confidence=0.5)

webcam =cv2.VideoCapture()
webcam.open(0, cv2.CAP_DSHOW)

# labdmark jari
fingersTipsIds = [4, 8, 12, 16, 20]

def fingersUp(landmarks):
    fingers = []

    if len(landmarks) < 21:
        return fingers  # Data tidak lengkap

    # Deteksi jempol tergantung label tangan
    if handDetection.hand_label == "Right":
        if landmarks[4][1] > landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Kalau tangan kiri, arah X dibalik
        if landmarks[4][1] < landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # jari pepek
    for tipId in [8, 12, 16, 20]:
        if landmarks[tipId][2] < landmarks[tipId - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    status , frame = webcam.read()
    frame = cv2.flip(frame, 1)
    landmarks = handDetection.findHandLandMarks(image=frame,draw=True)

    if len(landmarks) >= 21:
        fingers = fingersUp(landmarks)
        now = time.time()

        if now - lastGestureTime > gestureCooldown:
            # Geser ke kanan
            if fingers == [0,1,0,0,0]:
                print("▶️ Kanan")
                pyautogui.press("right")
                lastGestureTime = now

            # Geser ke kiri
            elif fingers == [0,1,1,0,0]:
                print("◀️ Kiri")
                pyautogui.press("left")
                lastGestureTime = now

            # Lompat
            elif fingers == [0,1,1,1,0]:
                print("⬆️ Lompat")
                pyautogui.press("up")
                lastGestureTime = now

            # Slide
            elif fingers == [1,1,0,0,0]:
                print("⬇️ Slide")
                pyautogui.press("down")
                lastGestureTime = now

    cv2.imshow("hand Lanmark",frame)
    if cv2.waitKey(1) == ord('a'):
        break
cv2.destroyAllWindows()
webcam.release()