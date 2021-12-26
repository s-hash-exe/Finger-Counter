import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Images folder relative path
path = "fingers"

myList = os.listdir(path)
pTime=0

# List of images
overlayList = []

# Setting value of Detection Confidence
detector = htm.handDetector(detectionCon=0.75)

# Tips of each fingers
tipIds = [4,8,12,16,20]

for imPath in myList:
    image = cv2.imread(f'{path}/{imPath}')
    overlayList.append(image)

while True:
    ret, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        fingers=[]
        # Handling the left and right hand differences
        flag = lmList[tipIds[1]][1] > lmList[tipIds[2]][1]
        # Detecting the thumb separately
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            if flag:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if flag:
                fingers.append(0)
            else:
                fingers.append(1)
        # Looping for all fingers exceot the thumb
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Count of number of raised fingers
        totalFingers = fingers.count(1)

        # Overlaying the finger image based on the value of finger count
        overlayImg = cv2.resize(overlayList[totalFingers-1],(150,150))

        h,w,c = overlayImg.shape
        img[0:h,0:w] = overlayImg

        # Drawing a rectangle to display count
        cv2.rectangle(img, (20,225),(170,425),(0,255,0),-1)
        cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0),25)
    cTime = time.time()
    # FPS
    fps = int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"FPS: "+str(fps),(480,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow("Window", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()