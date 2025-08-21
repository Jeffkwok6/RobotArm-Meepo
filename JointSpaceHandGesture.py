import HandModule1 as htm
import cv2 as cv
import mediapipe as mp
import time 
import math
import numpy as np
##
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
##
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
##

detector = htm.handDetector()
pTime  = 0
cTime = 0
drawt = False
mode = ["singlehand","doublehand"]
modecheck = "no-hand"
modulus = 0
modecheckouter = ["joints","volume","filter"]
terminationconfig = [1,1,0,0,1]
onconfig = [1,0,1,1,1]
modecheckinner = "null"
jointangles = [0,0,0]
statecheck = ["off","on","null"]
cap = cv.VideoCapture(0,cv.CAP_DSHOW)
check = 0
onoff = 0
while True:
    sucess, img = cap.read()
    img = cv.resize(cv.flip(img,1),(1280,720))
    key = cv.waitKey(1)
    if key == ord('a'):
        check = check + 1
        print(check)
    elif key == ord('b'):
        check = check - 1
        print(check)              
    hands, img = detector.findHands(img,draw=drawt)
    if hands:
        hand1 = hands[0]
        checklist1 = hand1["lmList"]
        box1 = hand1["box"]
        center1 = hand1["center"]
        handType1 = hand1["type"]
        zedtype1 = hand1["zed"]
        fingers1 = detector.fingersUp(hand1,img,draw=drawt)
        modecheck = "singlehand"
        if len(hands) == 2:
            hand2 = hands[1]
            checklist2 = hand2["lmList"]
            box2 = hand2["box"]
            center2 = hand2["center"]
            handType2 = hand2["type"]
            zedtype2 = hand2["zed"]
            fingers2 = detector.fingersUp(hand2,img,draw=drawt)
            modecheck = "doublehand" 
            if fingers1 == terminationconfig and fingers2 == terminationconfig:
                length,info,ig,cx,cy = detector.distance(center1,center2,img)
                check = int(np.interp(length,[100,600],[1,2]))
                cv.putText(img,str(statecheck[check-1]),(cx-50,cy+50),cv.QT_FONT_NORMAL,5,(255,0,0),1)
            if fingers1 == onconfig and fingers2 == terminationconfig:
                length,info,ig,cx,cy = detector.distance(center1,center2,img)
                onoff = int(np.interp(length,[100,600],[1,2]))
                cv.putText(img,str(statecheck[onoff-1]),(cx-50,cy+50),cv.QT_FONT_NORMAL,5,(255,0,0),1)
            
        if statecheck[onoff-1] == "off" and statecheck[check-1] == "off":
            drawt = False
            if modecheck == mode[0]:
                cv.putText(img,str("Thanks for watching :)"),(center1[0],center1[1]),cv.QT_FONT_NORMAL,3,(00,255,0),3)

        if check == 2 and onoff == 2: 
            drawt = True
            if modecheck == "doublehand" and fingers1[1] == 1 and fingers2[1] == 1 and fingers1[0] == 1 and fingers2[0] ==1 and fingers1[2] == 1 and fingers2[2] ==1:
                length,info,ig,cx,cy = detector.distance(center1,center2,img)
                scale = int(np.interp(length,[150,800],[0,2]))
                modecheckinner = modecheckouter[scale]
                cv.putText(img,str(modecheckinner),(cx,cy),cv.QT_FONT_NORMAL,scale,(255,0,0),1)
                cv.putText(img,str(int(length)),(cx-120,cy-120),cv.QT_FONT_NORMAL,scale,(255,0,0),1)      
            if modecheckinner == modecheckouter[0]:
                cv.putText(img,str("on"),(10,40),cv.QT_FONT_NORMAL,2,(0,255,0),2)  
                cv.putText(img,str(modecheck),(20,120),cv.QT_FONT_NORMAL,1,(0,255,0),2)        
                cv.putText(img,str(modulus-1),(20,240),cv.QT_FONT_NORMAL,1,(0,255,0),2)       
                for i in range(len(jointangles)):
                    text = str("joint" + str(i)+ ": " + str(jointangles[i]))
                    cv.putText(img,text,(20,390+i*40),cv.QT_FONT_NORMAL,1,(0,255,255),2) 
                if modecheck == mode[0]:
                    if fingers1[4] == 0:
                        x1,y1 = zedtype1[8][1],zedtype1[8][2]
                        x2,y2 = zedtype1[4][1],zedtype1[4][2]
                        centerpose1,centerpose2 =[x1,y1],[x2,y2]
                        length,info,ig,cx,cy = detector.distance(centerpose1,centerpose2,img)
                        modulus = int(np.interp(length,[50,300],[0,4]))
                        cv.putText(img,str(modulus-1),(cx,cy),cv.QT_FONT_NORMAL,1,(0,255,0),2) 
                if modecheck == mode[1]:
                    if modulus > 0:
                        if fingers1[2]==0 and fingers1[3]==0 and fingers1[4] ==0 and fingers2[2]==0 and fingers2[3]==0 and fingers2[4] ==0:
                            length,info,ig,cx,cy = detector.distance(center1,center2,img)
                            scale = np.interp(length,[0,100],[1,2])
                            if modulus >=1 and modulus <= 3:
                                jointangles[modulus-1] = int(length)
                            cv.putText(img,str(int(length)),(cx,cy),cv.QT_FONT_NORMAL,scale,(255,0,0),1)    
            if modecheckinner == modecheckouter[1]:
                cv.putText(img,str(modecheckinner),(900,200),cv.QT_FONT_NORMAL,scale,(255,0,0),1)
                if fingers1[2]==0 and fingers1[3]==0 and fingers1[4] ==0 and fingers2[2]==0 and fingers2[3]==0 and fingers2[4] ==0:
                    length,info,ig,cx,cy = detector.distance(center1,center2,img)
                    scale = np.interp(length,[0,300],[0,2])
                    color = np.interp(length,[0,800],[0,255])
                    cv.line(img,(zedtype1[8][1],zedtype1[8][2]),(zedtype2[8][1],zedtype2[8][2]),(255,0,0),2)
                    cv.line(img,(zedtype1[4][1],zedtype1[4][2]),(zedtype2[4][1],zedtype2[4][2]),(255,0,0),2)
                    volmk = np.interp(length,[100,700],[minVol, maxVol-12])
                    volume.SetMasterVolumeLevel(int(volmk), None)
                    cv.putText(img,str(int(volmk)),(cx-120,cy),cv.QT_FONT_NORMAL,scale,(0,color,255-color),1)
    if check >= 4:
        print(check)
        check = 0
    cv.imshow("image",img)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows