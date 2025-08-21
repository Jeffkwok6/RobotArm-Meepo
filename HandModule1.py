import cv2 as cv 
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import numpy as np
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils    

class handDetector():
    def __init__(self, mode = False, maxHands = 2,modelComp=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComp = modelComp
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComp,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  
        self.tipIds = [4,8,12,16,20]
        self.fingers = []
        self.lmList= [] 
    def findHands(self,img,draw=True, flipType=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        dh,dw,dc = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                checklist = []
                xCord = []
                yCord = []
                zed = []
                for id, lm in enumerate(handLms.landmark):
                    gx,gy,gz = int(lm.x*dw),int(lm.y*dh), int(lm.z*dw)
                    zed.append([id,gx,gy,gz])
                    checklist.append([gx,gy,gz])
                    xCord.append(gx)
                    yCord.append(gy)
                xmin, xmax = min(xCord), max(xCord)
                ymin, ymax = min(yCord), max(yCord)
                ##
                boundx, boundy = xmax-xmin,ymax-ymin
                box = xmin,ymin,boundx,boundy
                cenx,ceny = box[0] + (box[2]//2), \
                            box[1] + (box[3]//2)
                myHand["lmList"] = checklist
                myHand["box"] = box
                myHand["center"] = (cenx,ceny)
                myHand["zed"] = zed
                #
                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)
                if draw:
                    #self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv.rectangle(img,(box[0]-20,box[1]-20),(box[0] + box[2] +20, box[1] + box[3]+20),(255, 0, 255), 2)
                    cv.putText(img,myHand["type"],(box[0]-30,box[1]-30),cv.QT_FONT_NORMAL,2,(255,0,255),2)
                    #x1,y1 = zed[8][1], zed[8][2]
                    #cv.circle(img,(x1,y1),20,(0,255,0))
        return allHands,img
    
    def distance(self,p1,p2,img=None,color=(255,0,255),scale = 1):
        x1,y1 = p1
        x2,y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv.circle(img, (x1, y1), scale, color, cv.FILLED)
            cv.circle(img, (x2, y2), scale, color, cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv.circle(img, (cx, cy), scale, color, cv.FILLED)
        return length, info, img,cx,cy
    
    def fingersUp(self, myHand,img,draw):
        fingers = []
        zed = myHand["zed"]
        myHandType = myHand["type"]
        checklist = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            if myHandType == "Right":
                if checklist[self.tipIds[0]][0] > checklist[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    cv.circle(img,(zed[4][1],zed[4][2]),15,(0,0,255),2)
            else:
                if checklist[self.tipIds[0]][0] < checklist[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    cv.circle(img,(zed[4][1],zed[4][2]),15,(0,0,255),2)
            for id in range(1, 5):
                if checklist[self.tipIds[id]][1] < checklist[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                    posx = zed[self.tipIds[id]][1]
                    posy = zed[self.tipIds[id]][2]
                    if draw:
                        cv.circle(img,(posx,posy),20,(0,255,0),2)
                else:
                    fingers.append(0)
        return fingers
    