import cv2
import numpy as np
import vehicles
from datetime import datetime

cnt_left=0
cnt_right=0

cap=cv2.VideoCapture("pivideo.mp4")

totalFrames = 0
start = datetime.now()

#Get width and height of video
width=int(cap.get(3))
w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=40

#line_left
line_left=int(2*(w/5))
line_right=int(w/2)
line_left_color=(0,0,255)
line_right_color=(255,0,0)
pt1 =  [line_left, 0]
pt2 =  [line_left, h]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [line_right, 0]
pt4 =  [line_right, h]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Kernals
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)

cars = []
max_p_age = 5
pid = 1


while(cap.isOpened()):
    ret,frame=cap.read()
    for i in cars:
        i.age_one()
    fgmask=fgbg.apply(frame)

    if ret==True:
        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)

        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
        
        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)

        #Find Contours
        countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area=cv2.contourArea(cnt)
          
            if area>areaTH:
                #Tracking
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)

                new=True
                if cx in range(0,width):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:  
                            new = False
                            i.updateCoords(cx, cy)

                            if i.going_Left(w, line_left)==True:
                                cnt_left+=1  
                            elif i.going_Right(0, line_right)==True:
                                cnt_right+=1
                            break
                        
                        if i.getState()=='1': 
                            if i.getDir()=='left'and i.getX()>0:
                                i.setDone()
                           
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i

                    if new==True: #If nothing is detected,create new
                        p=vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+=1

                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        str_left='LEFT: '+str(cnt_left)
        str_right='RIGHT: '+str(cnt_right)
        
        frame=cv2.polylines(frame,[pts_L1],False,line_left_color,thickness=2) 
        frame=cv2.polylines(frame,[pts_L2],False,line_right_color,thickness=2)
        
        cv2.putText(frame, str_left, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str_right, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        
        totalFrames += 1

        if cv2.waitKey(1)&0xff==ord('q'):
            break

    else:
        break

end = datetime.now()
diff = end - start
print("FPS: {:.2f}".format(totalFrames / (end - start).total_seconds()))
cap.release()
cv2.destroyAllWindows()
