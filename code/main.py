import cv2
import numpy as np
from MediapipeHand import MediapipeHand 
from HandGestureDetection import HandGestureDetection
from QuickDraw import QuickDraw

cap = cv2.VideoCapture(0) #數值是0代表讀取電腦的攝影機鏡頭
mediapipeHand = MediapipeHand() 
gestureDetection=HandGestureDetection() 
quickDraw = QuickDraw()

drawing = False #紀錄變數畫畫開始變數
startdrawing = False
positions_x=[] #紀錄食指位置x[8]變數
positions_y=[] #紀錄食指位置y[8]變數
pred_class = None
cv2.namedWindow("paint") #畫布視窗
paintWindow = cv2.imread('black.jpg') #黑色畫布

HandListIMG = cv2.imread('handlist.png')
cv2.imshow('Hand',HandListIMG)

'''
可以呼叫mediapipeHand.keypointsx[0~20]取得手指各點的x位置
呼叫mediapipeHand.keypointsy[0~20]取得手指各點的y位置
要用的時候記得轉成整數 例如:int(mediapipeHand.keypointsx[20])

        8   12  16  20
        |   |   |   |
        7   11  15  19
    4   |   |   |   |
    |   6   10  14  18
    3   |   |   |   |
    |   5---9---13--17
    2    \         /
     \    \       /
      1    \     /
       \    \   /
        ------0-
'''
while cap.isOpened(): #每一楨影像都可以視為一張照片 一次while迴圈 就是處理一楨影像(一張照片)
    mediapipeHand.HaveHandNow = 0
    image,handimg= mediapipeHand.HandKeyPoint(cap) #讓視訊顯示handtracking keypoint(手指關節點)
    
    
    if mediapipeHand.HaveHandNow == 1 : #畫面中是否有手
        prob,result = gestureDetection.hand_detect(handimg)
        #取得結果之後顯示在視訊上
        cv2.putText(image,result, (int(mediapipeHand.keypointsx[5]-150),int(mediapipeHand.keypointsy[5]-30)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image,str(prob), (int(mediapipeHand.keypointsx[5]-150),int(mediapipeHand.keypointsy[5]-10)), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        #print(result)
        if(result=='1'):#食指->開始畫畫模式
            drawing=True
            startdrawing = True
        elif (result=='5' and startdrawing == True):#結束畫畫模式
            #crop
            xmin=int(min(positions_x))-10
            xmax=int(max(positions_x))+10
            ymin=int(min(positions_y))-10
            ymax=int(max(positions_y))+10
            print(xmin,xmax,ymin,ymax)
            paintWindow = paintWindow*255.0 / paintWindow.max()
            print(paintWindow.shape[0],paintWindow.shape[1])
            if(xmin<0):
                xmin=0
            if(ymin<0):
                ymin=0
            if(xmax>600):
                xmax=600
            if(ymax>600):
                ymax=600
            paintWindow = paintWindow[ymin:ymax, xmin:xmax]
            cv2.imwrite("newestpaint.png", paintWindow) #save
            pred_class = quickDraw.draw_detect(paintWindow)
            #clear&reload
            drawing=False
            startdrawing = False
            positions_x=[] 
            positions_y=[] 
            age,handimg= mediapipeHand.HandKeyPoint(cap)
            paintWindow =cv2.imread('black.jpg')
        else:
            drawing = False
        if(drawing==True):
            positions_x.append(mediapipeHand.keypointsx[8])
            positions_y.append(mediapipeHand.keypointsy[8])
    if(drawing==True):
        for i in range (0,len(positions_x)):
            cv2.rectangle(image,(int(positions_x[i]),int(positions_y[i])),(int(positions_x[i]+3),int(positions_y[i]+3)),(0, 0, 0),-1)
            cv2.rectangle(paintWindow,(int(positions_x[i]),int(positions_y[i])),(int(positions_x[i]+3),int(positions_y[i]+3)),(255, 255, 255),-1)
            #print(mediapipeHand.keypointsx[8],mediapipeHand.keypointsy[8])
        cv2.line(paintWindow, (int(positions_x[i]),int(positions_y[i])), (int(positions_x[i-1]),int(positions_y[i-1])), (255, 255, 255),5)
        cv2.line(image, (int(positions_x[i]),int(positions_y[i])), (int(positions_x[i-1]),int(positions_y[i-1])), (255, 0, 0),3)

    if pred_class is not None:
        cv2.putText(image, "You are drawing " + pred_class, (150, 25),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    #顯示視訊畫面
    cv2.imshow('MediaPipe Hands',image)
    cv2.imshow("paint", paintWindow)#顯示畫布
    
    '''
    #用來儲存訓練集要用的資料
    if cv2.waitKey(5) & 0xFF == ord('s'):
        gestureDetection.SaveHandIMG(handimg)'''

    #按q可以關閉視窗
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

mediapipeHand.hands.close()
cap.release()
cv2.destroyAllWindows()
