"""
__________part-1__________
mediapipe handtracking
讓視訊中的影像顯示手上的21個keypoint
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
"""
import cv2
import mediapipe as mp
class MediapipeHand:  
    def __init__(self):  
        self.mp_drawing = mp.solutions.drawing_utils #mediapipe 的畫圖function
        self.mp_hands = mp.solutions.hands #mediapipe的 偵測手勢模型
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        """
            keypointsx[0~20]依序為手掌的21 x座標 keypointsy[0~21]為對應的y座標 index對應到哪一點可以參考附檔的照片
            使用cv2.circle要記得keypoints 強制轉換為int opencv讀取整數值座標 
        """
        self.keypointsx = []
        self.keypointsy = []
        
        self.HaveHandNow = 0 #現在畫面中是否有手
    
    def HandKeyPoint(self,cap):        
        count = 1
        x = 0
        y = 0
        
        # For webcam input:
        
        success, image = cap.read() #讀取影像 success為flag image就是要處理的影像可視為一張照片
        frameWidth = image.shape[1]
        frameHeight = image.shape[0]
    
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)#將自拍圖像水平翻轉符合現實方向 調轉色彩通道
        image.flags.writeable = False #為了增加效能先限制檔案更改權限
        results = self.hands.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        handimg = image.copy()
        handimg.fill(0)
        '''if not success:
            print("Ignoring empty camera frame.")
            return image'''
    
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:#讀取多個手部
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(handimg, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
            if count :
                self.keypointsx.clear()
                self.keypointsy.clear()
                for data_point in hand_landmarks.landmark: #依序讀取手的21點
                    x = data_point.x * frameWidth #將被normalize過的座標x還原 --> 乘影像的寬度
                    y = data_point.y * frameHeight #將被normalize過的座標y還原 --> 乘影像的長度
                    self.keypointsx.append(x)
                    self.keypointsy.append(y)
            #cv2.circle(image, (int(self.keypointsx[0]),int(self.keypointsy[0])), 5, (255, 0,0 ), thickness=-1, lineType=cv2.FILLED)
            #cv2.circle(image, (int(keypointsx[1]),int(keypointsy[1])), 5, (255, 0,0 ), thickness=-1, lineType=cv2.FILLED)
            if(len(self.keypointsx)==21):
                self.HaveHandNow = 1
                cv2.circle(image, (int(self.keypointsx[1]),int(self.keypointsy[1])), 5, (255, 0,0 ), thickness=-1, lineType=cv2.FILLED)
            else:
                self.HaveHandNow = 0
    
        """
        keypointsx[0~20]依序為手掌的21 x座標 keypointsy[0~21]為對應的y座標 index對應到哪一點可以參考附檔的照片
        使用cv2.circle要記得keypoints 強制轉換為int opencv讀取整數值座標 
        若要存成單張照片可使用cv2.imwrite把這一振影像存下來
        """
        return image,handimg #image是包含手追蹤和視訊的影像,handimg是只有手