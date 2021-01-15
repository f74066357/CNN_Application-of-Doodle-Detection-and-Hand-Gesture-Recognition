import cv2
import numpy as np
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from collections import deque
import os
"""
__________part-2__________
hand gesture detection
收集手勢資料並訓練
在視訊的每一frame套用模型並顯示結果
"""
class HandGestureDetection:  
    def __init__(self):  
        self.count=0
        self.model = tf.keras.models.load_model('Hand_CNN.h5')
    
    #在每一frame產生圖片並套用model回傳答案
    def hand_detect(self,Handimg):
        pred_probab, pred_class = self.keras_predict(self.model, self.ProduceHandIMG(Handimg))
        pred_probab = int(pred_probab * 1000) / 1000
        result = str(pred_class)
        if(pred_class==4):
            result = '4: OK'
        elif pred_class == 6:
            result = '6: love <3'
        elif pred_class == 8:
            result = '8: good :D'
        return pred_probab, result
        
    def keras_process_image(self,img):
        image_x = 28
        image_y = 28
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, image_x, image_y, 1))
        return img       
    
    #套用model
    def keras_predict(self,model, image):
        processed = self.keras_process_image(image)
        #print("processed: " + str(processed.shape))
        pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class    

    #RGB轉灰階
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    #處理手部圖片 變成28x28的黑白圖
    def ProduceHandIMG(self,Handimg):
        haveRec = 0 #有無在圖片中找到手
        gray = cv2.cvtColor(Handimg, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0) #use Gaussian Blur to remove the noise
        binaryIMG = cv2.Canny(blurred, 20, 160) #edge detection functions to get better results
        
        #第一次找輪廓,為了加粗線條
        '''如果有error改成下面這行
        contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)'''
        _,contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binaryIMG, contours, -1, (255,255,255), 5)
        #cv2.imwrite('1.jpg', binaryIMG)
        
        #第二次找輪廓,為了找手部範圍
        '''如果有error改成下面這行
        contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)'''
        _,contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            #找到包围盒的坐标系
            x,y,w,h=cv2.boundingRect(c)
            haveRec = 1
            #cv2.rectangle(Handimg,(x,y),(x+w,y+h),(255,255,255),3)#绿色
        if haveRec == 1:#畫面中有找到手
            centerx=int(x+w/2)
            centery=int(y+h/2)
            if w>h:
                crop_img = binaryIMG[centery-int(w/2)-10:centery-int(w/2)+w+10, centerx-int(w/2)-10:centerx-int(w/2)+w+10]
            else:   
                crop_img = binaryIMG[centery-int(h/2)-10:centery-int(h/2)+h+10, centerx-int(h/2)-10:centerx-int(h/2)+h+10]
            #cv2.imwrite('2.jpg', crop_img)
            try:    
                new_img = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA) #縮小
            except:
                new_img = binaryIMG.copy()
                new_img.fill(0)
                new_img = cv2.resize(new_img, (28, 28), interpolation=cv2.INTER_AREA) #縮小
        else:
            new_img = binaryIMG.copy()
            new_img.fill(0)
            new_img = cv2.resize(new_img, (28, 28), interpolation=cv2.INTER_AREA) #縮小
        return new_img
        
    #產生訓練集圖片並儲存
    def SaveHandIMG(self,Handimg):
        crop_img = self.ProduceHandIMG(Handimg)
        
        print('save image :'+str(self.count))
        new_img = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA) #縮小
        cv2.imwrite(str(self.count)+'.jpg', new_img)
        self.count = self.count + 1