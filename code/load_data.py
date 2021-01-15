'''
用來整理手勢訓練集資料
整理成feature (x) 和label (y)
'''
import numpy as np
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import matplotlib.pyplot as plt
import pickle
import cv2


#RGB轉灰階
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

x = []
y = []


num_dir=9 #資料夾0~8分別是不同手勢 要比資料夾總數多1
num_data=300 #一個資料夾有三百張相同手勢
for i in range(num_dir):
    for j in range(num_data):
        file = 'HandDataset/'+str(i)+'/'+str(j)+'.jpg'
        data = cv2.imread(file)
        data = rgb2gray(data)  
        data = data.reshape(28,28,1)
        x.append(data)
        l = []
        for k in range(num_dir):
            if k == i :
                l.append(1)
            else:
                l.append(0)
        l = np.array(l).astype('float32')
        y.append(l)
print('i:'+str(i))
print('j:'+str(j))
x = np.array(x).astype('float32')
y = np.array(y).astype('float32')

print(x.shape)
print(y.shape)

test = x[840]
ans = y[840]
print(test.shape) 
print(ans)
plt.imshow(test.reshape(28,28), cmap = 'Greys') # 顯示圖片
plt.axis('off') # 不顯示座標軸
plt.show()


features = x
labels = y

with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)


with open("features", "rb") as f:
    check_features = np.array(pickle.load(f))
with open("labels", "rb") as f:
    check_labels = np.array(pickle.load(f))

print(check_features.shape)
print(check_labels.shape)
