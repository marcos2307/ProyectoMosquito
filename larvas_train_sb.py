import numpy as np
from imutils import contours
from skimage import measure

from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths

import numpy as np
import argparse
import imutils
import cv2

import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import time
t0 = int(round(time.time() * 1000))

data = []
samples = []
nonzero = []
bin_n = 16

data = []
samples = []
nonzero = []
bin_n = 16
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi)) 
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) 
    return hist

def outsideArea(x,y,shape):
        sx = shape[1] - 1
        sy = shape[0] - 1
        x1 = [0,325,600,sx]
        x2 = [360,0,sx,550]
        y1 = [725,0,0,620]
        y2=[sy,300,250,sy]
        
        outside = False
        for i in range(4):
            v1 = [x2[i]-x1[i], y2[i]-y1[i]]    # Vector 1
            v2 = [x2[i]-x, y2[i]-y]  
            xp0 = v1[0]*v2[1] - v1[1]*v2[0]
            if i < 2:
                xp0 = xp0*-1
            if xp0 > 0:
                outside = True
        return outside

def preprocessing(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

image_width = 128
image_height = 128

def formatImage(img):
    img = img.resize((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((128, 128, 3))
    # Normalize
    x = (x - 128.0) / 128.0
    return x

train = True

#larvas_list_train = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44]
larvas_list_test = [3,6,9,12,15,18,21,24,27,30,33,36,39,42]

#larvas_list_train = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
larvas_list_train = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44]

label1 = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
label2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
label3 = [1, 0, 1, 1, 0, 0, 1]
label4 = [0, 0, 1, 1, 1, 0, 0, 0, 1]
label5 = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
label6 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
label7 = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
label8 = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
label9 = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
label10 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
label11 = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
label12 = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
label13 = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
label14 = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
label15 = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
label16 = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
label17 = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
label18 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
label19 = [0, 1, 0, 0, 0, 0, 0, 1]
label20 = [0, 1, 0, 1, 1, 0, 0, 0, 1]
label21 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
label22 = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
label23 = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
label24 = [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1]
label25 = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
label26 = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
label27 = [0, 0, 1, 1, 0, 0, 0, 0]
label28 = [1, 0, 0, 0, 0, 0]
label29 = [0, 0, 0, 0, 0, 0, 0, 0, 1]
label30 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
label31 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
label32 = [0, 0, 1, 1, 0, 0, 1]
label33 = [0, 0, 1, 1, 0, 0, 0, 0, 1]
label34 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
label35 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
label36 = [0, 0, 0, 0, 1, 1, 0, 1]
label37 = [0, 1, 1, 0, 0, 0, 0, 0, 1]
label38 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
label39 = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
label40 = [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
label41 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
label42 = [0, 0, 0, 0, 0, 0, 0, 1, 0]
label43 = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
label44 = [0, 0, 0, 0, 0, 0, 1, 0]
label45 = [1, 0, 0, 0, 1, 0, 1, 0, 0]

label_train = label1+label2+label4+label5+label7+label8+label10+label11+label13+label14+label16+label17+label19+label20+label22+label23+label25+label26+label28+label29+label31+label32+label34+label35+label37+label38+label40+label41+label43+label44
label_test = label3+label6+label9+label12+label15+label18+label21+label24+label27+label30+label33+label36+label39+label42

for i in range(0,len(label_train)):
    label_train[i] = abs(1-label_train[i])
for i in range(0,len(label_test)):
    label_test[i] = abs(1-label_test[i])

X_train = []
y_train = label_train

X_test = []
y_test = label_test


for larvas in larvas_list_train:
    input_str = "C:/Users/gomez/ProyectoMosquito/larvas/larvas"+str(larvas)+".jpg"
    original = cv2.imread(input_str)
    original = original[75:975,750:1650]
    roi = original.copy()
    
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    image = cv2.medianBlur(image,9)
    

    
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,5)
    thresh_str = "C:/media/lsd/raspifotos/snaps2/thresh_arducam"+str(larvas)+".png"
    cv2.imwrite(thresh_str,thresh)
    labels = measure.label(thresh, connectivity=1, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
     
    
    for lb in np.unique(labels):
        if lb == 0:
            continue
    
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == lb] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        if numPixels > 120 and numPixels < 745:
            mask = cv2.add(mask, labelMask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    total = 0
    radiuslist = []
    detectados_mat=[]
    
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        if not outsideArea(cX,cY,image.shape):
            detectado = roi[int(cY-radius):int(cY+radius),int(cX-radius):int(cX+radius)]
            im_pil = Image.fromarray(detectado)
            img_r = formatImage(im_pil)
            X_train.append(img_r)
            
               


            
            
for larvas in larvas_list_test:
    input_str = "C:/media/lsd/raspifotos/arducam"+str(larvas)+".jpg"
    original = cv2.imread(input_str)
    original = original[75:975,750:1650]
    roi = original.copy()
    
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    image = cv2.medianBlur(image,9)
    
    
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,5)
    thresh_str = "C:/media/lsd/raspifotos/results/thresh_arducam"+str(larvas)+".png"
    cv2.imwrite(thresh_str,thresh)
    labels = measure.label(thresh, neighbors=4, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    
    for lb in np.unique(labels):
        if lb == 0:
            continue
    
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == lb] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        if numPixels > 120 and numPixels < 745:
            mask = cv2.add(mask, labelMask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    total = 0
    radiuslist = []
    detectados_mat=[]
    
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        if not outsideArea(cX,cY,image.shape):
            detectado = roi[int(cY-radius):int(cY+radius),int(cX-radius):int(cX+radius)]
            im_pil = Image.fromarray(detectado)
            img_r = formatImage(im_pil)
            X_test.append(img_r)
    
            


X_train = np.array(X_train)
X_test = np.array(X_test)


X_train = X_train.reshape(X_train.shape[0], 128, 128, 3)
X_test = X_test.reshape(X_test.shape[0], 128, 128, 3)
input_shape = (128, 128, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_test', X_test.shape[0])


input_shape = (128,128,3)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization


import time
t1 = int(round(time.time() * 1000))


model = Sequential()
model.add(Conv2D(128, kernel_size = (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=X_train,y=y_train, batch_size=50, epochs=20, verbose=1)    


t2 = int(round(time.time() * 1000))
print(t2-t1)

t3 = int(round(time.time() * 1000))
model.evaluate(X_test,y_test)

y_pred = model.predict(X_test)

y_pred_list = []

for y in y_pred:
    if y[1] >= y[0]:
        y_pred_list.append(1)
    else:
        y_pred_list.append(0)
        
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_list)

tn, fp, fn, tp = cm.ravel()
acc = (tp+tn)/(tp+tn+fp+fn)
rec = tp/(tp+fn)
prec = tp/(tp+fp)
f1 = 2*(prec*rec)/(prec+rec)
print("Accuracy: " + str(acc))
print("Precision: " + str(prec))
print("Recall: " + str(rec))
print("F1: " + str(f1))

t4 = int(round(time.time() * 1000))
print(t4-t3)
print(t4-t0)
