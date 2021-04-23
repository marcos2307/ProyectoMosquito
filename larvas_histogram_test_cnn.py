import time
import numpy as np
import cv2
t3 = int(round(time.time() * 1000))
train = False
lista_conteo = []
results = np.array
results = []
larvas_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]

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

X_test = []
for larvas in larvas_list:
    input_str = "C:/Users/gomez/ProyectoMosquito/larvas/larvas"+str(larvas)+".jpg"
    original = cv2.imread(input_str)
    original = original[375:2750, 700:2850]
    roi = original.copy()
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image,9)
    cv2.line(original,(0,1800),(1350,original.shape[0]-1),(255,0,0),5)
    cv2.line(original,(800,0),(0,550),(255,0,0),5)
    cv2.line(original,(1750,0),(original.shape[1]-1,500),(255,0,0),5)
    cv2.line(original,(original.shape[1]-1,1450),(1300,original.shape[0]-1),(255,0,0),5)
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,5)
    labels = measure.label(thresh, neighbors=4, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
     
    for label in np.unique(labels):
        if label == 0:
            continue
    
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        if numPixels > 250 and numPixels < 1850:
            mask = cv2.add(mask, labelMask)
      
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    
    def outsideArea(x,y,shape):
        sx = shape[1] - 1
        sy = shape[0] - 1
        x1 = [0,800,1750,sx]
        x2 = [1350,0,sx,1300]
        y1 = [1800,0,0,1450]
        y2=[sy,550,500,sy]    
        outside = False
        for i in range(4):
            v1 = [x2[i]-x1[i], y2[i]-y1[i]]
            v2 = [x2[i]-x, y2[i]-y]  
            xp0 = v1[0]*v2[1] - v1[1]*v2[0]
            if i < 2:
                xp0 = xp0*-1
            if xp0 > 0:
                outside = True
        return outside
    
    total = 0
    total_larvas = 0
    total_no = 0
    radiuslist = []
    detectados_mat=[]
    samples_test=[]
    X_test = []

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        if not outsideArea(cX,cY,image.shape):
                detectado = roi[int(cY-radius):int(cY+radius),int(cX-radius):int(cX+radius)]
                im_pil = Image.fromarray(detectado)
                img_r = formatImage(im_pil)
                X_test.append(img_r)
                detectados_mat.append(detectado)   
                
                

                cv2.circle(original, (int(cX), int(cY)), int(radius),
                    (0, 255, 0), 3)
                cv2.putText(original, "#{}".format(total + 1), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                total = total + 1
                radiuslist.append(radius)
        else:
                cv2.circle(original, (int(cX), int(cY)), int(radius),
                    (0, 0, 255), 3)
                
                
    output_str = "C:/media/lsd/raspifotos/results/result"+str(larvas)+".png"
    cv2.imwrite(output_str,original)
    thresh_str = "C:/media/lsd/raspifotos/results/thresh"+str(larvas)+".png"
    cv2.imwrite(thresh_str,thresh)

    X_test=np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], 128, 128, 3)
    input_shape = (128, 128, 3)
    # Making sure that the values are float so that we can get decimal points after division
    X_test = X_test.astype('float32')
    

    

    if train == False:

        
        y_pred = model.predict(X_test)

        y_pred_list = []
        
        for y in y_pred:
            if y[1] >= y[0]:
                y_pred_list.append(1)
            else:
                y_pred_list.append(0)
        
        y_pred = y_pred_list
        
        results = np.concatenate((results, y_pred))
        roi2 = roi.copy()
        total = 0
        for (i, c) in enumerate(cnts):

            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            if not outsideArea(cX,cY,image.shape):
                   
                    aaaa = y_pred[total]
                    color = (255, 0, 0)
                    if aaaa==0:
                        total_larvas = total_larvas + 1
                        color = (0, 255, 0)
                    if aaaa==1:
                        color = (0, 0, 255)
                        total_no = total_no + 1
                    cv2.circle(roi2, (int(cX), int(cY)), int(radius),
                        color, 3)
                    cv2.putText(roi2, "#{}".format(aaaa), (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    total = total + 1
                    radiuslist.append(radius)
            else:
                    cv2.circle(original, (int(cX), int(cY)), int(radius),
                        (0, 0, 255), 3)
        clasif_str = "C:/media/lsd/raspifotos/results/cnn_"+str(larvas)+".png"           
        cv2.imwrite(clasif_str,roi2)
        lista_conteo.append([larvas,total_larvas,total_no,total])

a = [row[1] for row in lista_conteo]
b = [row[2] for row in lista_conteo]

print("larvas avg: " + str(np.mean(a)))
print("otros avg: " + str(np.mean(b)))


label1 = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
label2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
label3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
label4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
label5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
label6 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
label7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
label8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
label9 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
label10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
label11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
label12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
label13 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label14 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label15 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label16 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
label17 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label18 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label19 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label20 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
label21 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label22 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
label23 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label24 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label25 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label26 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label27 = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label28 = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label29 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label30 = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label31 = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label32 = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label33 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label34 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label35 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label36 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label37 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label38 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label39 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label40 = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label41 = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label42 = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label43 = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label44 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label45 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
label = label3+label6+label9+label12+label15+label18+label21+label24+label27+label30+label33+label36+label39+label42+label45

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label, results)
tn, fp, fn, tp = confusion_matrix(label, results).ravel()
acc = (tp+tn)/(tp+tn+fp+fn)
rec = tp/(tp+fn)
prec = tp/(tp+fp)
f1 = 2*(prec*rec)/(prec+rec)
