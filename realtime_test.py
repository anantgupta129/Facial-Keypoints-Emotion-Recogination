import os
import cv2
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
import numpy as np

json_path = 'expression model/FacialExpression-model.json'
weights_path = 'expression model/FacialExpression_weights.hd5'

with open(json_path, 'r') as json_file:
    json_savedModel = json_file.read()
    
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(weights_path)

emotions = ['angry', 'disgust', 'sad', 'happy', 'surprise']

casacde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, test_img = cap.read()
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = casacde.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255,150,150), 2)
        roi_gray = gray_img[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]!=0):  #if image is present?
            img_pix = roi_gray.astype('float')/255      #convert to float and normalize 
            img_pix = image.img_to_array(img_pix)      #image to np array
            img_pix = np.expand_dims(img_pix, axis=0)    

            pred = model.predict(img_pix)[0]
            label = emotions[pred.argmax()]
            cv2.putText(test_img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,150,150), 3)

    cv2.imshow('Facial Emotion Detection', test_img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


