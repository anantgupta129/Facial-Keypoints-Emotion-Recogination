# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="AnSIHhcU7GS2"
# # 1: IMPORT & EXPLORE DATASET FOR FACIAL EXPRESSION DETECTION

# %% id="OK9Bxx3Zr1yM"
# Import the necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers

# %% id="nC_n-ip146O0" outputId="9370bc92-da05-4c22-c56c-8a712e6d4f38" colab={"base_uri": "https://localhost:8080/", "height": 359}
# read the csv files for the facial expression data
facialexpression_df = pd.read_csv('/content/drive/My Drive/Proj/Untitled folder/icml_face_data.csv')
facialexpression_df.head(10)

# %% id="kHJdgvax5j3G" outputId="e2a23383-fdbf-4424-adcb-e1b73991f153" colab={"base_uri": "https://localhost:8080/", "height": 137}
facialexpression_df[' pixels'][0] # String format


# %% id="uRYtII7P5sh7"
def string2array(x):
  '''
   function to convert pixel values in string format to array format
  '''
  return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

def resize(x):
  '''
   Resize images from (48, 48) to (96, 96)
  '''
  img = x.reshape(48,48)
  return cv2.resize(x, dsize=(96,96), interpolation=cv2.INTER_CUBIC)


# %% id="2Picj0oF52G2" outputId="11bd8799-09cc-452c-9771-3e6264797b73" colab={"base_uri": "https://localhost:8080/", "height": 419}
facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: resize(string2array(x)))
facialexpression_df

# %% [markdown] id="Xo9lF2_O7RcH"
# # 2: VISUALIZE IMAGES AND PLOT LABELS

# %% id="zWtnTcvL7PEW" outputId="a199bdd5-bbf7-4b3e-d1f9-8705a1792c0d" colab={"base_uri": "https://localhost:8080/", "height": 873}
label_to_text = {0:'anger', 1:'disgust', 2:'sad', 3:'happiness', 4: 'surprise'}
emotions = [0, 1, 2, 3, 4]
count = 0

fig, axs = plt.subplots(5,5, figsize=(12,12))
for i in emotions:
  data = facialexpression_df[facialexpression_df['emotion'] == i]
  for img in data[' pixels']:
    img = img.reshape(96,96)
    axs[i][count].imshow(img, cmap='gray')
    axs[i][count].title.set_text(label_to_text[i])
    count +=1
    if count==5:
      break
  count = 0
fig.tight_layout()

# %% id="fi3myh978MQj" outputId="bb5cdfa2-c20c-4539-cc67-86a2cfeccf28" colab={"base_uri": "https://localhost:8080/"}
facialexpression_df.emotion.value_counts().index

# %% id="ZGV5d1cw-W1Y" outputId="fc740f19-975d-4849-afa2-352fbc001763" colab={"base_uri": "https://localhost:8080/"}
facialexpression_df.emotion.value_counts()

# %% id="jzeBfVEb-bVf" outputId="ba59dc22-1ab1-4519-d1d9-3000287877c5" colab={"base_uri": "https://localhost:8080/", "height": 374}
plt.figure(figsize=(6,6))
sns.barplot(x=[label_to_text[i] for i in facialexpression_df.emotion.value_counts().index],
            y=facialexpression_df.emotion.value_counts()
           );

# %% [markdown] id="9t_CN9OhEVEn"
# # 3: DATA PREPARATION AND IMAGE AUGMENTATION

# %% id="xz7LIrOn--8w" outputId="ac8d7e3e-24cc-417e-e7b2-2d167f4e1874" colab={"base_uri": "https://localhost:8080/"}
# split the dataframe in to features and labels
from keras.utils import to_categorical

X = facialexpression_df[' pixels']
y = to_categorical(facialexpression_df['emotion'])

X = np.stack(X, axis=0)
X = X.reshape(24568, 96, 96,1)
X.shape, y.shape

# %% id="DaJkGYaeGG6n" outputId="4962e235-500d-4f09-80a5-3aa7535d51b1" colab={"base_uri": "https://localhost:8080/"}
# spliting dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
print("Train Set", X_train.shape, y_train.shape)
print("Val Set", X_val.shape, y_val.shape)
print("Test Set",X_test.shape, y_test.shape)

# %% id="rTNaFcr-HdpF"
# image normalization

X_train = X_train/255
X_val   = X_val /255
X_test  = X_test/255

# %% id="G2L5lX59H2B-"
# data argumentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             shear_range=0.1,
                             horizontal_flip=True,
                             fill_mode="nearest"
                            )


# %% [markdown] id="b1C04OsnIyAD"
# # 4: BUILD AND TRAIN DEEP LEARNING MODEL FOR FACIAL EXPRESSION CLASSIFICATION

# %% id="aPHicEIfIhHo"
def res_block(X, filter, stage):

  # Convolutional_block
  X_copy = X

  f1 , f2, f3 = filter

  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


  # Short path
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 1
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 2
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X

# %% id="ZwQzVLdRI2_4" outputId="262fff73-864d-400c-e6bd-2105447c57b3" colab={"base_uri": "https://localhost:8080/"}
input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides= (2, 2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides= (2, 2))(X)

# 2 - stage
X = res_block(X, filter= [64, 64, 256], stage= 2)

# 3 - stage
X = res_block(X, filter= [128, 128, 512], stage= 3)

# 4 - stage
# X = res_block(X, filter= [256, 256, 1024], stage= 4)

# Average Pooling
X = AveragePooling2D((4, 4), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)

model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model.summary()

# %% id="pcQkdc8cI-TE" outputId="453b7224-d784-4969-ac7e-ed49e80b9de7" colab={"base_uri": "https://localhost:8080/", "height": 1000}
plot_model(model, show_shapes=True, to_file='/content/drive/My Drive/Proj/Untitled folder/expression model/facial-expression-model.png')

# %% id="m-pOxs6qJXMY"
# compile the network
model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# define callbacks functions
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
path = '/content/drive/My Drive/Proj/Untitled folder/expression model/FacialExpression_weights.hdf5'
checkpointer = ModelCheckpoint(filepath = path, verbose = 1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='min')

# %% id="gbnS9WBFKQZQ" outputId="2abf4329-c063-4730-ba39-9c2e218a44b8" colab={"base_uri": "https://localhost:8080/"}
h = model.fit(datagen.flow(X_train, y_train, batch_size=64),
              validation_data=(X_val, y_val), 
              steps_per_epoch=len(X_train) // 64,
              epochs= 50, 
              callbacks=[checkpointer, earlystopping, reduce_lr]
              )

# %% id="-Hc4D698QuFP"
# saving model architecure
model_json = model.to_json()
with open("/content/drive/My Drive/Proj/Untitled folder/expression model/FacialExpression-model.json","w") as json_file:
  json_file.write(model_json)

# %% [markdown] id="Zvi1tkBqRHd4"
# # 5: ASSESS THE PERFORMANCE OF TRAINED FACIAL EXPRESSION CLASSIFIER MODEL

# %% id="YyTchr3ePkrs" outputId="98237e0a-119a-4746-e8fd-7c257ed21ddb" colab={"base_uri": "https://localhost:8080/"}
h.history.keys()

# %% id="XbbqbPuBPo-T" outputId="5c2f9695-2484-4d2f-c35b-4b7d36827e3a" colab={"base_uri": "https://localhost:8080/", "height": 404}
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])

plt.subplot(1,2,2)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Acc vs Epoch')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()

# %% id="OBjUKdhxKi_i" outputId="972c064d-3a33-4248-9efc-51a83abcc5ef" colab={"base_uri": "https://localhost:8080/"}
_, acc = model.evaluate(X_test, y_test)
print("Accuracy on test set {:.2f} %".format(acc*100))

# %% id="qgeTkg19PgXv"
pred = np.argmax(model.predict(X_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)

# %% id="1tyN2qjgRrqX" outputId="80c30cf8-4d43-4bc7-e057-f022a489ca0b" colab={"base_uri": "https://localhost:8080/", "height": 881}
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, pred)
print(cm)
plt.figure(figsize = (10, 10));
sns.heatmap(cm, annot = True);

print(classification_report(y_true, pred))

# %% id="k7iB7xgLSMJ3" outputId="ed81560b-0868-4229-8c50-3bc1f488df7b" colab={"base_uri": "https://localhost:8080/", "height": 1000}
import random
l = 5
w = 6

fig, axs = plt.subplots(l, w, figsize=(20,18))
axs = axs.ravel()

for i in np.arange(0, l*w):
  k = random.randint(0, len(X_test))
  axs[i].imshow(X_test[k].reshape(96,96), cmap='gray')
  axs[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[pred[k]], label_to_text[y_true[k]]))
  axs[i].axis('off')

fig.tight_layout()  

# %% id="x3Oo8uwSUjf1"
fig.savefig('/content/drive/My Drive/Proj/Untitled folder/expression model/exp-result.png')

# %% id="zVTtdpt6WAix"
