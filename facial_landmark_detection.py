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

# %% [markdown] id="dVTymJearEJn"
# # 1: IMPORT LIBRARIES AND DATASETS

# %% id="oXQvU_ijrFp8" executionInfo={"status": "ok", "timestamp": 1605501437674, "user_tz": -330, "elapsed": 1404, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="ef934cbc-5b2c-4c70-9a97-0ba5fcca4fdf" colab={"base_uri": "https://localhost:8080/"}
# Mount the drive
from google.colab import drive
drive.mount('/content/drive')

# %% id="vQFbssqfr_UL" executionInfo={"status": "ok", "timestamp": 1605501438066, "user_tz": -330, "elapsed": 1772, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
# Import the necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# %% id="klj2F-OQr_Wn" executionInfo={"status": "ok", "timestamp": 1605501440400, "user_tz": -330, "elapsed": 4102, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
# load facial key points data
training = pd.read_csv('/content/drive/My Drive/Proj/Untitled folder/data.csv')

# %% id="xTb3fLL8r_ZQ" executionInfo={"status": "ok", "timestamp": 1605501440406, "user_tz": -330, "elapsed": 4090, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="04ebd537-7d1f-4012-812e-3b89681a2966" colab={"base_uri": "https://localhost:8080/", "height": 1000}
training.head()

# %% id="9yFsN8SAtHum" executionInfo={"status": "ok", "timestamp": 1605501440408, "user_tz": -330, "elapsed": 4076, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="1de2fde6-344f-4238-ecae-0c7bc04babd5" colab={"base_uri": "https://localhost:8080/"}
# Obtain relavant information about the dataframe
training.info()

# %% id="AJhdlYRqtKVK" executionInfo={"status": "ok", "timestamp": 1605501440409, "user_tz": -330, "elapsed": 4067, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="1e61d387-7f81-4478-b8a1-c2f14d748a50" colab={"base_uri": "https://localhost:8080/"}
# Check if null values exist in the dataframe
training.isnull().sum().sum()

# %% id="qtGMT3ozr_g3" executionInfo={"status": "ok", "timestamp": 1605501440941, "user_tz": -330, "elapsed": 4589, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
# Since values for the image are given as space separated string, separate the values using ' ' as separator.
# Then convert this into numpy array using np.fromstring and convert the obtained 1D array into 2D array of shape (96, 96)
training['Image'] = training['Image'].apply(lambda x: np.fromstring(x, dtype = int, sep = ' ').reshape(96, 96))

# %% id="KE1RIIy8r_fN" executionInfo={"status": "ok", "timestamp": 1605501440945, "user_tz": -330, "elapsed": 4580, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="44d0a5fb-8dd5-4d9a-bb85-021da79c7214" colab={"base_uri": "https://localhost:8080/"}
# Obtain the Shape of the image
training['Image'][0].shape

# %% id="06r0LqnKflxU" executionInfo={"status": "ok", "timestamp": 1605501440947, "user_tz": -330, "elapsed": 4571, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="eb85ef8c-3e6d-4f81-ec62-823fc0027801" colab={"base_uri": "https://localhost:8080/", "height": 317}
training.describe()

# %% [markdown] id="bNkYttG_tOKf"
# # 2: PERFORM Data VISUALIZATION

# %% id="tPbtjUqHhYTX" executionInfo={"status": "ok", "timestamp": 1605501448818, "user_tz": -330, "elapsed": 12427, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="a63eda59-759d-405c-82f4-5979e7fbae84" colab={"base_uri": "https://localhost:8080/", "height": 913}
import random # for visualizing random images

fig = plt.figure(figsize=(20, 20))
for i in range(30):
    k = random.randint(1, len(training))
    ax = fig.add_subplot(5, 6, i + 1)    
    image = plt.imshow(training['Image'][k],cmap = 'gray')
    for j in range(1,31,2):
        plt.plot(training.loc[k][j-1], training.loc[k][j], 'rx')
    

# %% [markdown] id="wbqDwd1mteJ4"
# # 3: IMAGE DATA AUGMENTATION

# %% id="H3TLq1UbtazX" executionInfo={"status": "ok", "timestamp": 1605501448821, "user_tz": -330, "elapsed": 12424, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
# Create a new copy of the dataframe
import copy
training_copy = copy.copy(training)

# %% id="ypyn10X7tbrb" executionInfo={"status": "ok", "timestamp": 1605501448830, "user_tz": -330, "elapsed": 12424, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="03f6b591-0149-42ae-afbc-5d0fe0b882cb" colab={"base_uri": "https://localhost:8080/"}
# Obtain the columns in the dataframe
columns = training_copy.columns[:-1]
columns

# %% [markdown] id="wPVlUd9sdNIh"
# ## Horizontal Flip - flip the images along y axis

# %% id="hyFb3o1ztbyr" executionInfo={"status": "ok", "timestamp": 1605501448831, "user_tz": -330, "elapsed": 12422, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
training_copy['Image'] = training_copy['Image'].apply(lambda x: np.flip(x, axis = 1))

# since we are flipping horizontally, y coordinate values would be the same
# Only x coordiante values would change, all we have to do is to subtract our initial x-coordinate values from width of the image(96)
for i in range(len(columns)):
  if i%2 == 0:
    training_copy[columns[i]] = training_copy[columns[i]].apply(lambda x: 96. - float(x) )

# %% id="fIM1786rtb37" executionInfo={"status": "ok", "timestamp": 1605501448835, "user_tz": -330, "elapsed": 12415, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="6c75512e-87bd-419d-f764-eccd6e0523d9" colab={"base_uri": "https://localhost:8080/", "height": 216}
# Show the Original image
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(training['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(training.loc[0][j-1], training.loc[0][j], 'rx')

# Show the Horizontally flipped image
plt.subplot(1,2,2)
plt.title('Horizontally flipped image')
plt.imshow(training_copy['Image'][0],cmap='gray')
for j in range(1, 31, 2):
        plt.plot(training_copy.loc[0][j-1], training_copy.loc[0][j], 'rx')

# %% id="-g67tKjwtbua" executionInfo={"status": "ok", "timestamp": 1605501448838, "user_tz": -330, "elapsed": 12408, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="8a063e9d-5c2e-4324-a4e8-42803d8efea1" colab={"base_uri": "https://localhost:8080/"}
# Concatenate the original dataframe with the augmented dataframe
augmented_df = np.concatenate((training, training_copy))
augmented_df.shape

# %% [markdown] id="M3enV5bbeqYk"
# ## Randomingly increasing the brightness of the images

# %% id="COzWKpthtvRW" executionInfo={"status": "ok", "timestamp": 1605501448856, "user_tz": -330, "elapsed": 12416, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="429091bb-da41-478c-fcba-61718cc06c36" colab={"base_uri": "https://localhost:8080/"}
# We multiply pixel values by random values between 1.5 and 2 to increase the brightness of the image
# we clip the value between 0 and 255

import random

training_copy = copy.copy(training)
training_copy['Image'] = training_copy['Image'].apply(lambda x:np.clip(random.uniform(1.5, 2)* x, 0.0, 255.0))
augmented_df = np.concatenate((augmented_df, training_copy))
augmented_df.shape

# %% id="S_Pvd2qqtvkF" executionInfo={"status": "ok", "timestamp": 1605501450671, "user_tz": -330, "elapsed": 14220, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="ff753bd0-12be-4947-be36-373e25f07d1b" colab={"base_uri": "https://localhost:8080/", "height": 216}
# Show the Original image
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(training['Image'][2], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(training.loc[2][j-1], training.loc[2][j], 'rx')

# Show the Bightened image
plt.subplot(1,2,2)
plt.title('Horizontally flipped image')
plt.imshow(training_copy['Image'][2],cmap='gray')
for j in range(1, 31, 2):
        plt.plot(training_copy.loc[2][j-1], training_copy.loc[2][j], 'rx')

# %% [markdown] id="tz56c0e0t71Y"
# # 4: PERFORM DATA NORMALIZATION AND TRAINING TEST SPLIT

# %% id="sNdOOb0Ctvei" executionInfo={"status": "ok", "timestamp": 1605501450684, "user_tz": -330, "elapsed": 14222, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="304003a2-a449-4fa1-9fdc-4915f3ff3fc5" colab={"base_uri": "https://localhost:8080/"}
# Obtain the value of images which is present in the 31st column (since index start from 0, we refer to 31st column by 30)
img = augmented_df[:,30]

# Normalize the images
img = img/255.

# Create an empty array of shape (x, 96, 96, 1) to feed the model
X = np.empty((len(img), 96, 96, 1))

# Iterate through the img list and add image values to the empty array after expanding it's dimension from (96, 96) to (96, 96, 1)
for i in range(len(img)):
  X[i,] = np.expand_dims(img[i], axis = 2)

# Convert the array type to float32
X = np.asarray(X).astype(np.float32)
X.shape

# %% id="hPNN-RKftvcF" executionInfo={"status": "ok", "timestamp": 1605501450702, "user_tz": -330, "elapsed": 14232, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="4088308f-3b7d-4122-bbd9-c77a8a60d246" colab={"base_uri": "https://localhost:8080/"}
# Obtain the value of x & y coordinates which are to used as target.
y = augmented_df[:,:30]
y = np.asarray(y).astype(np.float32)
y.shape

# %% [markdown] id="gkGpZ7ple86r"
# ## Split the data into train and test data

# %% id="cl7FHUUMtvZr" executionInfo={"status": "ok", "timestamp": 1605501450709, "user_tz": -330, "elapsed": 14230, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="e9123d35-b008-44e9-de3f-419a7fa7526f" colab={"base_uri": "https://localhost:8080/"}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
print('Training size is {} and test size is {}'.format(len(X_train), len(X_test)))


# %% [markdown] id="yIXkiaHBuNrg"
# # 5: BUILD DEEP RESIDUAL NEURAL NETWORK KEY FACIAL POINTS DETECTION MODEL 

# %% id="7lCAPjVotvXi" executionInfo={"status": "ok", "timestamp": 1605501450710, "user_tz": -330, "elapsed": 14228, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
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

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1), name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
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

# %% id="NB9dPqlTtvVH" executionInfo={"status": "ok", "timestamp": 1605501459636, "user_tz": -330, "elapsed": 23144, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="8ee71cb3-aee2-4956-9c28-c13e4cf086ed" colab={"base_uri": "https://localhost:8080/"}
input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3,3))(X_input)

# 1 - stage
X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2 - stage
X = res_block(X, filter= [64,64,256], stage= 2)

# 3 - stage
X = res_block(X, filter= [128,128,512], stage= 3)

# 4 - stage
X = res_block(X, filter= [256,256,1024], stage= 4)

# Average Pooling
X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(4096, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation = 'relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation = 'relu')(X)


model = Model( inputs= X_input, outputs = X)

adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
model.compile(loss = "mean_squared_error", 
              optimizer = adam , 
              metrics = [tf.keras.metrics.RootMeanSquaredError()]
             )
model.summary()

# %% id="IBcVYI3RgwiI" executionInfo={"status": "ok", "timestamp": 1605501459648, "user_tz": -330, "elapsed": 23136, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="7af3d581-b795-49af-de15-0ce981b221b9" colab={"base_uri": "https://localhost:8080/", "height": 1000}
plot_model(model, to_file='/content/drive/My Drive/Proj/Untitled folder/Model/landmark.png', show_shapes=True)

# %% [markdown] id="IJCLhas0ulnq"
# # 6: TRAIN KEY FACIAL POINTS DETECTION DEEP LEARNING MODEL

# %% id="2pj-gfDCusny" executionInfo={"status": "ok", "timestamp": 1605501459652, "user_tz": -330, "elapsed": 23136, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
# save the best model with least validation loss
checkpointer = ModelCheckpoint(filepath="/content/drive/My Drive/Proj/Untitled folder/Model/FacialKeyPoints_weights.hdf5",
                               verbose=1, 
                               save_best_only=True
                              )
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=6, 
                              verbose=1, 
                              min_delta=0.0001
                             )
callbacks = [checkpointer, reduce_lr]

# %% id="IvWHFJyousme" executionInfo={"status": "ok", "timestamp": 1605502017941, "user_tz": -330, "elapsed": 581414, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="84a387f2-7d10-4fa0-d427-e635fadf4971" colab={"base_uri": "https://localhost:8080/"}
h = model.fit(X_train, y_train, 
              batch_size=32, 
              epochs=100, 
              validation_split=0.15, 
              callbacks=callbacks
             )

# %% id="2VLk1ZKPusjM" executionInfo={"status": "ok", "timestamp": 1605502098914, "user_tz": -330, "elapsed": 1395, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
# save the model architecture to json file for future use

model_json = model.to_json()
with open("drive/My Drive/Proj/Untitled folder/Model/FacialKeyPoints_model.json","w") as json_file:
  json_file.write(model_json)


# %% [markdown] id="c-cVW8K62Pai"
# # 7: MODEL PERFOMANCE & EVAUATION

# %% id="-XuTiBexu2j5" executionInfo={"status": "ok", "timestamp": 1605502126865, "user_tz": -330, "elapsed": 2455, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="fb4e51d3-a0aa-4ff9-9acc-07c36ef950b6" colab={"base_uri": "https://localhost:8080/"}
# Evaluate the model
result = model.evaluate(X_test, y_test)

# %% id="NA-LZmf-u2iE" executionInfo={"status": "ok", "timestamp": 1605502127861, "user_tz": -330, "elapsed": 1032, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="fd849d7e-2266-46ed-e9b5-3271fde72934" colab={"base_uri": "https://localhost:8080/"}
# Get the model keys 
h.history.keys()

# %% id="TJfpDL2xu2cV" executionInfo={"status": "ok", "timestamp": 1605502129995, "user_tz": -330, "elapsed": 1384, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="8e4d4afc-6947-4c7c-a682-7fd373f176b0" colab={"base_uri": "https://localhost:8080/", "height": 295}
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss (mean squared error)')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


# %% [markdown] id="7y6Hs0jRo7SZ"
# # 8: VISUALIZING RESUTS 

# %% id="sVS9wbUmnS7q" executionInfo={"status": "ok", "timestamp": 1605502133893, "user_tz": -330, "elapsed": 1805, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}}
pred = model.predict(X_test)

# %% id="5hRgEPhbnx9W" executionInfo={"status": "ok", "timestamp": 1605502140563, "user_tz": -330, "elapsed": 7387, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="009d065a-15f7-49c0-a52d-f7562b768dec" colab={"base_uri": "https://localhost:8080/", "height": 812}
fig = plt.figure(figsize=(20,14))

for i in range(30):
  k = random.randint(1, len(X_test))    # chosing a random prediction 
  fig.add_subplot(5,6,i+1)
  plt.imshow(X_test[k].reshape(96,96), cmap='gray')
  for j in range(1,31,2):
    plt.plot(pred[k][j-1], pred[k][j], 'rx')

# %% [markdown] id="v1NRpJjpYsOy"
# most of predictions seems correct but some are mistaken

# %% id="OXHtIy-MYmMz"
