# Ocular-Disease-Recognition
#Human eyes does not only reveal emotions, they reveal the "diseases" as well.

## Importing libraries and modules
# Necessary utility modules and libraries
import os
import shutil
import pathlib
import random
import datetime
import cv2

# Plotting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import gaussian, convolve2d
import seaborn as sns

# Libraries for building the model

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout, Activation, GlobalAveragePooling2D, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.applications import DenseNet121, ResNet50, InceptionV3, Xception, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l2, l1
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix

## Data Loading and Pre-processing
classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
df_temp = pd.read_csv("/content/trainLabels.csv")
len(df_temp), df_temp
df_temp['level'].value_counts()
class_code = {0: "No_DR",
              1: "Mild",
              2: "Moderate",
              3: "Severe",
              4: "Proliferate_DR"}
df_temp.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
def mapping_temp(df, root=dir_path):
    class_code = {0: "No_DR",
                  1: "Mild",
                  2: "Moderate",
                  3: "Severe",
                  4: "Proliferate_DR"}
    df['label'] = list(map(class_code.get, df['diagnosis']))
    df['path'] = [i[1]['label']+'/'+i[1]['id_code']+'.jpeg' for i in df.iterrows()]
    return df

df_temp = mapping_temp(df_temp)
df_temp
# Dropping the diagnosis column because the model assigns different codes for prediction
df_temp.drop(['diagnosis'], axis=1, inplace=True)
df_temp
# wiener filter
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def isbright(image, dim=227, thresh=0.4):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh
    def image_preprocessing(img):
    # 1. Read the image
#     img = mpimg.imread(img_path)
    img = img.astype(np.uint8)

    # 2. Extract the green channel of the image
    b, g, r = cv2.split(img)

    # 3.1. Apply CLAHE to intensify the green channel extracted image
    clh = cv2.createCLAHE(clipLimit=4.0)
    g = clh.apply(g)

    # 3.2. Convert enhanced image to grayscale
    merged_bgr_green_fused = cv2.merge((b, g, r))
    img_bw = cv2.cvtColor(merged_bgr_green_fused, cv2.COLOR_BGR2GRAY)

    # 4. Remove the isolated pixels using morphological cleaning operation.
    kernel1 = np.ones((1, 1), np.uint8)
    morph_open = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel1)

    # 5. Extract blood vessels using mean-C thresholding.
    thresh = cv2.adaptiveThreshold(morph_open, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)

    # 6. Applying morph_open operation
    kernel2 = np.ones((2, 2), np.uint8)
    morph_open2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)

    # 6. Stacking the image into 3 channels
    stacked_img = np.stack((morph_open2,)*3, axis=-1)

    return stacked_img.astype("float64")
    p = "/content/eyepacs_preprocess/eyepacs_preprocess/20738_right.jpeg"
img = mpimg.imread(p)
pro = image_preprocessing(img)
filename = os.path.basename(p)
plt.imshow(pro.astype("uint8"), cmap="gray");
# cv2.imwrite(filename, pro)
# plt.imshow(img)
random_img_path = [dir_path+'/'+img for img in random.sample(os.listdir(dir_path), 50)]
random_img_path
plt.figure(figsize=(20, 15))
plt.suptitle("Image Dataset for CLAHE Processed Images", fontsize=20)

for i in range(1, 51):
    plt.subplot(5, 10, i)
    img = mpimg.imread(random_img_path[i-1])
    img_pro = image_preprocessing(img)
    plt.imshow(img_pro.astype("uint8"), cmap="gray", aspect="auto")
    plt.axis(False);
    for i in range(5):
    os.mkdir('./'+class_code[i])
    import os
import shutil
# for i in df_temp.iloc[:5, :].iterrows():
#     print(i[1][2])
res = [[i[1][1], i[1][2]] for i in df_temp.iterrows()]
for i in res:
    des = './'+i[0]+'/'
    src = dir_path+'/'+i[1].split('/')[1]
    shutil.copy(src, des)
    # The model assigns labels in ascending order
classes = sorted(classes)
classes
train_df_temp = {}
test_df_temp = {}
for i in range(5):
    df = df_temp[df_temp['label']==classes[i]]['id_code'].to_list()
    random.seed(42)
    x = random.sample(df, int(0.8*len(df)))
    for j in x:
        train_df_temp[j] = i
    for j in df:
        if j not in train_df_temp.keys():
            test_df_temp[j] = i
train_df_temp = pd.DataFrame(train_df_temp.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
test_df_temp = pd.DataFrame(test_df_temp.items(), columns=['id_code', 'diagnosis']).sample(frac=1, random_state=42)
train_df_temp
class_code = {0: "Mild",
              1: "Moderate",
              2: "No_DR",
              3: "Proliferate_DR",
              4: "Severe"}
train_df_temp['label'] = list(map(class_code.get, train_df_temp['diagnosis']))
train_df_temp['path'] = [i[1]['label']+'/'+i[1]['id_code']+'.jpeg' for i in train_df_temp.iterrows()]
test_df_temp['label'] = list(map(class_code.get, test_df_temp['diagnosis']))
test_df_temp['path'] = [i[1]['label']+'/'+i[1]['id_code']+'.jpeg' for i in test_df_temp.iterrows()]
# Initializing the input size
IMG_SHAPE = (224, 224)
N_SPLIT = 3
EPOCHS = 10
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   preprocessing_function = image_preprocessing)
validation_datagen = ImageDataGenerator(rescale = 1./255,
                                        preprocessing_function = image_preprocessing)

train_data = train_datagen.flow_from_dataframe(dataframe=train_df_temp,
                                               directory='./',
                                               x_col='path',
                                               y_col='label',
                                               class_mode="categorical",
                                               batch_size=32,
                                               seed=42,
                                               target_size=IMG_SHAPE)

valid_data = validation_datagen.flow_from_dataframe(dataframe=test_df_temp,
                                                   directory='./',
                                                   x_col='path',
                                                   y_col='label',
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   seed=42,
                                                   target_size=IMG_SHAPE)

# Initializing the early stopping callback
es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
def cm(y_true, y_pred):
    classes.sort()
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                     index = classes,
                     columns = classes)
    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

def metrics(y_true, y_pred):
#     print(classification_report(y_true, y_pred, target_names=classes))
    acc = accuracy_score(y_true, y_pred)
    res = []
    for l in [0,1,2,3,4]:
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                          np.array(y_pred)==l,
                                                          pos_label=True,
                                                          average=None)
        res.append([classes[l],recall[0],recall[1]])
    df_res = pd.DataFrame(res,columns = ['class','sensitivity','specificity'])
    return df_res, acc
    # Function to train model
def train_model(model_test, epochs=EPOCHS, lr=0.001):
    # Compile the model
    model_test.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      metrics=['accuracy'])

    history = model_test.fit(train_data,
                           validation_data=valid_data,
                           steps_per_epoch=int(0.2* int(train_data.n//train_data.batch_size)),
                           epochs=epochs,
                           validation_steps=int(valid_data.n//valid_data.batch_size),
                           callbacks=[es])
    return history.history

# Function to make predictions on the test data
def make_predictions(model_test):
    # Evaluate the model
    predictions = model_test.predict(valid_data, verbose=1)
    y_preds = np.argmax(predictions, axis=1)
    return y_preds
    # Function to plot the performance metrics
def plot_result(hist):
    plt.figure(figsize=(20, 10));
    plt.suptitle(f"Performance Metrics", fontsize=20)

    # Actual and validation losses
    plt.subplot(1, 2, 1);
    plt.plot(hist['loss'], label='train')
    plt.plot(hist['val_loss'], label='validation')
    plt.title('Train and val loss curve')
    plt.legend()

    # Actual and validation accuracy
    plt.subplot(1, 2, 2);
    plt.plot(hist['accuracy'], label='train')
    plt.plot(hist['val_accuracy'], label='validation')
    plt.title('Train and val accuracy curve')
    plt.legend()
    # Observing the images
view_random_images(root_dir='./')
# View random images in the dataset
def view_random_images(root_dir, classes=classes):
    class_paths = [root_dir + "/" + image_class for image_class in classes]
    # print(class_paths)
    images_path = []
    labels = []
    for i in range(len(class_paths)):
        random_images = random.sample(os.listdir(class_paths[i]), 10)
        random_images_path = [class_paths[i]+'/'+img for img in random_images]
        for j in random_images_path:
            images_path.append(j)
            labels.append(classes[i])
    images_path

    plt.figure(figsize=(17, 10))
    plt.suptitle("Image Dataset", fontsize=20)

    for i in range(1, 51):
        plt.subplot(5, 10, i)
        img = mpimg.imread(images_path[i-1])
        plt.imshow(img, aspect="auto")
        plt.title(labels[i-1])
        plt.axis(False);

        ## Modelling (base Models)
## We'll use the following ImageNet models for training the images and observe the variations of the accuracy of the predicitions as predicted by the models:
* AlexNet
* DenseNet121
        # View random images in the dataset
def view_random_images(root_dir, classes=classes):
    class_paths = [root_dir + "/" + image_class for image_class in classes]
    # print(class_paths)
    images_path = []
    labels = []
    for i in range(len(class_paths)):
        random_images = random.sample(os.listdir(class_paths[i]), 10)
        random_images_path = [class_paths[i]+'/'+img for img in random_images]
        for j in random_images_path:
            images_path.append(j)
            labels.append(classes[i])
    images_path

    plt.figure(figsize=(17, 10))
    plt.suptitle("Image Dataset", fontsize=20)

    for i in range(1, 51):
        plt.subplot(5, 10, i)
        img = mpimg.imread(images_path[i-1])
        plt.imshow(img, aspect="auto")
        plt.title(labels[i-1])
        plt.axis(False);
  
  ## 1. AlexNet
  # Basic CNN model for AlexNet
model_alexnet = tf.keras.Sequential([
    Conv2D(input_shape=IMG_SHAPE+(3,), filters=96,kernel_size=11,strides=4,activation='relu'),
    MaxPool2D(pool_size=3,strides=2),
    Conv2D(filters=256,kernel_size=5,strides=1,padding='valid',activation='relu'),
    MaxPool2D(pool_size=3,strides=2),
    Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'),
    Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu'),
    Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu'),
    MaxPool2D(pool_size=3,strides=2),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dropout(0.5),
    Flatten(),
Dense(len(classes), activation='softmax')
], name="model_AlexNet")

  # Summary of AlexNet model
model_alexnet.summary()
model_alexnet_history = train_model(model_alexnet)
model_alexnet.save('model_alexnet_wiener_clahe_g.h5')
model_alexnet_results = model_alexnet.evaluate(valid_data, batch_size=32)
y_preds_alexnet = make_predictions(model_alexnet)
y_true = valid_data.classes
cm(y_true, y_preds_alexnet)
# Performance metrics for AlexNet
plot_result(model_alexnet_history)

##DenseNet
# Basic architecture of DenseNet
model_densenet=DenseNet121(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
x=model_densenet.output
x= Flatten()(x)
x= Dense(1024, activation='relu')(x)
x= Dense(512, activation='relu')(x)
x= Dropout(0.5)(x)
output=Dense(len(classes),activation='softmax')(x) #FC-layer
model_denseNet=tf.keras.Model(inputs=model_densenet.input,outputs=output)

# Summary of the denseNet model
model_denseNet.summary()

# Freezing the base model
for layer in model_denseNet.layers[:-5]:
    layer.trainable=False

model_denseNet_history = train_model(model_denseNet)

model_denseNet.save('model_densenet_wiener_clahe_g.h5')

# Evaluation metrics for denseNet model
model_denseNet_result = model_denseNet.evaluate(valid_data, batch_size=32)
model_denseNet_result

y_preds_model_denseNet = make_predictions(model_denseNet)

# Metrics for denseNet
metrics(y_true, y_preds_model_denseNet)

# Confusion Matrix
cm(y_true, y_preds_model_denseNet)
