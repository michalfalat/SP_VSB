from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import cv2
from image_helper import image_resize
from nn_result import NNResult
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

ctr = 0
class_names = ['steering', 'shifting', 'wrong']

model = None
modelName = "model_15_45.dat"


def save_train_frame(frame, classType, width=64):
    frame = image_resize(frame, width)
    global ctr
    path = "./program/train_nn/" + classType + "/" + classType + str(ctr) + ".png"
    print(path)
    cv2.imwrite(path, frame)
    ctr += 1


def load_images_from_folder(classType, size = 64):
    images = []
    labels = []
    folder = "./program/train_nn/" + classType + "/"
    print("Loading images from: ", folder)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size))
        if img is not None:
            images.append(np.array(img))
            labels.append(get_class_index(classType))
    print("Done. (" + str(len(images)) + " images)")
    return images, labels


def get_class_index(classType):
    if classType == "steering":
        return 0
    elif classType == "shifting":
        return 1
    elif classType == "wrong":
        return 2


def run_train():
    size = 32
    # class_names = ['steering', 'shifting']
    steering_images, steering_labels = load_images_from_folder("steering", size)
    shifting_images, shifting_labels = load_images_from_folder("shifting", size)
    wrong_images, wrong_labels = load_images_from_folder("wrong", size)


    train_images = steering_images + shifting_images + wrong_images
    train_labels = steering_labels + shifting_labels + wrong_labels

    c = list(zip(train_images, train_labels))
    random.shuffle(c)

    train_images, train_labels = zip(*c)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    train_images = train_images / 255.0

    train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.1)

    matplotlib.use('TKAgg', warn=False, force=True)

    # print(train_images.shape)  # (9584, 64, 64)
    # print(train_labels.shape)  # (9584,)

    # print(test_images.shape)  # (2396, 64, 64)
    # print(test_labels.shape)  # (2396,)

    train_images = train_images.reshape(train_images.shape[0], size, size, 1)
    test_images = test_images.reshape(test_images.shape[0], size, size, 1)

    # print("after reshape: ")
    # print(train_images.shape)  # (9584, 64, 64)
    # print(train_labels.shape)  # (9584,)

    # print(test_images.shape)  # (2396, 64, 64)
    # print(test_labels.shape)  # (2396,)

    #one-hot encode target column
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # print("after one-hot: ")
    # print(train_images.shape)  # (9584, 64, 64)
    # print(train_labels.shape)  # (9584,)

    # print(test_images.shape)  # (2396, 64, 64)
    # print(test_labels.shape)  # (2396,)

    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train_history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=12, batch_size=128)
    

    tf.compat.v1.keras.models.save_model(model, modelName)

    print_accurancy(model, test_images, test_labels)
    show_loss(train_history, modelName)
    show_accuracy(train_history, modelName)


def show_accuracy(data, modelName):
    plt.plot(data.history['acc'])
    plt.plot(data.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig('loss_' + modelName +  '.pdf')
    plt.clf()


def show_loss(data, modelName):
    plt.plot(data.history['loss'])
    plt.plot(data.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig('accuracy_' + modelName + '.pdf')
    plt.clf()


def print_accurancy(model, test_images, test_labels):
    y_pred = model.predict(test_images)
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(test_labels)):
        test.append(np.argmax(test_labels[i]))
    a = accuracy_score(pred, test)
    print('Final Accuracy is:', a*100 , '%')

def evaluate(frame, size, printInfo = False):
    # frame = image_resize(frame, size)
    frame = cv2.resize(frame, (size, size))
    frame = np.array([frame])
    global model
    if model is None:
        model = tf.compat.v1.keras.models.load_model(modelName)

    frame = frame.reshape(frame.shape[0], size, size, 1)
    prediction = model.predict(frame)[0]
    label = ""
    result = NNResult()
    for i in range(len(prediction)):
        label  += class_names[i] + ': ' + str(prediction[i] * 100) + " %.\t"
        result[class_names[i]] = prediction[i] * 100
    if(printInfo == True):
        result.print_info()
    return result




def run_test():
    test_images, test_labels = load_images_from_folder("test")
    test_images = np.array(test_images) / 255.0

    print("loading model...")
    model = tf.compat.v1.keras.models.load_model("model.data")

    print("Model loaded.")
    print("Testing data...")
    predictions = model.predict(test_images)
    print("Testing finished. Plotting...")

    matplotlib.use('TKAgg', warn=False, force=True)

    num_rows = 5
    num_cols = 6    
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def draw_nn_result(frame, text, pos, color):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return frame
