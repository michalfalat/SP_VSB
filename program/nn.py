from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import cv2
from image_helper import image_resize
import os
import matplotlib
import matplotlib.pyplot as plt
import random
ctr = 35
class_names = ['steering', 'shifting']

model = None


def save_train_frame(frame, classType, width=64):
    frame = image_resize(frame, width)
    global ctr
    path = "./program/train_nn/" + classType + "/" + classType + str(ctr) + ".png"
    print(path)
    cv2.imwrite(path, frame)
    ctr += 1


def load_images_from_folder(classType):
    images = []
    labels = []
    folder = "./program/train_nn/" + classType + "/"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        if img is not None:
            images.append(np.array(img))
            labels.append(get_class_index(classType))
    return images, labels


def get_class_index(classType):
    if classType == "steering":
        return 0
    elif classType == "shifting":
        return 1
    else:
        return 1


def run_train():
    # class_names = ['steering', 'shifting']
    steering_images, steering_labels = load_images_from_folder("steering")
    shifting_images, shifting_labels = load_images_from_folder("shifting")


    train_images = steering_images + shifting_images
    train_labels = steering_labels + shifting_labels

    c = list(zip(train_images, train_labels))
    random.shuffle(c)

    train_images, train_labels = zip(*c)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    train_images = train_images / 255.0
    # train_labels = train_labels / 256.0

    matplotlib.use('TKAgg', warn=False, force=True)

    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=20)

    tf.compat.v1.keras.models.save_model(model, "model.data")

def evaluate(frame, width = 64):
    frame = image_resize(frame, width)
    frame = cv2.resize(frame, (64, 64))
    frame = np.array([frame])
    global model
    if model is None:
        model = tf.compat.v1.keras.models.load_model("model.data")
    prediction = model.predict(frame)[0]
    label = ""
    for i in range(len(prediction)):
        label  += class_names[i] + ': ' + str(prediction[i] * 100) + " %.\t"
    print(label)




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
