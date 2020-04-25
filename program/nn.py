from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf
from image_helper import image_resize
from nn_result import NNResult
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

frame_counter = 0
class_names = ['steering', 'shifting', 'wrong']

saved_model = None
# model_name = "model_15_45.dat"


def save_train_frame(frame, model_path,  class_type, width=64):
    frame = image_resize(frame, width)
    global frame_counter
    path = model_path + class_type + "/" + class_type + str(frame_counter) + ".png"
    print(path)
    cv2.imwrite(path, frame)
    frame_counter += 1


def load_images_from_folder(model_path, class_type, size=64):
    images = []
    labels = []
    folder = model_path + class_type + "/"
    print("Loading images from: ", folder)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size))
        if img is not None:
            images.append(np.array(img))
            labels.append(get_class_index(class_type))
    print("Loading finished. Loaded " + str(len(images)) + " images)")
    return images, labels


def get_class_index(class_type):
    try:
        return class_names.index(class_type)
    except ValueError:
        return 0

def run_train(args, size = 32):
    steering_images, steering_labels = load_images_from_folder(args.path, "steering", size)
    shifting_images, shifting_labels = load_images_from_folder(args.path, "shifting", size)
    wrong_images, wrong_labels = load_images_from_folder(args.path, "wrong", size)

    train_images = steering_images + shifting_images + wrong_images
    train_labels = steering_labels + shifting_labels + wrong_labels

    joined_images_with_labels = list(zip(train_images, train_labels))
    random.shuffle(joined_images_with_labels)

    train_images, train_labels = zip(*joined_images_with_labels)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    train_images = train_images / 255.0
    train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.1)

    matplotlib.use('TKAgg', warn=False, force=True)

    train_images = train_images.reshape(train_images.shape[0], size, size, 1)
    test_images = test_images.reshape(test_images.shape[0], size, size, 1)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # define layers for model
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=args.epochs, batch_size=128)

    # save model to filesystem
    tf.compat.v1.keras.models.save_model(model, args.modelName)

    print_accurancy(model, test_images, test_labels)

    # show and save results from training
    show_loss(args, train_history)
    show_accuracy(args, train_history)


def show_accuracy(args, data):
    plt.plot(data.history['acc'])
    plt.plot(data.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_' + args.modelName + '.pdf')
    plt.clf()


def show_loss(args, data):
    plt.plot(data.history['loss'])
    plt.plot(data.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_' + args.modelName + '.pdf')
    plt.clf()


def print_accurancy(model, test_images, test_labels):
    # converting predictions to label
    y_prediction = model.predict(test_images)
    prediction_list = list()
    for i in range(len(y_prediction)):
        prediction_list.append(np.argmax(y_prediction[i]))

    # converting one hot encoded test label to label
    test_list = list()
    for i in range(len(test_labels)):
        test_list.append(np.argmax(test_labels[i]))
    final_accuracy = accuracy_score(prediction_list, test_list)
    print('Final accuracy is:', final_accuracy * 100, '%')


def evaluate(args, frame, size, printInfo=False):
    frame = cv2.resize(frame, (size, size))
    frame = np.array([frame])
    global saved_model
    if saved_model is None:
        saved_model = tf.compat.v1.keras.models.load_model(args.modelName)

    frame = frame.reshape(frame.shape[0], size, size, 1)
    prediction = saved_model.predict(frame)[0]
    label = ""
    result = NNResult()
    for i in range(len(prediction)):
        label += class_names[i] + ': ' + str(prediction[i] * 100) + " %.\t"
        result[class_names[i]] = prediction[i] * 100
    if printInfo is True:
        result.print_info()
    return result


def run_test(args):
    test_images, test_labels = load_images_from_folder(args.path, "test")
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

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]), color=color)


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


def draw_nn_result(frame, text, label_position, color):
    cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return frame
