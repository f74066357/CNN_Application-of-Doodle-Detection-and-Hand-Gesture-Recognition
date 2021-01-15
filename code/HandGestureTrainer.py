import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import cv2

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

#from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
#from tensorflow.python.keras.layers import Dense,Flatten, Conv2D
#from tensorflow.python.keras.layers import MaxPooling2D, Dropout

#from keras.utils import np_utils, print_summary
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import os
import os.path as path
import argparse
from keras import backend as K

import matplotlib.image as mpimg # mpimg 用於讀取圖片

########################################################################################################################
# Set ArgumentParser so you can set all variables from terminal

parser = argparse.ArgumentParser(description='Test Arguments')

# model params
parser.add_argument('--model_name', default='hand_cnn1')       # the name of the saved model

# dataset params
parser.add_argument('--export_images', default=False)           # instead of training a model, save MNIST to .png's
parser.add_argument('--export_number', default=10)              # number of MNIST images to export (if enabled)
parser.add_argument('--plot_images', default=False)             # instead of training a model, display MNIST images

# training params
parser.add_argument('--epochs', default=5)                      # number of epochs to train model for
parser.add_argument('--batch_size', default=128)                # batch size to use for training

args = parser.parse_args()

########################################################################################################################
# Global Vars

model_name = args.model_name
export_images = args.export_images
export_number = args.export_number
plot_images = args.plot_images
epochs = args.epochs
batch_size = args.batch_size

def keras_model(image_x, image_y):
    num_of_classes = 9 ##記得改
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "QuickDraw.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

def export_model(saver, model, input_node_names, output_node_name):
    if not path.exists('out'):
        os.mkdir('out')

    tf.train.write_graph(K.get_session().graph_def, 'out', model_name + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, False,
                              'out/' + model_name + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + model_name + '.bytes', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + model_name + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")




def main():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    print(features.shape)
    print(labels.shape)
    features, labels = shuffle(features, labels)

    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(train_y[100])
    plt.imshow(train_x[100].reshape(28,28), cmap = 'Greys')
    plt.show()

    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    
    model, callbacks_list = keras_model(28,28)
    print(model.summary())
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=20, batch_size=64)
    
    
    
    model.save('Hand_CNN.h5')

    #export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_3/Softmax")


    #test result
    predict = model.predict_classes(test_x)
    pick = np.random.randint(1,90,5)
    for i in range(5):
        print('ans:'+str(predict[pick[i]]))
        plt.imshow(test_x[pick[i]].reshape(28,28), cmap = 'Greys')
        plt.show()
        #cv2.imshow('and:'+str(predict[pick[i]]), test_x[pick[i]].reshape(28,28))
        #plt.imshow(test_x[pick[i]].reshape(28,28),cmap='Greys')
        #plt.title(predict[pick[i]])
        #plt.axis("off")


main()
