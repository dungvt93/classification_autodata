# coding:utf-8

import os,cv2
import numpy as np
#import matplotlib.pyplot as plt
import datetime

from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential,Model, load_model
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator
#from numba import cuda
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img

class CustomModel:
    def __init__(self):
        self.local_model = None
        self.nb_classes = 2
        self.classes_name = ['NG','OK']
        self.batch_size = 25
        self.steps_per_epoch = 8
        self.val_batch_size = 8
        self.val_steps_per_epoch = 10 
        self.epochs = 50
        self.img_row = 256
        self.img_col = 256

    def load(self,trained_model):
        self.local_model = load_model(trained_model)
        self.local_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])    

    def create_model(self):
        # VGG16モデルと学習済み重みをロード
        # Fully-connected層（FC）はいらないのでinclude_top=False）
        input_tensor = Input(shape=(self.img_row,self.img_col,3))
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        
        # FC層を構築
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self.nb_classes, activation='softmax'))

        # VGG16とFCを接続
        model = Model(input=vgg16.input, output=top_model(vgg16.output))
        
        # 最後のconv層の直前までの層をfreeze
        for layer in model.layers[:18]:
            layer.trainable = False

        model.summary()

        # Fine-tuningのときはSGDの方がよい
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
        return model
    
    def train(self,resume=False):
        train_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory('./new_train/', target_size=(self.img_row,self.img_col), batch_size=self.batch_size, classes=self.classes_name)

        val_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        val_generator = val_datagen.flow_from_directory('./new_train_validate/', target_size=(self.img_row,self.img_col), batch_size=self.val_batch_size, classes=self.classes_name)

        if resume and self.local_model is not None:
            model = self.local_model
        else:
            model = self.create_model()

        filepath="neji-{epoch:02d}-{loss:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        #history = model.fit_generator(train_generator,steps_per_epoch=self.steps_per_epoch,epochs=self.epochs)
        history = model.fit_generator(train_generator,steps_per_epoch=self.steps_per_epoch,epochs=self.epochs,validation_data=val_generator,validation_steps=self.val_steps_per_epoch, callbacks=callbacks_list)
        
        # list all data in history
        print(history.history.keys())
        now = datetime.datetime.now()
        time = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + ' ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)

        # summarize history for accuracy

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy_' + time + '.jpg')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss_' + time + '.jpg')
        plt.show()

        #model.save("neji.h5")



    def evaluate(self):
        test_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_generator = test_datagen.flow_from_directory('./test/', target_size=(self.img_row,self.img_col), batch_size=self.batch_size, classes=self.classes_name)
       
        self.local_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
        score = model.evaluate_generator(test_generator,steps=10)
        print(score)

    def print_eval(self):
        number = 0
        dir_path = './new_train_validate/OK'
        for filename in os.listdir(dir_path):
            img = cv2.imread(dir_path + "/" + filename)
            result = model.test(img)
            print(filename + ": " + str(result) +"\n")
            if (result == 1):
                number +=1
        print(str(number) + "/xx")
   
    def test(self,img):
        #test_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        img = cv2.imread(img)
        img = cv2.resize(img,(self.img_row,self.img_col))
        img = np.reshape(img,[1,self.img_row,self.img_col,3])
        img = img / 255.
        
        
        #y_pred1 = model.predict_generator(test_image)
        #print(np.argmax(y_pred1,axis=1))
        y_pred2 = self.local_model.predict(img)
        print(np.argmax(y_pred2,axis=1))
        return np.argmax(y_pred2,axis=1)


model = CustomModel()
model.load("neji-47-0.2617.h5")
#model.train(resume=False)
#model.print_eval()
#model.evaluate()
#model.test("train/lego_anomaly/2019-04-01 10:09:48.764466-4.jpg")
model.test("test.jpg")
#model.train()



