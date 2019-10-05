import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import datasets

class CnnModel:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 15
        self.IMG_HEIGHT = 150
        self.IMG_WIDTH = 150
        self.train_dir = None
        self.validation_dir = None
        self.train_cats_dir = None
        self.train_dogs_dir = None
        self.validation_cats_dir = None
        self.validation_dogs_dir = None
        self.model = None

    def execute(self):
        self.download_data()
        sample_training_images = self.preparation_data()
        self.plotImages(sample_training_images[:5])

    def download_data(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 픽셀 값을 0~1 사이로 정규화합니다.
        train_images, test_images = train_images / 255.0, test_images / 255.0
        _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
        PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
        self.train_dir = os.path.join(PATH, 'train')
        self.validation_dir = os.path.join(PATH, 'validation')
        self.train_cats_dir = os.path.join(self.train_dir, 'cats')  # directory with our training cat pictures
        self.train_dogs_dir = os.path.join(self.train_dir, 'dogs')  # directory with our training dog pictures
        self.validation_cats_dir = os.path.join(self.validation_dir, 'cats')  # directory with our validation cat pictures
        self.validation_dogs_dir = os.path.join(self.validation_dir, 'dogs')  # directory with our validation dog pictures
        num_cats_tr = len(os.listdir(self.train_cats_dir))
        num_dogs_tr = len(os.listdir(self.train_dogs_dir))
        num_cats_val = len(os.listdir(self.validation_cats_dir))
        num_dogs_val = len(os.listdir(self.validation_dogs_dir))
        total_train = num_cats_tr + num_dogs_tr
        total_val = num_cats_val + num_dogs_val
        print('total training cat images:', num_cats_tr)
        print('total training dog images:', num_dogs_tr)
        print('total validation cat images:', num_cats_val)
        print('total validation dog images:', num_dogs_val)
        print("--")
        print("Total training images:", total_train)
        print("Total validation images:", total_val)

    def preparation_data(self)->object:
        train_image_generator \
            = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
        validation_image_generator \
            = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
        train_data_gen \
            = train_image_generator\
            .flow_from_directory(batch_size=self.batch_size,
                                 directory=self.train_dir,
                                 shuffle=True,
                                 target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                 class_mode='binary')
        val_data_gen \
            = validation_image_generator\
            .flow_from_directory(batch_size=self.batch_size,
                                 directory=self.validation_dir,
                                 target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                 class_mode='binary')
        sample_training_images, _ = next(train_data_gen)
        return sample_training_images

    def plotImages(self,images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def create_model(self):
        self.model = Sequential([
            Conv2D(16, 3, padding='same',
                   activation='relu',
                   input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

if __name__ == '__main__':
    m = CnnModel()
    m.execute()