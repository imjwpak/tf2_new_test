import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FashionModel:
    def __init__(self):
        pass

    def execute(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        print('-------Train set spec ----------')
        print('훈련이미지 :', train_images.shape)
        print('훈련이미지 수 :', len(train_labels))
        print('훈련이미지 라벨:', train_labels)
        print('-------Test set spec ----------')
        print('테스트이미지 :', test_images.shape)
        print('테스트이미지 수:', len(test_labels))
        print('테스트이미지 라벨:', test_labels)

if __name__ == '__main__':
    f = FashionModel()
    f.execute()