import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, num_of_classes):
        model = Sequential()
        inputShape = K.image_data_format() == "channels_first" and (depth, height, width) or (height, width, depth)

        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape, activation="relu"))
        model.add(Flatten())
        model.add(Dense(num_of_classes, activation="softmax"))

        return model
