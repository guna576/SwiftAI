import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, Input, Activation, add
from tensorflow.keras.regularizers import l2


class ResNet10:

    @staticmethod
    def res_module(data, filters, stride, red=False, reg=0.0001, mom=0.9, eps=2e-5):
        short = data

        bn1 = BatchNormalization(epsilon=eps, momentum=mom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(filters, (3, 3), strides=1, padding="same", kernel_regularizer=l2(reg), use_bias=False)(act1)
        print(conv1)

        bn2 = BatchNormalization(epsilon=eps, momentum=mom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(filters, (3, 3), strides=stride, padding="same", kernel_regularizer=l2(reg), use_bias=False)(
            act2)
        print(conv2)
        if red:
            short = Conv2D(filters, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
            print(short)

        return add([conv2, short])

    @staticmethod
    def build(inputShape=(160, 160, 3), classes=2):
        inp = Input(shape=(32, 32, 3))
        x = BatchNormalization(epsilon=2e-5, momentum=0.9)(inp)
        x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), use_bias=False, kernel_regularizer=l2(0.0001))(x)
        x = MaxPool2D((3, 3), strides=2)(x)

        # block1
        x = ResNet10.res_module(x, 64, 1, red=True)
        x = ResNet10.res_module(x, 64, 1, )

        # block2
        x = ResNet10.res_module(x, 128, 2, red=True)
        x = ResNet10.res_module(x, 128, 1)

        x = BatchNormalization(epsilon=2e-5, momentum=0.9)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((4, 4))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(0.0001))(x)
        x = Activation("softmax")(x)

        model = tf.keras.models.Model(inp, x, name="resnet10")

        return model