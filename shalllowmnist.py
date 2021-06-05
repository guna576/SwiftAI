from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from swiftcore.vision.imageutil import SimplePreprocessor, SimpleDatasetLoader, ImageToArrayPreprocessor
from swiftcore.vision.applications import ShallowNet


# ap = argparse.ArgumentParser()


# ap.add_argument("-ta", "--target_accuracy", required=False, help="path to input data")
# args = vars(ap.parse_args())

# print("[Info] loading images")
# target_accuracy = args["target_accuracy"] == None and 95 or int(args["target_acccuracy"])


# sp = SimplePreprocessor(32, 32)
# iap = ImageToArrayPreprocessor()

# sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
# (data, labels) = sdl.load(image_paths, verbose=500)
# data = data.astype("float")/255.0


(x_train, y_train),( x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=28, height=28, depth=1, num_of_classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("acc")> 97):
            self.model.stop_training = True

callbacks = MyCallback()

print("[INFO] training network...")
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=32, epochs=3, verbose=1, callbacks=callbacks)

print("[INFO] evaluating network...")
predictions = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1),target_names=["0", "1", "2","3","4","5",'6','7','8','9']))

