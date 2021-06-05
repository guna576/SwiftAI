from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from swiftcore.vision.imageutil import SimplePreprocessor, SimpleDatasetLoader, ImageToArrayPreprocessor
from swiftcore.vision.applications import ShallowNet


ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help="path to input data")
ap.add_argument("-ta", "--target_accuracy", required=False, help="path to input data")
args = vars(ap.parse_args())

print("[Info] loading images")
image_paths = list(paths.list_images(args["dataset"]))
target_accuracy = args["target_accuracy"] == None and 95 or int(args["target_acccuracy"])


sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float")/255.0


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, num_of_classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if(logs.get("acc")> target_accuracy):
            self.model.stop_training = True


callbacks = MyCallback()

print("[INFO] training network...")
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=32, epochs=100, verbose=1, callbacks=callbacks)

print("[INFO] evaluating network...")
predictions = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1),target_names=["cat", "dog", "panda"]))

