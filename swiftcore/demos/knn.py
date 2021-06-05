import sys
sys.path.append("")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse
from swiftcore.vision.imageutil import SimplePreprocessor, SimpleDatasetLoader

aps = argparse.ArgumentParser()
aps.add_argument("-d", "--dataset", required=False, help="path to input dataset")
aps.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
aps.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(aps.parse_args())

print("[INFO] loading images")

# image_paths = list(paths.list_images(args['dataset']))
image_paths = list(paths.list_images("E:\Computer Vision\swiftpackages\\basicclassifier\datasets\\animals"))

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
data =data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)


print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(x_train, y_train)
print(classification_report(y_test, model.predict(x_test), target_names=le.classes_))
