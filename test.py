import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from swiftcore.vision.applications import ShallowNet, VGG16, ResNet50V2, ResNet50, MobileNetV2, MobileNet, InceptionV3
from swiftcore.vision.applications import ResNet101, ResNet152, Xception, VGG19, ResNet101V2, ResNet152V2
from swiftcore.vision.applications import ResNet10, ResNet18, ResNet34
import argparse



ap = argparse.ArgumentParser()

ap.add_argument("-mn", "--model_name", required=False, help="path to input data")
args = vars(ap.parse_args())

model_val = {
    'inception_v3': InceptionV3,
    'vgg16': VGG16,
    'vgg19': VGG19,
    'resnet10': ResNet10,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet50v2': ResNet50V2,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'resnet101' : ResNet101,
    'resnet101v2' : ResNet101V2,
    'resnet152' : ResNet152,
    'resnet152v2' : ResNet152V2,
    'xception' : Xception
}

print(list(model_val.keys()))


def create_model(name=None, image=None):
    if name not in model_val.keys():
        print(f"{name} model is not available")
        return

    from tensorflow.keras.applications import imagenet_utils
    processed_inp = imagenet_utils.preprocess_input
    input_shape = (224, 224)
    image = "sample_img/" + image + ".jpg"

    if name in ['inception_v3', 'xception']:
        from swiftcore.vision.applications.inceptionv3 import preprocess_input
        processed_inp = preprocess_input
        input_shape = (299, 299)

    model = model_val[name]
    model = model.build()

    img = load_img(image, target_size=input_shape)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(processed_inp(img))
    P = imagenet_utils.decode_predictions(pred)

    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    orig = cv2.imread(image)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(orig, "Label: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)


name = args["model_name"]
image = 'cup'
create_model(name=name, image=image)
# ['inception_v3', 'vgg16', 'vgg19',_ 'resnet50', 'resnet50v2', 'mobilenet', 'mobilenetv2', 'resnet101', 'resnet101v2', 'resnet152', 'resnet152v2', 'xception']
# python test.py --model_name 'mobilenetv2'
