import numpy as np
import cv2
import os

class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        """
        accpets the preprocessors in form of list
        :param preprocessors: list
        """
        self.preprocessors = preprocessors and preprocessors or []

    def load(self, image_paths, verbose=-1):

        """
        This function loads the images from image paths and preprocesses them
        :param image_paths: list of image paths
        :param verbose: integer
        :return: returns a tuple of data and labels
        """
        data, labels = [], []

        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:

                print(f"[INFO] processed {i + 1}/{len(image_paths)}")

        return (np.array(data), np.array(labels))