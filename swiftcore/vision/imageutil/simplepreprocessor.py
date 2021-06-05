import cv2

class SimplePreprocessor:
    """
    Summary :
        This a simple class to perform basic preprocessing operations.

    Parameters :
        width: The target width of our input image after resizing.
        height: The target height of our input image after resizing.
        inter: An optional parameter used to control which interpolation algorithm is used when resizing.
    """


    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """
        accepts the image and does the required processing(resizing)
        :param image:
        :return: returns the processed image to the calling function
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
