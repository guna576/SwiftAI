import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils

class VGG19:
    @staticmethod
    def build(include_top = True,
              input_shape=None,
              weights="imagenet",
              input_tensor = None,
              pooling=None,
              classes = 1000,
              classifier_activation = "softmax"
              ):

        """
        Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.vgg19.preprocess_input` for an example.
        Arguments:
      include_top: whether to include the 3 fully-connected
          layers at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)`
          (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 input channels,
          and width and height should be no smaller than 32.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional block.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional block, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.

        Returns:
        A `keras.Model` instance.
        """

        model = tf.keras.applications.vgg19.VGG19(include_top = include_top,
                                                  input_shape= input_shape,
                                                  weights=weights,
                                                  input_tensor = input_tensor,
                                                  pooling = pooling,
                                                  classes = classes,
                                                  classifier_activation = classifier_activation
                                                  )

        return model


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.

    Arguments
    x: A 4D numpy array consists of RGB values within [0, 255].

    Returns
    Preprocessed array.

    Raises
    ValueError: In case of unknown `data_format` argument.
    """
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='caffe')


def decode_predictions(preds, top=5):
    """Decodes the prediction result from the model.

    Arguments
    preds: Numpy tensor encoding a batch of predictions.
    top: Integer, how many top-guesses to return.

    Returns
    A list of lists of top class prediction tuples
    `(class_name, class_description, score)`.
    One list of tuples per sample in batch input.

    Raises
    ValueError: In case of invalid shape of the `preds` array (must be 2D).
    """
    return imagenet_utils.decode_predictions(preds, top=top)