import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils

class InceptionV3:
    @staticmethod
    def build(include_top = True,
              input_shape=None,
              weights="imagenet",
              input_tensor = None,
              pooling=None,
              classes = 1000,
              classifier_activation = "softmax"
              ):

        """Instantiates the Inception v3 architecture.

  Reference paper:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in the `tf.keras.backend.image_data_format()`.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.inceptionv3.preprocess_input` for an example.

  Arguments:
    include_top: Boolean, whether to include the fully-connected
      layer at the top, as the last layer of the network. Default to `True`.
    weights: One of `None` (random initialization),
      `imagenet` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Default to `imagenet`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
      to use as image input for the model. `input_tensor` is useful for sharing
      inputs between multiple different networks. Default to None.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(299, 299, 3)` (with `channels_last` data format)
      or `(3, 299, 299)` (with `channels_first` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 75.
      E.g. `(150, 150, 3)` would be one valid value.
      `input_shape` will be ignored if the `input_tensor` is provided.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` (default) means that the output of the model will be
          the 4D tensor output of the last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Default to 1000.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """

        model = tf.keras.applications.InceptionV3(include_top = include_top,
                                                  input_shape= input_shape,
                                                  weights=weights,
                                                  input_tensor = input_tensor,
                                                  pooling = pooling,
                                                  classes = classes,
                                                  # classifier_activation = classifier_activation
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
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


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