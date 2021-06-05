import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils

class MobileNet:
    @staticmethod
    def build(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              classifier_activation='softmax',
              **kwargs
              ):

        """Instantiates the MobileNet architecture.

  Reference paper:
  - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
    Applications](https://arxiv.org/abs/1704.04861)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in the `tf.keras.backend.image_data_format()`.

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.mobilenet.preprocess_input` for an example.

  Arguments:
    input_shape: Optional shape tuple, only to be specified if `include_top`
      is False (otherwise the input shape has to be `(224, 224, 3)` (with
      `channels_last` data format) or (3, 224, 224) (with `channels_first`
      data format). It should have exactly 3 inputs channels, and width and
      height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
      valid value. Default to `None`.
      `input_shape` will be ignored if the `input_tensor` is provided.
    alpha: Controls the width of the network. This is known as the width
      multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
      decreases the number of filters in each layer. - If `alpha` > 1.0,
      proportionally increases the number of filters in each layer. - If
      `alpha` = 1, default number of filters from the paper are used at each
      layer. Default to 1.0.
    depth_multiplier: Depth multiplier for depthwise convolution. This is
      called the resolution multiplier in the MobileNet paper. Default to 1.0.
    dropout: Dropout rate. Default to 0.001.
    include_top: Boolean, whether to include the fully-connected layer at the
      top of the network. Default to `True`.
    weights: One of `None` (random initialization), 'imagenet' (pre-training
      on ImageNet), or the path to the weights file to be loaded. Default to
      `imagenet`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to
      use as image input for the model. `input_tensor` is useful for sharing
      inputs between multiple different networks. Default to None.
    pooling: Optional pooling mode for feature extraction when `include_top`
      is `False`.
      - `None` (default) means that the output of the model will be
          the 4D tensor output of the last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will be applied.
    classes: Optional number of classes to classify images into, only to be
      specified if `include_top` is True, and if no `weights` argument is
      specified. Defaults to 1000.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    **kwargs: For backwards compatibility only.
  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """

        model = tf.keras.applications.MobileNet(input_shape=input_shape,
                                              alpha=alpha,
                                              depth_multiplier=depth_multiplier,
                                              dropout=dropout,
                                              include_top=include_top,
                                              weights=weights,
                                              input_tensor=input_tensor,
                                              pooling=pooling,
                                              classes=classes,
                                              classifier_activation=classifier_activation,
                                              **kwargs
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