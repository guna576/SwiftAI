import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils

class MobileNetV2:
    @staticmethod
    def build(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                # classifier_activation='softmax',
                **kwargs
              ):

        """
        Arguments:
    input_shape: Optional shape tuple, to be specified if you would
      like to use a model with an input image resolution that is not
      (224, 224, 3).
      It should have exactly 3 inputs channels (224, 224, 3).
      You can also omit this option if you would like
      to infer input_shape from an input_tensor.
      If you choose to include both input_tensor and input_shape then
      input_shape will be used if they match, if the shapes
      do not match then we will throw an error.
      E.g. `(160, 160, 3)` would be one valid value.
    alpha: Float between 0 and 1. controls the width of the network.
      This is known as the width multiplier in the MobileNetV2 paper,
      but the name is kept for consistency with `applications.MobileNetV1`
      model in Keras.
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
          are used at each layer.
    include_top: Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to `True`.
    weights: String, one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: Optional Keras tensor (i.e. output of
      `layers.Input()`)
      to use as image input for the model.
    pooling: String, optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model
          will be the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a
          2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: Integer, optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    **kwargs: For backwards compatibility only.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape or invalid alpha, rows when
      weights='imagenet'
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
    """

        model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                              alpha=alpha,
                                              include_top=include_top,
                                              weights=weights,
                                              input_tensor=input_tensor,
                                              pooling=pooling,
                                              classes=classes,
                                              # classifier_activation=classifier_activation,
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