import json
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from typing import List, Tuple
from tensorflow.python.keras import backend as K
import tensorflow.python.keras.layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Activation, Lambda, BatchNormalization
from tensorflow.python.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.python.keras.layers import Convolution1D, Dense
from tensorflow.python.keras.models import Input, Model
import tensorflow as tf
from tensorflow.python import keras

def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0):
    # type: (Layer, int, int, int, str, float) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    
    prev_x = x
    for k in range(2):
        zero_padding = (kernel_size - 1) * dilation_rate
        
        x = tf.keras.layers.Lambda(lambda inputs: 
                                   tf.pad(inputs, 
                                          tf.constant([(0, 0,), (1, 0), (0, 0)]) * zero_padding)
                                  )(x)
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding='valid')(x)
#         x = BatchNormalization()(x)  # TODO should be WeightNorm here.
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
        
    prev_x = Conv1D(nb_filters, 1, padding='valid')(prev_x)
    
    
    res_x = keras.layers.add([prev_x, x])
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        # print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations


class TCN:
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        # 1D FCN.
        x = Convolution1D(self.nb_filters, 1, padding=self.padding)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                x, skip_out = residual_block(x,
                                             dilation_rate=d,
                                             nb_filters=self.nb_filters,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             dropout_rate=self.dropout_rate)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x


def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=True,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 opt='adam',
                 lr=0.002):
    # type: (...) -> keras.Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.
        opt: Optimizer name.
        lr: Learning rate.
    Returns:
        A compiled keras TCN.
    """

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences, name)(input_layer)

    print('x.shape=', x.shape)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(1)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print(f'model.x = {input_layer.shape}')
    print(f'model.y = {output_layer.shape}')
    print('Adam with norm clipping.')
    return model


def tcn(input_shape, n_classes):
    x_base = x = tf.keras.layers.Input(shape=input_shape)
    x = TCN(256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32, 64])(x)
    x = tf.keras.layers.Activation('relu')(x)

    logits = tf.keras.layers.Dense(n_classes, activation=None)(x)
    pred = Activation('softmax')(logits)
    
    return tf.keras.models.Model(x_base, [logits, pred])

def vgg_extractor(ngf=64):
    inputs = Input(shape=(48, None, 1), name='the_input')
    #1. convnet layers
    m = Conv2D(
        64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv1')(inputs)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Dropout(0.5, name='drop1')(m)
    m = Conv2D(
        128,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Dropout(0.5, name='drop2')(m)
    m = Conv2D(
        256,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv3')(m)
    m = Conv2D(
        256,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(
        512,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv5')(m)
    m = BatchNormalization(axis=-1)(m)
    m = Conv2D(
        512,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='conv6')(m)
    m = BatchNormalization(axis=-1)(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(
        512,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        name='conv7')(m)
    
    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)
    model = Sequential(Model(inputs, m).layers[1:])
    return model


def build_model(n_classes, ngf=64):
    inputs = Input(shape=[48,None,1], name='basemodel_input')
    
    # two models are:
    vgg = Sequential(vgg_extractor(ngf=ngf).layers)
    
    # pass the inputs through these two models
    vgg_out = vgg(inputs)
    
    tcn_model = tcn(vgg_out.shape[1:], n_classes)


    cnn_logits, cnn_pred = tcn_model(vgg_out)
    basemodel = tf.keras.models.Model(inputs,  [cnn_logits, cnn_pred])
    return basemodel



