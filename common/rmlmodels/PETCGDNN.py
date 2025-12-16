import os
import tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"
WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')
import math
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape, \
        GaussianNoise, Activation
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import LSTM, Lambda, Multiply, Add, Subtract, MaxPool2D, GRU, LeakyReLU, BatchNormalization
except ImportError:
    from keras.models import Model
    from keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape, \
        GaussianNoise, Activation
    from keras.layers import Conv2D
    from keras.layers import LSTM, Lambda, Multiply, Add, Subtract, MaxPool2D, GRU, LeakyReLU, BatchNormalization

def cal1(x):
    y = tf.keras.backend.cos(x)
    return y

def cal2(x):
    y = tf.keras.backend.sin(x)
    return y

def PETCGDNN(weights=None,
           input_shape=[128, 2],
           input_shape2=[128],
           classes=11,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5 # dropout rate (%)
    input = Input(input_shape+[1], name='input1')
    input1 = Input(input_shape2, name='input2')
    input2 = Input(input_shape2, name='input3')

    x1 = Flatten()(input)
    x1 = Dense(1, name='fc2')(x1)
    x1 = Activation('linear')(x1)

    cos1= Lambda(cal1)(x1)
    sin1 = Lambda(cal2)(x1)
    x11 = Multiply()([input1, cos1])
    x12 = Multiply()([input2, sin1])
    x21 = Multiply()([input2, cos1])
    x22 = Multiply()([input1, sin1])
    y1 = Add()([x11,x12])
    y2 = Subtract()([x21,x22])
    y1 = Reshape(target_shape=(128, 1), name='reshape1')(y1)
    y2 = Reshape(target_shape=(128, 1), name='reshape2')(y2)
    x11 = concatenate([y1, y2])
    x3 = Reshape(target_shape=((128, 2, 1)), name='reshape3')(x11)

    # spatial feature
    x3 = Conv2D(75, (8,2), padding='valid', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(
        x3)
    x3 = Conv2D(25, (5,1), padding='valid', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(
        x3)
    # temporal feature
    x4 = Reshape(target_shape=((117,25)), name='reshape4')(x3)
    x4= GRU(units=128)(x4)

    x = Dense(classes, activation='softmax', name='softmax')(x4)

    model = Model(inputs = [input,input1,input2], outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

try:
    from tensorflow import keras
    from tensorflow.keras.utils import plot_model
except ImportError:
    import keras
    from keras.utils import plot_model

if __name__ == '__main__':
    model = PETCGDNN(None, classes=10)

    # 兼容 TensorFlow 2.x: 移除不支持的 decay 参数
    try:
        adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    except TypeError:
        # 如果某些参数不支持，尝试简化版本
        adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    plot_model(model, to_file='model.png', show_shapes=True)  # print model
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())