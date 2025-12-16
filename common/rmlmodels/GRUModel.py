import os

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax
    from tensorflow.keras.layers import Bidirectional,Flatten,GRU
    from tensorflow.keras.utils import plot_model
except ImportError:
    from keras.models import Model
    from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax
    from keras.layers import Bidirectional,Flatten,GRU
    from keras.utils import plot_model

def GRUModel(weights=None,
             input_shape=[128,2],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input = Input(input_shape,name='input')
    x = input

    #GRU Unit
    x = GRU(units=128,return_sequences=True)(x)
    x = GRU(units=128)(x)

    #DNN
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = input,outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

try:
    from tensorflow import keras
except ImportError:
    import keras
if __name__ == '__main__':
    model = GRUModel(None,input_shape=(128,2),classes=11)

    # 兼容 TensorFlow 2.x: lr -> learning_rate, 移除不支持的 decay 参数
    try:
        adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    except TypeError:
        # 如果 learning_rate 不支持，尝试使用 lr
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    plot_model(model, to_file='model.png',show_shapes=True) # print model

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())