import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Softmax
except ImportError:
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Softmax

def LSTM2Model(weights=None,
             input_shape=[128, 2],
             classes=11,
             **kwargs):
    """
    LSTM2模型定义
    
    参数:
        weights: 预训练权重文件路径，如果为None则随机初始化
        input_shape: 输入数据形状，默认为[128, 2]
        classes: 分类数量，默认为11
        
    返回:
        model: Keras模型
    """
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input_layer = Input(shape=input_shape, name='input')
    x = input_layer

    # LSTM层 - 移除不支持的use_cudnn参数
    x = LSTM(units=128, return_sequences=True)(x)
    x = LSTM(units=128)(x)

    # 输出层
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    # 加载权重
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = LSTM2Model(None, input_shape=(128, 2), classes=11)
    model.summary()