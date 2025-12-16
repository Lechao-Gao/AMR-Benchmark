import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, TimeDistributed, Reshape
except ImportError:
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, TimeDistributed, Reshape

def DAEModel(weights=None,
             input_shape=[128, 2],
             classes=11,
             **kwargs):
    """
    DAE (Denoising Auto-Encoder)模型定义
    
    参数:
        weights: 预训练权重文件路径，如果为None则随机初始化
        input_shape: 输入数据形状，默认为[128, 2]
        classes: 分类数量，默认为11
        
    返回:
        model: Keras模型，具有两个输出：分类输出和重构输出
    """
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input_layer = Input(shape=input_shape, name='input')
    x = input_layer
    dr = 0  # dropout rate

    # LSTM层 - 使用标准LSTM替代CuDNNLSTM
    # 添加use_cudnn='auto'参数以在可用时使用CUDNN加速
    x, s, c = LSTM(units=32, return_state=True, return_sequences=True)(x)
    x = Dropout(dr)(x)
    x, s1, c1 = LSTM(units=32, return_state=True, return_sequences=True)(x)
    
    # 分类器
    xc = Dense(32, activation='relu')(s1)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(16, activation='relu')(xc)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(classes, activation='softmax', name='xc')(xc)

    # 解码器
    xd = TimeDistributed(Dense(2), name='xd')(x)

    model = Model(inputs=input_layer, outputs=[xc, xd])

    # 加载权重
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = DAEModel(None, input_shape=(128, 2), classes=11)
    model.summary()