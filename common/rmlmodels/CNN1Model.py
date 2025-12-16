import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
try:
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.keras.layers import Reshape, Dense, Dropout, Activation, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
except ImportError:
    import keras
    import keras.models as models
    from keras.layers import Reshape, Dense, Dropout, Activation, Flatten
    from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

def CNN1Model(weights=None,
             input_shape=[2, 128, 1],
             classes=11,
             **kwargs):
    """
    CNN1模型定义 - 单卷积层CNN模型
    
    参数:
        weights: 预训练权重文件路径，如果为None则随机初始化
        input_shape: 输入数据形状，应为[2, 128, 1]
        classes: 分类数量
        
    返回:
        model: Keras模型
    """
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    dr = 0.5  # dropout rate (%)
    model = models.Sequential()
    
    # 只使用一个卷积层，而不是CNN2中的两个
    model.add(Conv2D(50, (2, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform', input_shape=input_shape))
    model.add(Dropout(dr))
    
    # 展平层
    model.add(Flatten())
    
    # 全连接层
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(classes, kernel_initializer='he_normal', name="dense2"))
    model.add(Activation('softmax'))

    # 加载权重
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    print(CNN1Model().summary())