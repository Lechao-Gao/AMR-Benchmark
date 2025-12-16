import os
import numpy as np
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Dense,
        Dropout,
        Activation,
        Flatten,
        Add,
        Conv2D,
    )
except ImportError:
    from keras.models import Model
    from keras.layers import (
        Input,
        Dense,
        Dropout,
        Activation,
        Flatten,
        Add,
        Conv2D,
    )

def ResNetModel(weights=None,
             input_shape=[2, 128, 1],
             classes=11,
             **kwargs):
    """
    ResNet模型定义
    
    参数:
        weights: 预训练权重文件路径，如果为None则随机初始化
        input_shape: 输入数据形状，默认为[2, 128, 1]
        classes: 分类数量，默认为11
        
    返回:
        model: Keras模型
    """
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    dr = 0.6  # dropout rate
    
    # 输入层
    input_layer = Input(shape=input_shape, name='input')
    
    # 第一个卷积层
    x = Conv2D(256, (1, 3), padding='same', name="conv1", kernel_initializer='glorot_uniform')(input_layer)
    x = Activation('relu')(x)
    
    # 第二个卷积层
    x = Conv2D(256, (2, 3), padding='same', name="conv2", kernel_initializer='glorot_uniform')(x)
    
    # 残差连接
    x1 = Add()([input_layer, x])
    x1 = Activation('relu')(x1)
    
    # 第三个卷积层
    x = Conv2D(80, (1, 3), activation="relu", padding='same', name="conv3", kernel_initializer='glorot_uniform')(x1)
    
    # 第四个卷积层
    x = Conv2D(80, (1, 3), activation="relu", padding='same', name="conv4", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    
    # 展平
    x = Flatten()(x)
    
    # 全连接层
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(dr)(x)
    
    # 输出层
    output = Dense(classes, activation='softmax', name='softmax')(x)
    
    # 创建模型
    model = Model(inputs=input_layer, outputs=output)
    
    # 加载权重
    if weights is not None:
        model.load_weights(weights)
    
    return model

if __name__ == '__main__':
    model = ResNetModel(None, input_shape=[2, 128, 1], classes=11)
    model.summary()