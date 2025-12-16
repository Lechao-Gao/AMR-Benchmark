import os
import numpy as np
import tensorflow as tf

try:
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Dense, Conv1D, Conv2D, MaxPool1D, MaxPool2D,
                                         ReLU, Dropout, Softmax, concatenate, LSTM, GRU,
                                         Permute, Reshape, ZeroPadding2D, Activation,
                                         Add, GlobalAveragePooling1D, GlobalAveragePooling2D,
                                         BatchNormalization, Lambda, Multiply,
                                         LayerNormalization, MultiHeadAttention)
    from tensorflow.keras.activations import swish
except ImportError:
    import keras
    from keras import backend as K
    from keras.models import Model
    from keras.layers import (Input, Dense, Conv1D, Conv2D, MaxPool1D, MaxPool2D,
                             ReLU, Dropout, Softmax, concatenate, LSTM, GRU,
                             Permute, Reshape, ZeroPadding2D, Activation,
                             Add, GlobalAveragePooling1D, GlobalAveragePooling2D,
                             BatchNormalization, Lambda, Multiply)
    try:
        from keras.layers import LayerNormalization, MultiHeadAttention
    except ImportError:
        from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
    try:
        from keras.activations import swish
    except ImportError:
        from tensorflow.keras.activations import swish

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

class MambaBlock(tf.keras.layers.Layer):
    """
    优化的Mamba块实现，增强训练稳定性
    """
    def __init__(self, d_model, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = d_model

        # 增强的Mamba结构
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.conv1d = Conv1D(d_model, 3, padding='same', activation='swish')
        self.proj = Dense(d_model * 2, use_bias=True,
                         kernel_initializer='glorot_uniform')
        self.out_proj = Dense(d_model, use_bias=True,
                             kernel_initializer='glorot_uniform')
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, x, training=None):
        """
        增强的前向传播，提高训练稳定性
        """
        # 残差连接
        residual = x
        x = self.norm1(x, training=training)

        # 卷积处理
        x = self.conv1d(x, training=training)
        x = self.dropout1(x, training=training)

        # 投影和门控
        xz = self.proj(x)  # (batch, length, d_model * 2)
        x, z = tf.split(xz, 2, axis=-1)  # 各自 (batch, length, d_model)

        # 门控机制 - 添加稳定性
        x = x * tf.nn.sigmoid(z)
        x = self.dropout2(x, training=training)

        # 输出投影
        output = self.out_proj(x)
        output = self.norm2(output, training=training)

        return output + residual
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
        })
        return config

class DualAttentionBlock(tf.keras.layers.Layer):
    """
    优化的双注意力块，增强训练稳定性
    """
    def __init__(self, d_model, num_heads=8, dropout_rate=0.15, **kwargs):
        super(DualAttentionBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # 自注意力 - 优化配置
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            use_bias=True,
            kernel_initializer='glorot_uniform'
        )

        # 交叉注意力 - 优化配置
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            use_bias=True,
            kernel_initializer='glorot_uniform'
        )

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)

        # 优化的前馈网络
        self.ffn = tf.keras.Sequential([
            Dense(d_model * 2, activation='gelu',  # 减少FFN大小，使用GELU激活
                  kernel_initializer='glorot_uniform'),
            Dropout(dropout_rate),
            Dense(d_model, kernel_initializer='glorot_uniform')
        ])

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, training=None):
        # 自注意力分支
        attn1 = self.self_attention(x, x, training=training)
        x1 = self.norm1(x + self.dropout1(attn1, training=training))

        # 交叉注意力分支
        attn2 = self.cross_attention(x1, x1, training=training)
        x2 = self.norm2(x1 + self.dropout2(attn2, training=training))

        # 前馈网络
        ffn_out = self.ffn(x2, training=training)
        output = self.norm3(x2 + self.dropout3(ffn_out, training=training))

        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

class SmallKernelConv(tf.keras.layers.Layer):
    """
    小核卷积层，用于细粒度特征提取和噪声抑制
    """
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(SmallKernelConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv = Conv1D(filters, kernel_size, padding='same')
        self.norm = BatchNormalization()
        self.activation = ReLU()
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config

def MambaAttentionModel(weights=None,
                       input_shape=[2, 128],
                       classes=11,
                       d_model=256,
                       num_mamba_layers=4,
                       num_attention_layers=2,
                       **kwargs):
    """
    Mamba双注意力模型
    
    参数:
        weights: 预训练权重路径
        input_shape: 输入形状 [I/Q通道数, 序列长度]
        classes: 分类数量
        d_model: 模型维度
        num_mamba_layers: Mamba层数量
        num_attention_layers: 注意力层数量
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                        '`None` (random initialization), '
                        'or the path to the weights file to be loaded.')
    
    # 多特征输入
    input_iq = Input(shape=input_shape + [1], name='input_iq')  # I/Q数据
    
    # === 多特征输入模块 ===
    
    # 1. I/Q数据处理分支
    x_iq = Reshape((input_shape[1], input_shape[0]))(input_iq)  # (batch, 128, 2)
    x_iq = Conv1D(64, 7, padding='same', activation='relu', name='iq_conv1')(x_iq)
    x_iq = BatchNormalization()(x_iq)
    x_iq = Conv1D(128, 5, padding='same', activation='relu', name='iq_conv2')(x_iq)
    x_iq = BatchNormalization()(x_iq)
    
    # 2. 幅度/相位序列处理分支
    # 计算幅度和相位
    iq_complex = Lambda(
        lambda x: tf.complex(x[:, :, 0], x[:, :, 1]),
        output_shape=lambda input_shape: (input_shape[0], input_shape[1])
    )(x_iq)
    magnitude = Lambda(
        lambda x: tf.abs(x),
        output_shape=lambda input_shape: input_shape
    )(iq_complex)
    phase = Lambda(
        lambda x: tf.math.angle(x),
        output_shape=lambda input_shape: input_shape
    )(iq_complex)
    
    # 合并幅度和相位
    amp_phase = Lambda(
        lambda x: tf.stack([x[0], x[1]], axis=-1),
        output_shape=lambda input_shape: (input_shape[0][0], input_shape[0][1], 2)
    )([magnitude, phase])
    x_amp_phase = Conv1D(64, 5, padding='same', activation='relu', name='amp_phase_conv1')(amp_phase)
    x_amp_phase = BatchNormalization()(x_amp_phase)
    x_amp_phase = Conv1D(128, 3, padding='same', activation='relu', name='amp_phase_conv2')(x_amp_phase)
    x_amp_phase = BatchNormalization()(x_amp_phase)
    
    # 3. 时频图处理分支 (使用STFT的简化版本)
    # 这里使用卷积来模拟时频特征提取
    x_spectrogram = Reshape((input_shape[0], input_shape[1], 1))(input_iq)
    x_spectrogram = Conv2D(32, (3, 3), padding='same', activation='relu', name='spec_conv1')(x_spectrogram)
    x_spectrogram = MaxPool2D((1, 2))(x_spectrogram)
    x_spectrogram = Conv2D(64, (3, 3), padding='same', activation='relu', name='spec_conv2')(x_spectrogram)
    x_spectrogram = GlobalAveragePooling2D()(x_spectrogram)
    x_spectrogram = Dense(128, activation='relu')(x_spectrogram)
    x_spectrogram = Lambda(
        lambda x: tf.expand_dims(x, 1),
        output_shape=lambda input_shape: (input_shape[0], 1, input_shape[1])
    )(x_spectrogram)  # (batch, 1, 128)
    x_spectrogram = Lambda(
        lambda x: tf.tile(x, [1, input_shape[1], 1]),
        output_shape=lambda input_shape: (input_shape[0], input_shape[1], input_shape[2])
    )(x_spectrogram)  # (batch, 128, 128)
    
    # === 增强特征融合 ===
    # 先对每个分支进行归一化
    x_iq_norm = LayerNormalization(epsilon=1e-6)(x_iq)
    x_amp_phase_norm = LayerNormalization(epsilon=1e-6)(x_amp_phase)
    x_spectrogram_norm = LayerNormalization(epsilon=1e-6)(x_spectrogram)
    
    # 加权融合 - 使用注意力机制
    x_fused = Add()([x_iq_norm, x_amp_phase_norm, x_spectrogram_norm])
    x_fused = Dense(d_model, activation='gelu', name='feature_fusion',
                   kernel_initializer='glorot_uniform')(x_fused)
    x_fused = LayerNormalization(epsilon=1e-6)(x_fused)
    x_fused = Dropout(0.1)(x_fused)

    # === 小核卷积层 ===
    x = SmallKernelConv(d_model, kernel_size=3)(x_fused)

    # === 优化的Mamba主干结构 ===
    for i in range(num_mamba_layers):
        x = MambaBlock(d_model, name=f'mamba_block_{i}')(x)

        # 更频繁地插入注意力块以提高表达能力
        if (i + 1) % 2 == 0:
            x = DualAttentionBlock(d_model, name=f'dual_attention_{i//2}')(x)

    # === 增强的分类头 ===
    # 多尺度池化
    global_avg = GlobalAveragePooling1D()(x)
    global_max = Lambda(lambda x: tf.reduce_max(x, axis=1))(x)
    
    # 融合不同池化结果
    x_pooled = concatenate([global_avg, global_max], name='multi_scale_pooling')
    
    # 渐进式分类层
    x = Dense(512, activation='gelu', name='dense1',
              kernel_initializer='glorot_uniform')(x_pooled)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='gelu', name='dense2',
              kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='gelu', name='dense3',
              kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.2)(x)
    
    output = Dense(classes, activation='softmax', name='classification_output',
                   kernel_initializer='glorot_uniform')(x)
    
    # 构建模型
    model = Model(inputs=input_iq, outputs=output, name='MambaAttentionModel')
    
    # 加载权重
    if weights is not None:
        model.load_weights(weights)
    
    return model

if __name__ == '__main__':
    # 测试模型
    model = MambaAttentionModel(
        input_shape=[2, 128],
        classes=11,
        d_model=256,
        num_mamba_layers=4,
        num_attention_layers=2
    )
    
    # 编译模型
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=optimizer
    )
    
    print('模型层数:', len(model.layers))
    print('模型参数总数:', model.count_params())
    model.summary()