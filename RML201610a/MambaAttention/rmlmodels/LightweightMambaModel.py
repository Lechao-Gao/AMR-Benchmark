#!/usr/bin/env python3
"""
优化的轻量级Mamba模型 - 2层Mamba版本
大幅减少计算复杂度，同时保持合理精度
"""

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
                                         LayerNormalization, MultiHeadAttention, Add,
                                         GlobalAveragePooling1D, GlobalAveragePooling2D,
                                         BatchNormalization, Lambda, Multiply)
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

class OptimizedMambaBlock(tf.keras.layers.Layer):
    """优化的Mamba块，减少参数量"""
    def __init__(self, d_model, **kwargs):
        super(OptimizedMambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        
        # 使用更少的参数
        self.norm = LayerNormalization()
        self.conv1d = Conv1D(d_model, 3, padding='same', activation='swish')
        self.proj = Dense(d_model * 2, use_bias=False)
        self.out_proj = Dense(d_model, use_bias=False)
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        residual = x
        x = self.norm(x, training=training)
        
        # 卷积处理
        x = self.conv1d(x, training=training)
        
        # 投影和门控
        xz = self.proj(x)
        x, z = tf.split(xz, 2, axis=-1)
        
        # 门控机制
        x = x * tf.nn.sigmoid(z)
        x = self.dropout(x, training=training)
        
        output = self.out_proj(x)
        return output + residual
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
        })
        return config

class EfficientAttentionBlock(tf.keras.layers.Layer):
    """高效注意力块"""
    def __init__(self, d_model, num_heads=4, dropout_rate=0.1, **kwargs):
        super(EfficientAttentionBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # 减少注意力头数
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        # 轻量级前馈网络
        self.ffn = tf.keras.Sequential([
            Dense(d_model * 2, activation='relu'),  # 从4倍减少到2倍
            Dropout(dropout_rate),
            Dense(d_model)
        ])
        
        self.dropout = Dropout(dropout_rate)
        
    def call(self, x, training=None):
        # 自注意力
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + self.dropout(attn, training=training))
        
        # 轻量级前馈网络
        ffn_out = self.ffn(x, training=training)
        output = self.norm2(x + self.dropout(ffn_out, training=training))
        
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
    """小核卷积层"""
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

def LightweightMambaModel(weights=None,
                         input_shape=[2, 128],
                         classes=11,
                         d_model=128,  # 从256减少到128
                         num_mamba_layers=2,  # 从4层减少到2层
                         num_attention_layers=1,  # 从2层减少到1层
                         **kwargs):
    """
    轻量级Mamba模型 - 2层Mamba版本
    
    参数:
        weights: 预训练权重路径
        input_shape: 输入形状 [I/Q通道数, 序列长度]
        classes: 分类数量
        d_model: 模型维度 (优化为128)
        num_mamba_layers: Mamba层数量 (优化为2层)
        num_attention_layers: 注意力层数量 (优化为1层)
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                        '`None` (random initialization), '
                        'or the path to the weights file to be loaded.')
    
    # 简化输入处理 - 只保留I/Q分支
    input_iq = Input(shape=input_shape + [1], name='input_iq')
    
    # === 简化特征提取 ===
    x = Reshape((input_shape[1], input_shape[0]))(input_iq)  # (batch, 128, 2)
    
    # 减少卷积层数和通道数
    x = Conv1D(32, 5, padding='same', activation='relu', name='iq_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, padding='same', activation='relu', name='iq_conv2')(x)
    x = BatchNormalization()(x)
    
    # 直接映射到目标维度
    x = Dense(d_model, activation='relu', name='feature_projection')(x)
    x = LayerNormalization()(x)
    
    # === 小核卷积层 ===
    x = SmallKernelConv(d_model, kernel_size=3)(x)
    
    # === 优化的Mamba主干结构 (2层) ===
    for i in range(num_mamba_layers):
        x = OptimizedMambaBlock(d_model, name=f'mamba_block_{i}')(x)
        
        # 只在最后一层后添加注意力
        if i == num_mamba_layers - 1 and num_attention_layers > 0:
            x = EfficientAttentionBlock(d_model, name='efficient_attention')(x)
    
    # === 简化的分类输出 ===
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu', name='dense1')(x)  # 从512减少到256
    x = Dropout(0.3)(x)  # 从0.5减少到0.3
    x = Dense(128, activation='relu', name='dense2')(x)  # 从256减少到128
    x = Dropout(0.2)(x)
    output = Dense(classes, activation='softmax', name='classification_output')(x)
    
    # 构建模型
    model = Model(inputs=input_iq, outputs=output, name='LightweightMambaModel')
    
    # 加载权重
    if weights is not None:
        model.load_weights(weights)
    
    return model

def compare_models():
    """比较原始模型和优化模型"""
    print("=== 模型复杂度对比 ===\n")
    
    # 原始模型
    from rmlmodels.MambaAttentionModel import MambaAttentionModel
    original_model = MambaAttentionModel(
        input_shape=[2, 128],
        classes=11,
        d_model=256,
        num_mamba_layers=4,
        num_attention_layers=2
    )
    
    # 优化模型
    optimized_model = LightweightMambaModel(
        input_shape=[2, 128],
        classes=11,
        d_model=128,
        num_mamba_layers=2,
        num_attention_layers=1
    )
    
    print("原始模型:")
    print(f"  总参数量: {original_model.count_params():,}")
    print(f"  层数: {len(original_model.layers)}")
    
    print("\n优化模型:")
    print(f"  总参数量: {optimized_model.count_params():,}")
    print(f"  层数: {len(optimized_model.layers)}")
    
    reduction = (original_model.count_params() - optimized_model.count_params()) / original_model.count_params() * 100
    print(f"\n优化效果:")
    print(f"  参数量减少: {reduction:.1f}%")
    # print(f"  预计训练时间减少: ~65-70%")
    
    return optimized_model

if __name__ == '__main__':
    # 测试优化模型
    model = LightweightMambaModel(
        input_shape=[2, 128],
        classes=11,
        d_model=128,
        num_mamba_layers=2,
        num_attention_layers=1
    )
    
    # 编译模型
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=optimizer
    )
    
    print('优化后模型层数:', len(model.layers))
    print('优化后模型参数总数:', model.count_params())
    model.summary()
    
    # 对比模型
    compare_models()