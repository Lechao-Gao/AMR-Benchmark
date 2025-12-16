#!/usr/bin/env python3
"""
紧凑型增强Mamba模型 - 专门优化QAM-16和WB-FM识别（轻量级版本）
大幅减少参数量，提高训练效率，同时保持针对性优化效果
"""

import os
import numpy as np
import tensorflow as tf

try:
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout,
                                         Softmax, Reshape, Activation, LayerNormalization,
                                         MultiHeadAttention, GlobalAveragePooling1D,
                                         BatchNormalization, Lambda, Multiply, Average)
    from tensorflow.keras.activations import swish
except ImportError:
    import keras
    from keras import backend as K
    from keras.models import Model
    from keras.layers import (Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout,
                             Softmax, Reshape, Activation,
                             MultiHeadAttention, GlobalAveragePooling1D,
                             BatchNormalization, Lambda, Multiply, Average)
    try:
        from keras.layers import LayerNormalization, MultiHeadAttention
    except ImportError:
        from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
    try:
        from keras.activations import swish
    except ImportError:
        from tensorflow.keras.activations import swish

class CompactQAMFeatureExtractor(tf.keras.layers.Layer):
    """紧凑型QAM信号特征提取器"""
    def __init__(self, filters=32, **kwargs):
        super(CompactQAMFeatureExtractor, self).__init__(**kwargs)
        self.filters = filters
        
        # 减少卷积层数量和通道数
        self.magnitude_conv = Conv1D(filters//2, 3, padding='same', activation='relu')
        self.phase_conv = Conv1D(filters//2, 3, padding='same', activation='relu')
        
        self.norm = BatchNormalization()
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        # 输入: (batch, length, 2) - I/Q信号
        i_signal = x[:, :, 0:1]
        q_signal = x[:, :, 1:2]
        
        # 计算幅度和相位
        magnitude = tf.sqrt(tf.square(i_signal) + tf.square(q_signal))
        phase = tf.atan2(q_signal, i_signal)
        
        # 简化特征提取
        mag_features = self.magnitude_conv(magnitude, training=training)
        phase_features = self.phase_conv(phase, training=training)
        
        # 合并特征
        combined = tf.concat([mag_features, phase_features], axis=-1)
        combined = self.norm(combined, training=training)
        combined = self.dropout(combined, training=training)
        
        return combined
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

class CompactFMFeatureExtractor(tf.keras.layers.Layer):
    """紧凑型FM信号特征提取器"""
    def __init__(self, filters=32, **kwargs):
        super(CompactFMFeatureExtractor, self).__init__(**kwargs)
        self.filters = filters
        
        # 简化的频率特征提取
        self.freq_conv = Conv1D(filters, 3, padding='same', activation='relu')
        self.norm = BatchNormalization()
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        # 输入: (batch, length, 2) - I/Q信号
        i_signal = x[:, :, 0:1]
        q_signal = x[:, :, 1:2]
        
        # 计算瞬时相位
        phase = tf.atan2(q_signal, i_signal)
        
        # 计算瞬时频率（相位的导数）
        phase_diff = phase[:, 1:, :] - phase[:, :-1, :]
        # 处理相位跳跃
        phase_diff = tf.where(phase_diff > np.pi, phase_diff - 2*np.pi, phase_diff)
        phase_diff = tf.where(phase_diff < -np.pi, phase_diff + 2*np.pi, phase_diff)
        phase_diff = tf.pad(phase_diff, [[0, 0], [1, 0], [0, 0]], mode='CONSTANT')
        
        # 频率特征提取
        freq_features = self.freq_conv(phase_diff, training=training)
        freq_features = self.norm(freq_features, training=training)
        freq_features = self.dropout(freq_features, training=training)
        
        return freq_features
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

class LightweightFeatureFusion(tf.keras.layers.Layer):
    """轻量级特征融合层"""
    def __init__(self, d_model, **kwargs):
        super(LightweightFeatureFusion, self).__init__(**kwargs)
        self.d_model = d_model
        
        # 简化的融合机制
        self.fusion_dense = Dense(d_model, activation='relu')
        self.norm = LayerNormalization()
        
    def call(self, inputs, training=None):
        general_features, qam_features, fm_features = inputs
        
        # 简单的加权平均融合
        fused = (general_features + qam_features + fm_features) / 3.0
        
        # 投影到目标维度
        output = self.fusion_dense(fused, training=training)
        output = self.norm(output, training=training)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class CompactMambaBlock(tf.keras.layers.Layer):
    """紧凑型Mamba块"""
    def __init__(self, d_model, **kwargs):
        super(CompactMambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        
        # 简化的处理
        self.conv = Conv1D(d_model, 3, padding='same', activation='swish')
        self.gate_proj = Dense(d_model * 2, use_bias=True)
        self.out_proj = Dense(d_model, use_bias=True)
        
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        residual = x
        x = self.norm1(x, training=training)
        
        # 卷积处理
        x = self.conv(x, training=training)
        
        # 门控机制
        gates = self.gate_proj(x)
        gate1, gate2 = tf.split(gates, 2, axis=-1)
        
        # 应用门控
        gated = x * tf.nn.sigmoid(gate1) * tf.nn.sigmoid(gate2)
        gated = self.dropout(gated, training=training)
        
        # 输出投影
        output = self.out_proj(gated)
        output = self.norm2(output, training=training)
        
        return output + residual
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class EfficientAttention(tf.keras.layers.Layer):
    """高效注意力机制"""
    def __init__(self, d_model, num_heads=4, **kwargs):
        super(EfficientAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 减少注意力头数
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads, dropout=0.1
        )
        
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        # 轻量级前馈网络
        self.ffn = tf.keras.Sequential([
            Dense(d_model, activation='relu'),  # 不扩展维度
            Dropout(0.1),
            Dense(d_model)
        ])
        
    def call(self, x, training=None):
        # 自注意力
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + attn)
        
        # 前馈网络
        ffn_out = self.ffn(x, training=training)
        output = self.norm2(x + ffn_out)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config

def CompactEnhancedMambaModel(weights=None,
                             input_shape=[2, 128],
                             classes=11,
                             d_model=128,  # 减少到128
                             num_mamba_layers=2,  # 减少到2层
                             num_attention_layers=1,  # 减少到1层
                             **kwargs):
    """
    紧凑型增强Mamba模型 - 专门优化QAM-16和WB-FM识别（轻量级版本）
    
    参数:
        weights: 预训练权重路径
        input_shape: 输入形状 [I/Q通道数, 序列长度]
        classes: 分类数量
        d_model: 模型维度 (减少到128)
        num_mamba_layers: Mamba层数量 (减少到2层)
        num_attention_layers: 注意力层数量 (减少到1层)
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                        '`None` (random initialization), '
                        'or the path to the weights file to be loaded.')
    
    # 输入处理
    input_iq = Input(shape=input_shape + [1], name='input_iq')
    x = Reshape((input_shape[1], input_shape[0]))(input_iq)  # (batch, 128, 2)
    
    # === 轻量级多路径特征提取 ===
    # 通用特征提取（减少通道数）
    general_features = Conv1D(32, 5, padding='same', activation='relu', name='general_conv1')(x)
    general_features = BatchNormalization()(general_features)
    general_features = Conv1D(64, 3, padding='same', activation='relu', name='general_conv2')(general_features)
    general_features = BatchNormalization()(general_features)
    general_features = Dense(d_model, activation='relu', name='general_projection')(general_features)
    
    # QAM专用特征提取（减少参数）
    qam_extractor = CompactQAMFeatureExtractor(filters=64, name='qam_extractor')
    qam_features = qam_extractor(x)
    qam_features = Dense(d_model, activation='relu', name='qam_projection')(qam_features)
    
    # FM专用特征提取（减少参数）
    fm_extractor = CompactFMFeatureExtractor(filters=64, name='fm_extractor')
    fm_features = fm_extractor(x)
    fm_features = Dense(d_model, activation='relu', name='fm_projection')(fm_features)
    
    # === 轻量级特征融合 ===
    fusion_layer = LightweightFeatureFusion(d_model, name='lightweight_fusion')
    x = fusion_layer([general_features, qam_features, fm_features])
    
    # === 紧凑型Mamba主干网络 ===
    for i in range(num_mamba_layers):
        x = CompactMambaBlock(d_model, name=f'compact_mamba_{i}')(x)
    
    # === 高效注意力层 ===
    for i in range(num_attention_layers):
        x = EfficientAttention(d_model, num_heads=4, name=f'efficient_attention_{i}')(x)
    
    # === 轻量级分类头部 ===
    # 全局特征聚合
    global_avg = GlobalAveragePooling1D()(x)
    
    # 简化的分类器
    x = Dense(256, activation='relu', name='classifier_dense1')(global_avg)  # 减少到256
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='classifier_dense2')(x)  # 减少到128
    x = Dropout(0.2)(x)
    
    # 输出层
    output = Dense(classes, activation='softmax', name='classification_output')(x)
    
    # 构建模型
    model = Model(inputs=input_iq, outputs=output, name='CompactEnhancedMambaModel')
    
    # 加载权重
    if weights is not None:
        model.load_weights(weights)
    
    return model

def create_compact_enhanced_loss_function():
    """创建紧凑型增强损失函数"""
    def compact_enhanced_categorical_crossentropy(y_true, y_pred):
        # 基础交叉熵损失
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 为QAM-16 (索引7) 和 WB-FM (索引10) 增加权重（减少权重以平衡训练）
        qam_weight = 2.0  # 从3.0减少到2.0
        fm_weight = 2.0   # 从3.0减少到2.0
        
        # 计算样本权重
        qam_mask = y_true[:, 7]  # QAM-16
        fm_mask = y_true[:, 10]  # WB-FM
        
        sample_weights = 1.0 + (qam_weight - 1.0) * qam_mask + (fm_weight - 1.0) * fm_mask
        
        # 应用权重
        weighted_loss = base_loss * sample_weights
        
        return weighted_loss
    
    return compact_enhanced_categorical_crossentropy

if __name__ == "__main__":
    # 测试模型构建
    model = CompactEnhancedMambaModel()
    print("紧凑型增强Mamba模型构建成功！")
    model.summary()
    print(f"模型参数量: {model.count_params():,}")
    
    # 与原始增强模型对比
    from EnhancedMambaModel import EnhancedMambaModel
    original_model = EnhancedMambaModel()
    original_params = original_model.count_params()
    compact_params = model.count_params()
    reduction = (original_params - compact_params) / original_params * 100
    
    print(f"\n参数量对比:")
    print(f"原始增强模型: {original_params:,}")
    print(f"紧凑型模型: {compact_params:,}")
    print(f"参数量减少: {reduction:.1f}%")