#!/usr/bin/env python3
"""
增强版Mamba模型 - 专门优化QAM-16和WB-FM识别
针对性解决这两种调制类型的识别率低问题
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
                                         BatchNormalization, Lambda, Multiply, Average)
    from tensorflow.keras.activations import swish
except ImportError:
    import keras
    from keras import backend as K
    from keras.models import Model
    from keras.layers import (Input, Dense, Conv1D, Conv2D, MaxPool1D, MaxPool2D,
                             ReLU, Dropout, Softmax, concatenate, LSTM, GRU,
                             Permute, Reshape, ZeroPadding2D, Activation,
                             Add, GlobalAveragePooling1D, GlobalAveragePooling2D,
                             BatchNormalization, Lambda, Multiply, Average)
    try:
        from keras.layers import LayerNormalization, MultiHeadAttention
    except ImportError:
        from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
    try:
        from keras.activations import swish
    except ImportError:
        from tensorflow.keras.activations import swish

class QAMSpecificFeatureExtractor(tf.keras.layers.Layer):
    """QAM信号专用特征提取器"""
    def __init__(self, filters=64, **kwargs):
        super(QAMSpecificFeatureExtractor, self).__init__(**kwargs)
        self.filters = filters
        
        # QAM信号的星座图特征提取
        self.magnitude_conv = Conv1D(filters//2, 3, padding='same', activation='relu', name='qam_mag_conv')
        self.phase_conv = Conv1D(filters//2, 3, padding='same', activation='relu', name='qam_phase_conv')
        
        # IQ不平衡检测
        self.iq_imbalance_conv = Conv1D(filters//4, 5, padding='same', activation='relu', name='iq_imbalance')
        
        # 符号速率特征
        self.symbol_rate_conv = Conv1D(filters//4, 7, padding='same', activation='relu', name='symbol_rate')
        
        self.norm = BatchNormalization()
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        # 假设输入是 (batch, length, 2) - I/Q信号
        i_signal = x[:, :, 0:1]  # I分量
        q_signal = x[:, :, 1:2]  # Q分量
        
        # 计算幅度和相位
        magnitude = tf.sqrt(tf.square(i_signal) + tf.square(q_signal))
        phase = tf.atan2(q_signal, i_signal)
        
        # 特征提取
        mag_features = self.magnitude_conv(magnitude, training=training)
        phase_features = self.phase_conv(phase, training=training)
        
        # IQ不平衡特征
        iq_diff = tf.abs(i_signal - q_signal)
        iq_features = self.iq_imbalance_conv(iq_diff, training=training)
        
        # 符号速率特征（通过差分检测）
        i_diff = tf.abs(i_signal[:, 1:, :] - i_signal[:, :-1, :])
        i_diff = tf.pad(i_diff, [[0, 0], [1, 0], [0, 0]], mode='CONSTANT')
        symbol_features = self.symbol_rate_conv(i_diff, training=training)
        
        # 合并所有特征
        combined = tf.concat([mag_features, phase_features, iq_features, symbol_features], axis=-1)
        combined = self.norm(combined, training=training)
        combined = self.dropout(combined, training=training)
        
        return combined
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

class FMSpecificFeatureExtractor(tf.keras.layers.Layer):
    """FM信号专用特征提取器"""
    def __init__(self, filters=64, **kwargs):
        super(FMSpecificFeatureExtractor, self).__init__(**kwargs)
        self.filters = filters
        
        # FM信号的频率偏移特征
        self.freq_dev_conv = Conv1D(filters//2, 3, padding='same', activation='relu', name='fm_freq_dev')
        
        # 瞬时频率特征
        self.inst_freq_conv = Conv1D(filters//4, 5, padding='same', activation='relu', name='inst_freq')
        
        # 频率调制深度
        self.mod_depth_conv = Conv1D(filters//4, 7, padding='same', activation='relu', name='mod_depth')
        
        self.norm = BatchNormalization()
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        # 假设输入是 (batch, length, 2) - I/Q信号
        i_signal = x[:, :, 0:1]  # I分量
        q_signal = x[:, :, 1:2]  # Q分量
        
        # 计算瞬时相位
        phase = tf.atan2(q_signal, i_signal)
        
        # 计算瞬时频率（相位的导数）
        phase_diff = phase[:, 1:, :] - phase[:, :-1, :]
        # 处理相位跳跃
        phase_diff = tf.where(phase_diff > np.pi, phase_diff - 2*np.pi, phase_diff)
        phase_diff = tf.where(phase_diff < -np.pi, phase_diff + 2*np.pi, phase_diff)
        phase_diff = tf.pad(phase_diff, [[0, 0], [1, 0], [0, 0]], mode='CONSTANT')
        
        # 频率偏移特征
        freq_features = self.freq_dev_conv(phase_diff, training=training)
        
        # 瞬时频率特征
        inst_freq_features = self.inst_freq_conv(phase_diff, training=training)
        
        # 调制深度特征（频率变化的幅度）
        freq_variance = tf.abs(phase_diff)
        mod_depth_features = self.mod_depth_conv(freq_variance, training=training)
        
        # 合并特征
        combined = tf.concat([freq_features, inst_freq_features, mod_depth_features], axis=-1)
        combined = self.norm(combined, training=training)
        combined = self.dropout(combined, training=training)
        
        return combined
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config

class AdaptiveFeatureFusion(tf.keras.layers.Layer):
    """自适应特征融合层"""
    def __init__(self, d_model, **kwargs):
        super(AdaptiveFeatureFusion, self).__init__(**kwargs)
        self.d_model = d_model
        
        # 注意力权重计算
        self.attention_weights = Dense(3, activation='softmax', name='fusion_weights')
        self.feature_projection = Dense(d_model, activation='relu', name='feature_proj')
        self.norm = LayerNormalization()
        
    def call(self, inputs, training=None):
        general_features, qam_features, fm_features = inputs
        
        # 计算全局特征用于权重计算
        global_feature = tf.reduce_mean(general_features, axis=1)  # (batch, d_model)
        weights = self.attention_weights(global_feature)  # (batch, 3)
        
        # 扩展权重维度
        weights = tf.expand_dims(weights, axis=1)  # (batch, 1, 3)
        
        # 加权融合
        w1, w2, w3 = tf.split(weights, 3, axis=-1)
        fused = w1 * general_features + w2 * qam_features + w3 * fm_features
        
        # 投影到目标维度
        output = self.feature_projection(fused, training=training)
        output = self.norm(output, training=training)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class EnhancedMambaBlock(tf.keras.layers.Layer):
    """增强的Mamba块，专门优化序列建模"""
    def __init__(self, d_model, **kwargs):
        super(EnhancedMambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        
        # 双向处理
        self.forward_conv = Conv1D(d_model, 3, padding='same', activation='swish')
        self.backward_conv = Conv1D(d_model, 3, padding='same', activation='swish')
        
        # 门控机制
        self.gate_proj = Dense(d_model * 3, use_bias=True)
        self.out_proj = Dense(d_model, use_bias=True)
        
        # 归一化
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        self.dropout = Dropout(0.1)
        
    def call(self, x, training=None):
        residual = x
        x = self.norm1(x, training=training)
        
        # 双向卷积
        forward = self.forward_conv(x, training=training)
        backward = self.backward_conv(tf.reverse(x, axis=[1]), training=training)
        backward = tf.reverse(backward, axis=[1])
        
        # 门控融合
        gates = self.gate_proj(x)  # (batch, length, d_model * 3)
        gate1, gate2, gate3 = tf.split(gates, 3, axis=-1)
        
        # 应用门控
        gated_forward = forward * tf.nn.sigmoid(gate1)
        gated_backward = backward * tf.nn.sigmoid(gate2)
        fusion_gate = tf.nn.sigmoid(gate3)
        
        # 融合双向信息
        combined = fusion_gate * gated_forward + (1 - fusion_gate) * gated_backward
        combined = self.dropout(combined, training=training)
        
        # 输出投影
        output = self.out_proj(combined)
        output = self.norm2(output, training=training)
        
        return output + residual
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class MultiScaleAttention(tf.keras.layers.Layer):
    """多尺度注意力机制"""
    def __init__(self, d_model, num_heads=8, **kwargs):
        super(MultiScaleAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 不同尺度的注意力
        self.local_attention = MultiHeadAttention(
            num_heads=num_heads//2, key_dim=d_model//num_heads, dropout=0.1
        )
        self.global_attention = MultiHeadAttention(
            num_heads=num_heads//2, key_dim=d_model//num_heads, dropout=0.1
        )
        
        # 尺度融合
        self.scale_fusion = Dense(d_model, activation='relu')
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        # 前馈网络
        self.ffn = tf.keras.Sequential([
            Dense(d_model * 2, activation='relu'),
            Dropout(0.1),
            Dense(d_model)
        ])
        
    def call(self, x, training=None):
        # 局部注意力（短序列）
        local_out = self.local_attention(x, x, training=training)
        
        # 全局注意力（降采样后的长序列）
        # 简单的降采样策略
        seq_len = tf.shape(x)[1]
        downsampled = x[:, ::2, :]  # 每隔一个取样
        global_out = self.global_attention(downsampled, downsampled, training=training)
        # 上采样回原始长度
        global_out = tf.repeat(global_out, 2, axis=1)[:, :seq_len, :]
        
        # 融合多尺度特征
        fused = self.scale_fusion(tf.concat([local_out, global_out], axis=-1))
        x = self.norm1(x + fused)
        
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

def EnhancedMambaModel(weights=None,
                      input_shape=[2, 128],
                      classes=11,
                      d_model=256,
                      num_mamba_layers=3,
                      num_attention_layers=2,
                      **kwargs):
    """
    增强版Mamba模型 - 专门优化QAM-16和WB-FM识别
    
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
    
    # 输入处理
    input_iq = Input(shape=input_shape + [1], name='input_iq')
    x = Reshape((input_shape[1], input_shape[0]))(input_iq)  # (batch, 128, 2)
    
    # === 多路径特征提取 ===
    # 通用特征提取
    general_features = Conv1D(64, 5, padding='same', activation='relu', name='general_conv1')(x)
    general_features = BatchNormalization()(general_features)
    general_features = Conv1D(128, 3, padding='same', activation='relu', name='general_conv2')(general_features)
    general_features = BatchNormalization()(general_features)
    general_features = Dense(d_model, activation='relu', name='general_projection')(general_features)
    
    # QAM专用特征提取
    qam_extractor = QAMSpecificFeatureExtractor(filters=128, name='qam_extractor')
    qam_features = qam_extractor(x)
    qam_features = Dense(d_model, activation='relu', name='qam_projection')(qam_features)
    
    # FM专用特征提取
    fm_extractor = FMSpecificFeatureExtractor(filters=128, name='fm_extractor')
    fm_features = fm_extractor(x)
    fm_features = Dense(d_model, activation='relu', name='fm_projection')(fm_features)
    
    # === 自适应特征融合 ===
    fusion_layer = AdaptiveFeatureFusion(d_model, name='adaptive_fusion')
    x = fusion_layer([general_features, qam_features, fm_features])
    
    # === 增强Mamba主干网络 ===
    for i in range(num_mamba_layers):
        x = EnhancedMambaBlock(d_model, name=f'enhanced_mamba_{i}')(x)
        
        # 在中间层添加多尺度注意力
        if i == num_mamba_layers // 2:
            x = MultiScaleAttention(d_model, name=f'multiscale_attention_{i}')(x)
    
    # === 最终注意力层 ===
    for i in range(num_attention_layers):
        x = MultiScaleAttention(d_model, name=f'final_attention_{i}')(x)
    
    # === 分类头部 ===
    # 全局特征聚合
    global_avg = GlobalAveragePooling1D()(x)
    global_max = tf.reduce_max(x, axis=1)
    
    # 特征融合
    combined_features = tf.concat([global_avg, global_max], axis=-1)
    
    # 分类器
    x = Dense(512, activation='relu', name='classifier_dense1')(combined_features)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', name='classifier_dense2')(x)
    x = Dropout(0.3)(x)
    
    # 标准分类输出
    output = Dense(classes, activation='softmax', name='classification_output')(x)
    
    # 构建模型
    model = Model(inputs=input_iq, outputs=output, name='EnhancedMambaModel')
    
    # 加载权重
    if weights is not None:
        model.load_weights(weights)
    
    return model

def create_enhanced_loss_function():
    """创建增强的损失函数，对QAM-16和WB-FM给予更高权重"""
    def enhanced_categorical_crossentropy(y_true, y_pred):
        # 基础交叉熵损失
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 为QAM-16 (索引7) 和 WB-FM (索引10) 增加权重
        qam_weight = 2.0
        fm_weight = 2.0
        
        # 计算样本权重
        qam_mask = y_true[:, 7]  # QAM-16
        fm_mask = y_true[:, 10]  # WB-FM
        
        sample_weights = 1.0 + (qam_weight - 1.0) * qam_mask + (fm_weight - 1.0) * fm_mask
        
        # 应用权重
        weighted_loss = base_loss * sample_weights
        
        return weighted_loss
    
    return enhanced_categorical_crossentropy

if __name__ == "__main__":
    # 测试模型构建
    model = EnhancedMambaModel()
    print("增强版Mamba模型构建成功！")
    model.summary()
    print(f"模型参数量: {model.count_params():,}")