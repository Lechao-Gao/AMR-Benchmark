#!/usr/bin/env python3
"""
轻量级Mamba模型训练脚本
解决模型构建卡住的问题，使用更小的模型规模
"""

import os
import sys
import time

# # 修复RTX 4090的ptxas兼容性问题
# from fix_ptxas import suppress_ptxas_warnings, check_system_compatibility, setup_workaround_for_rtx4090

# # 在导入任何深度学习库之前设置环境变量
# print("正在修复RTX 4090的ptxas兼容性问题...")
# suppress_ptxas_warnings()
# check_system_compatibility()
# setup_workaround_for_rtx4090()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import pickle
import csv

from common import mltools, rmldataset2016
import rmlmodels.MambaAttentionModel as mamba_attention
try:
    from tensorflow import keras
except ImportError:
    import keras

print("=== 轻量级Mamba双注意力调制信号识别模型 ===")
print("优化配置：减小模型规模以避免内存问题")

# 加载数据
print("正在加载RML2016.10a数据集...")
start_time = time.time()
(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
    rmldataset2016.load_data()

# 数据预处理
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)

print(f"训练数据形状: {X_train.shape}")
print(f"验证数据形状: {X_val.shape}")
print(f"测试数据形状: {X_test.shape}")

classes = mods
print(f"调制类型: {classes}")
print(f"信噪比范围: {snrs}")

# 轻量级配置
print("\n=== 使用轻量级配置 ===")
print("原始配置: d_model=256, 4层Mamba, 2层注意力")
print("优化配置: d_model=128, 2层Mamba, 1层注意力")

# 设置训练参数
nb_epoch = 100
batch_size = 128  # 减小批次大小
learning_rate = 0.001

print(f"\n训练参数配置:")
print(f"- 训练轮数: {nb_epoch}")
print(f"- 批次大小: {batch_size}")
print(f"- 初始学习率: {learning_rate}")

# 创建轻量级模型
print("\n正在构建轻量级Mamba模型...")
model_start_time = time.time()

try:
    model = mamba_attention.MambaAttentionModel(
        input_shape=[2, 128],
        classes=11,
        d_model=128,  # 减小模型维度
        num_mamba_layers=2,  # 减少层数
        num_attention_layers=1  # 减少注意力层
    )
    
    model_end_time = time.time()
    print(f"✅ 模型构建成功！耗时: {model_end_time - model_start_time:.2f}秒")
    
    print("\n模型结构概览:")
    model.summary()
    print(f"模型总参数量: {model.count_params():,}")
    
except Exception as e:
    print(f"❌ 模型构建失败: {e}")
    print("尝试进一步减小模型规模...")
    
    # 使用更小的配置
    print("\n尝试超轻量级配置...")
    model = mamba_attention.MambaAttentionModel(
        input_shape=[2, 128],
        classes=11,
        d_model=64,  # 进一步减小
        num_mamba_layers=1,  # 单层
        num_attention_layers=1
    )
    
    model.summary()
    print(f"超轻量级模型总参数量: {model.count_params():,}")

# 编译模型
print("\n正在编译模型...")
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=optimizer
)

# 设置权重保存路径
filepath = 'weights/MambaAttention_lightweight_best.h5'
os.makedirs('weights', exist_ok=True)
os.makedirs('figure', exist_ok=True)  # 确保figure目录存在

# 定义回调函数
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=5,  # 减小耐心值
        min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # 减小耐心值
        verbose=1,
        mode='auto',
        restore_best_weights=True
    ),
    keras.callbacks.CSVLogger(
        'training_log_lightweight.csv',
        append=True
    )
]

print("\n开始训练...")
# print("注意: 使用轻量级模型，训练速度会更快")

# 训练模型
history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=callbacks
)

# 显示训练历史
print("\n正在保存训练历史图表...")
mltools.show_history(history, save_prefix='lightweight_')

# 评估模型
print("\n正在评估模型性能...")
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(f"测试损失: {score[0]:.4f}")
print(f"测试准确率: {score[1]:.4f}")

# 预测和准确率分析
print("\n正在加载最佳权重进行预测...")
if os.path.exists(filepath):
    model.load_weights(filepath)
    print("✓ 权重加载成功")
else:
    print("⚠ 未找到权重文件，使用当前模型权重")

# 确保 predictresult 目录存在
os.makedirs('predictresult', exist_ok=True)

# 按信噪比分析性能
print("正在分析各信噪比下的性能...")
acc = {}
acc_mod_snr = np.zeros((len(classes), len(snrs)))
test_SNRs = [lbl[x][1] for x in test_idx]

for i, snr in enumerate(snrs):
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
    
    if len(test_X_i) == 0:
        continue
    
    # 预测
    test_Y_i_hat = model.predict(test_X_i, batch_size=batch_size)
    confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
    
    acc[snr] = 1.0 * cor / (cor + ncor)
    result = cor / (cor + ncor)
    
    # 保存准确率结果
    with open('accuray_res.csv', 'a', newline='') as f0:
        write0 = csv.writer(f0)
        write0.writerow([result])
    
    acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)

# 保存各调制类型准确率数据
fd = open('predictresult/acc_for_mod_on_mamba_attention_lightweight.dat', 'wb')
pickle.dump(acc_mod_snr, fd)
fd.close()

# 保存整体准确率数据
fd = open('predictresult/MambaAttention_lightweight_results.dat', 'wb')
pickle.dump(acc, fd)
fd.close()

# 保存总体准确率（ResNet格式）
print(acc)
fd = open('predictresult/mamba_attention_lightweight.dat', 'wb')
pickle.dump(("mamba_attention_lightweight", 0.5, acc), fd)
fd.close()

# 生成t-SNE可视化图
print("\n" + "="*60)
print("开始生成t-SNE可视化图...")
print("="*60)

# 获取测试数据的SNR信息
test_SNRs = [lbl[x][1] for x in test_idx]

# 选择几个代表性的SNR进行可视化（可以选择全部，但会比较慢）
# 这里选择低、中、高SNR各几个
selected_snrs_for_tsne = [-20, -10, 0, 10, 18]  # 可以根据需要调整
# 如果某些SNR不存在，自动过滤
selected_snrs_for_tsne = [s for s in selected_snrs_for_tsne if s in snrs]

# 如果选择的SNR太少，添加更多
if len(selected_snrs_for_tsne) < 3:
    # 添加更多SNR
    additional_snrs = [s for s in snrs if s not in selected_snrs_for_tsne]
    selected_snrs_for_tsne.extend(additional_snrs[:5-len(selected_snrs_for_tsne)])

try:
    # 确保模型已加载最佳权重
    if os.path.exists(filepath):
        print(f"加载最佳权重: {filepath}")
        model.load_weights(filepath)
    
    mltools.plot_tsne_visualization(
        model=model,
        X_data=X_test,
        Y_data=Y_test,
        snr_data=test_SNRs,
        classes=classes,
        snrs=snrs,
        feature_layer_name='dense3',  # MambaAttention的特征层名称（128维全连接层）
        save_filename='figure/tsne_mamba_attention',
        selected_snrs=selected_snrs_for_tsne,
        n_samples_per_snr=800,  # 每个SNR使用800个样本（可以调整）
        perplexity=30,
        n_iter=1000
    )
    print("t-SNE可视化完成！")
except Exception as e:
    print(f"生成t-SNE可视化时出错: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 训练完成 ===")