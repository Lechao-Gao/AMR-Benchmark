import os,random
import sys

# 添加项目根目录到Python路径，以便导入common模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless safe backend
import matplotlib.pyplot as plt
import matplotlib_chinese_init
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle, random, sys, h5py
import tensorflow as tf

try:
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
    from tensorflow.keras.regularizers import *
    from tensorflow.keras.optimizers import Adam, AdamW
    from tensorflow.keras.models import model_from_json
except ImportError:
    import keras
    from keras import backend as K
    from keras.callbacks import LearningRateScheduler, TensorBoard
    from keras.regularizers import *
    from keras.optimizers import Adam, AdamW
    from keras.models import model_from_json

import csv
from common import mltools, rmldataset2016
import rmlmodels.MambaAttentionModel as mamba_attention

print("=== Mamba双注意力调制信号识别模型 - 优化版本 ===")
print("模型架构: 多特征输入 -> 小核卷积 -> 特征融合 -> Mamba主干[嵌入注意力块] -> 分类输出")

# 加载数据
print("正在加载RML2016.10a数据集...")
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

# === 数据增强函数 ===
def add_noise_augmentation(X, noise_factor=0.01):
    """添加噪声增强"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

def time_shift_augmentation(X, shift_range=5):
    """时间平移增强"""
    X_aug = np.copy(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift != 0:
            X_aug[i] = np.roll(X[i], shift, axis=1)
    return X_aug

def amplitude_scaling_augmentation(X, scale_range=(0.8, 1.2)):
    """幅度缩放增强"""
    X_aug = np.copy(X)
    for i in range(X.shape[0]):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        X_aug[i] = X[i] * scale
    return X_aug

# 应用数据增强
print("正在应用数据增强...")
X_train_aug1 = add_noise_augmentation(X_train, noise_factor=0.005)
X_train_aug2 = time_shift_augmentation(X_train, shift_range=3)
X_train_aug3 = amplitude_scaling_augmentation(X_train, scale_range=(0.9, 1.1))

# 合并增强数据
X_train_combined = np.concatenate([X_train, X_train_aug1, X_train_aug2, X_train_aug3], axis=0)
Y_train_combined = np.concatenate([Y_train, Y_train, Y_train, Y_train], axis=0)

print(f"增强后训练数据形状: {X_train_combined.shape}")

# 设置训练参数 - 优化版本
nb_epoch = 200     # 增加训练轮数
batch_size = 128   # 进一步减小批次大小
learning_rate = 0.001  # 提高初始学习率

print(f"\n训练参数配置:")
print(f"- 训练轮数: {nb_epoch}")
print(f"- 批次大小: {batch_size}")
print(f"- 初始学习率: {learning_rate}")
print(f"- 数据增强倍数: 4x")

# 创建模型
print("\n正在构建Mamba双注意力模型...")
model = mamba_attention.MambaAttentionModel(
    input_shape=[2, 128],
    classes=11,
    d_model=256,
    num_mamba_layers=6,  # 增加Mamba层数
    num_attention_layers=3  # 增加注意力层数
)

# 使用AdamW优化器
optimizer = AdamW(
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    weight_decay=0.01  # 添加权重衰减
)

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=optimizer
)

print("\n模型结构概览:")
model.summary()
print(f"模型总参数量: {model.count_params():,}")

# 设置权重保存路径
filepath = 'weights/MambaAttention_optimized_best.h5'
os.makedirs('weights', exist_ok=True)

# 自定义学习率调度器
def cosine_annealing_with_warmup(epoch, lr):
    """余弦退火学习率调度器，带预热"""
    warmup_epochs = 10
    total_epochs = nb_epoch
    
    if epoch < warmup_epochs:
        # 预热阶段
        return learning_rate * (epoch + 1) / warmup_epochs
    else:
        # 余弦退火阶段
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return learning_rate * 0.5 * (1 + np.cos(np.pi * progress))

# 自定义回调函数 - 更激进的训练策略
class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0
        self.patience_counter = 0
        self.max_patience = 30
        
    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy', 0)
        
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.patience_counter = 0
            print(f"\n*** 新的最佳验证准确率: {current_val_acc:.4f} ***")
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.max_patience:
            print(f"\n*** 验证准确率在{self.max_patience}轮内未改善，但继续训练 ***")
            # 不停止训练，只是记录

# 定义回调函数 - 优化版本
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
        factor=0.8,  # 更温和的学习率衰减
        verbose=1,
        patience=20,  # 增加耐心值
        min_lr=1e-9,  # 更低的最小学习率
        cooldown=10   # 增加冷却期
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=100,  # 大幅增加耐心值
        verbose=1,
        mode='auto',
        restore_best_weights=True,
        min_delta=0.0005  # 减小最小改进阈值
    ),
    keras.callbacks.CSVLogger(
        'training_log_optimized.csv',
        append=True
    ),
    keras.callbacks.LearningRateScheduler(
        cosine_annealing_with_warmup,
        verbose=1
    ),
    CustomCallback(),
    # 添加TensorBoard监控
    keras.callbacks.TensorBoard(
        log_dir='./logs_optimized',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
]

print("\n开始训练...")
print("注意: 使用了数据增强和优化的训练策略")

# 训练模型
history = model.fit(
    X_train_combined,
    Y_train_combined,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=callbacks,
    shuffle=True  # 确保每个epoch都打乱数据
)

# 显示训练历史
print("\n正在保存训练历史图表...")
mltools.show_history(history)

# 评估模型
print("\n正在评估模型性能...")
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(f"测试损失: {score[0]:.4f}")
print(f"测试准确率: {score[1]:.4f}")

def generate_tsne_visualization(model, X_test, Y_test, lbl, snrs, test_idx, filepath, classes,
                                feature_layer='dense3', save_name='figure/tsne_mamba_attention_optimized'):
    """
    复用ResNet脚本的t-SNE流程，为优化版Mamba模型生成特征可视化。
    """
    print("\n" + "="*60)
    print("开始生成t-SNE可视化图...")
    print("="*60)

    test_SNRs = [lbl[x][1] for x in test_idx]

    selected_snrs_for_tsne = [-20, -10, 0, 10, 18]
    selected_snrs_for_tsne = [s for s in selected_snrs_for_tsne if s in snrs]

    if len(selected_snrs_for_tsne) < 3:
        additional_snrs = [s for s in snrs if s not in selected_snrs_for_tsne]
        selected_snrs_for_tsne.extend(additional_snrs[:5 - len(selected_snrs_for_tsne)])

    try:
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
            feature_layer_name=feature_layer,
            save_filename=save_name,
            selected_snrs=selected_snrs_for_tsne,
            n_samples_per_snr=800,
            perplexity=30,
            n_iter=1000
        )
        print("t-SNE可视化完成！")
    except Exception as e:
        print(f"生成t-SNE可视化时出错: {e}")
        import traceback
        traceback.print_exc()

def predict_and_analyze_optimized(model):
    """
    优化的模型预测和性能分析
    """
    print("\n正在加载最佳权重进行预测...")
    model.load_weights(filepath)

    # 整体混淆矩阵
    print("正在生成整体混淆矩阵...")
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(
        confnorm,
        labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
        save_filename='figure/mamba_attention_optimized_total_confusion'
    )

    # 按信噪比分析性能
    print("正在分析各信噪比下的性能...")
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))

    for i, snr in enumerate(snrs):
        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # 预测
        test_Y_i_hat = model.predict(test_X_i, batch_size=batch_size)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)

        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)

        # 保存准确率结果
        with open('accuray_res_optimized.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])

        # 生成混淆矩阵图
        mltools.plot_confusion_matrix(
            confnorm_i,
            labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
            title="Mamba双注意力模型混淆矩阵(优化版)",
            save_filename=f"figure/MambaAttention_Optimized_Confusion(SNR={snr})(ACC={100.0*acc[snr]:.2f}).png"
        )

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)

    # 保存各调制类型准确率数据
    fd = open('predictresult/acc_for_mod_on_mamba_attention_optimized.dat', 'wb')
    pickle.dump(acc_mod_snr, fd)
    fd.close()

    # 保存整体准确率数据
    print("保存预测结果...")
    print(f"各信噪比准确率: {acc}")
    fd = open('predictresult/MambaAttention_optimized_results.dat', 'wb')
    pickle.dump(acc, fd)
    fd.close()

    # 保存总体准确率（ResNet格式）
    print(acc)
    fd = open('predictresult/mamba_attention_optimized_d0.5.dat', 'wb')
    pickle.dump(("mamba_attention_optimized", 0.5, acc), fd)
    fd.close()

    # 绘制整体准确率曲线
    plt.figure(figsize=(12, 8))
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)), 'b-o', linewidth=3, markersize=8, label='优化版本')
    plt.xlabel("信噪比 (dB)", fontsize=14)
    plt.ylabel("分类准确率", fontsize=14)
    plt.title("Mamba双注意力模型 - 整体分类准确率曲线(优化版)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 添加数值标签
    for x, y in zip(snrs, [acc[x] for x in snrs]):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('figure/mamba_attention_optimized_overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 计算平均准确率
    avg_acc = np.mean(list(acc.values()))
    print(f"\n=== 优化模型性能总结 ===")
    print(f"平均准确率: {avg_acc:.4f}")
    print(f"最高准确率: {max(acc.values()):.4f} (SNR = {max(acc, key=acc.get)} dB)")
    print(f"最低准确率: {min(acc.values()):.4f} (SNR = {min(acc, key=acc.get)} dB)")

    return acc, acc_mod_snr

# 执行预测和分析
print("\n开始模型预测和性能分析...")
acc_results, acc_mod_snr_results = predict_and_analyze_optimized(model)

# 生成t-SNE可视化
generate_tsne_visualization(
    model=model,
    X_test=X_test,
    Y_test=Y_test,
    lbl=lbl,
    snrs=snrs,
    test_idx=test_idx,
    filepath=filepath,
    classes=classes,
    feature_layer='dense3',
    save_name='figure/tsne_mamba_attention_optimized'
)

print("\n=== 优化训练完成 ===")
print("结果文件已保存到:")
print("- 权重文件: weights/MambaAttention_optimized_best.h5")
print("- 训练日志: training_log_optimized.csv")
print("- 准确率数据: accuray_res_optimized.csv")
print("- 混淆矩阵图: figure/目录")
print("- 预测结果: predictresult/目录")
print("- TensorBoard日志: ./logs_optimized/")

# 保存模型架构
model_json = model.to_json()
with open('model_architecture_optimized.json', 'w') as json_file:
    json_file.write(model_json)

print("\n优化要点:")
print("1. 增加了数据增强(4x数据)")
print("2. 使用AdamW优化器with权重衰减")
print("3. 余弦退火学习率调度")
print("4. 增强的模型架构")
print("5. 更长的训练时间和更大的耐心值")
print("6. 多尺度池化和渐进式分类层")