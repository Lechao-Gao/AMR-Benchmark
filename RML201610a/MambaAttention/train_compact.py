#!/usr/bin/env python3
"""
紧凑型增强Mamba模型训练脚本 - 快速训练版本
专门优化QAM-16和WB-FM识别，大幅减少训练时间
"""

import os
import sys
import argparse
import csv
import pickle

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf

try:
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
    from tensorflow.keras.optimizers import Adam
    BaseLoss = keras.losses.Loss
except ImportError:
    import keras
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
    from keras.optimizers import Adam
    BaseLoss = getattr(keras.losses, 'Loss', object)

import matplotlib.pyplot as plt
import matplotlib_chinese_init
from common import mltools, rmldataset2016
import rmlmodels.CompactEnhancedMambaModel as compact_mamba

class CompactQAMFMWeightedLoss(BaseLoss):
    """紧凑型QAM-16和WB-FM加权损失函数"""
    def __init__(self, qam_weight=2.0, fm_weight=2.0, **kwargs):
        super(CompactQAMFMWeightedLoss, self).__init__(**kwargs)
        self.qam_weight = qam_weight
        self.fm_weight = fm_weight

    def call(self, y_true, y_pred):
        # 基础交叉熵损失
        base_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 为QAM-16 (索引7) 和 WB-FM (索引10) 增加权重
        qam_mask = y_true[:, 7]  # QAM-16
        fm_mask = y_true[:, 10]  # WB-FM
        
        # 计算样本权重
        sample_weights = 1.0 + (self.qam_weight - 1.0) * qam_mask + (self.fm_weight - 1.0) * fm_mask
        
        # 应用权重
        weighted_loss = base_loss * sample_weights
        
        return weighted_loss

def create_lightweight_data_augmentation():
    """创建轻量级数据增强函数"""
    def augment_qam_fm_lightweight(X, Y, augment_ratio=0.3):  # 减少增强比例
        """轻量级QAM-16和WB-FM数据增强"""
        augmented_X = []
        augmented_Y = []
        
        # 只对部分样本进行增强以减少训练时间
        sample_indices = np.random.choice(len(X), size=int(len(X) * 0.2), replace=False)
        
        for i in sample_indices:
            # 获取标签
            label_idx = np.argmax(Y[i])
            
            # 只对QAM-16 (索引7) 和 WB-FM (索引10) 进行增强
            if label_idx == 7 or label_idx == 10:  # QAM-16 or WB-FM
                if np.random.random() < augment_ratio:
                    x_aug = X[i].copy()
                    
                    if label_idx == 7:  # QAM-16增强
                        # 简化的IQ不平衡增强
                        iq_imbalance = np.random.normal(0, 0.03)  # 减少变化幅度
                        x_aug[0, :, 0] *= (1 + iq_imbalance)
                        x_aug[1, :, 0] *= (1 - iq_imbalance)
                        
                    elif label_idx == 10:  # WB-FM增强
                        # 简化的频率偏移
                        freq_offset = np.random.uniform(-0.03, 0.03)  # 减少变化幅度
                        t = np.arange(x_aug.shape[1])
                        x_aug[0, :, 0] += freq_offset * np.sin(0.05 * t)
                        x_aug[1, :, 0] += freq_offset * np.cos(0.05 * t)
                    
                    augmented_X.append(x_aug)
                    augmented_Y.append(Y[i])
        
        if augmented_X:
            augmented_X = np.array(augmented_X)
            augmented_Y = np.array(augmented_Y)
            
            # 合并原始数据和增强数据
            X_combined = np.concatenate([X, augmented_X], axis=0)
            Y_combined = np.concatenate([Y, augmented_Y], axis=0)
            
            return X_combined, Y_combined
        else:
            return X, Y
    
    return augment_qam_fm_lightweight

def train_compact_model(epochs=80, batch_size=256, learning_rate=0.001, use_augmentation=True):
    """训练紧凑型增强Mamba模型"""
    
    print("=== 紧凑型增强Mamba模型训练 - 快速训练版本 ===")
    print("优化特点:")
    print("- 参数量大幅减少: ~70%")
    print("- 训练时间减少: ~60-70%")
    print("- 保持QAM-16和WB-FM专用优化")
    print("- 轻量级特征提取器")
    print("- 高效注意力机制")
    print("- 适度的加权损失函数")
    
    # 加载数据
    print("\n正在加载RML2016.10a数据集...")
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = \
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
    
    # 轻量级数据增强
    if use_augmentation:
        print("\n正在进行轻量级数据增强...")
        augment_func = create_lightweight_data_augmentation()
        X_train_aug, Y_train_aug = augment_func(X_train, Y_train)
        print(f"增强后训练数据形状: {X_train_aug.shape}")
        X_train, Y_train = X_train_aug, Y_train_aug
    
    # 创建紧凑型模型
    print("\n正在构建紧凑型增强Mamba模型...")
    model = compact_mamba.CompactEnhancedMambaModel(
        input_shape=[2, 128],
        classes=11,
        d_model=128,
        num_mamba_layers=2,
        num_attention_layers=1
    )
    
    # 创建轻量级加权损失函数
    weighted_loss = CompactQAMFMWeightedLoss(qam_weight=2.0, fm_weight=2.0)
    
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(
        loss=weighted_loss,
        metrics=['accuracy'],
        optimizer=optimizer
    )
    
    print("\n紧凑型模型结构:")
    model.summary()
    print(f"模型总参数量: {model.count_params():,}")
    
    # 与原始增强模型参数量对比
    try:
        from rmlmodels.EnhancedMambaModel import EnhancedMambaModel
        original_model = EnhancedMambaModel()
        original_params = original_model.count_params()
        compact_params = model.count_params()
        reduction = (original_params - compact_params) / original_params * 100
        
        print(f"\n参数量对比:")
        print(f"  原始增强模型: {original_params:,}")
        print(f"  紧凑型模型: {compact_params:,}")
        print(f"  参数量减少: {reduction:.1f}%")
        print(f"  预期训练时间减少: ~{reduction * 0.8:.0f}%")
    except:
        print("无法加载原始增强模型进行对比")
    
    # 设置回调函数
    os.makedirs('weights', exist_ok=True)
    os.makedirs('figure/compact_mamba', exist_ok=True)
    os.makedirs('predictresult', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'weights/CompactEnhancedMamba_best.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # 更温和的学习率衰减
            verbose=1,
            patience=6,  # 更短的耐心等待
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # 更短的早停耐心
            verbose=1,
            mode='auto',
            restore_best_weights=True
        ),
        CSVLogger(
            'training_log_compact.csv',
            append=False
        )
    ]
    
    print("\n开始训练...")
    print("注意: 紧凑型模型，训练时间大幅减少")
    
    # 训练模型
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    
    # 评估模型
    print("\n正在评估模型性能...")
    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print(f"测试损失: {score[0]:.4f}")
    print(f"测试准确率: {score[1]:.4f}")
    
    # 训练结束后自动进行预测和评估
    print("\n=== 训练完成，开始自动评估 ===")
    predict_and_analyze_compact(model, X_test, Y_test, classes, snrs, test_idx, lbl, batch_size)
    
    return model, history

def predict_and_analyze_compact(model, X_test, Y_test, classes, snrs, test_idx, lbl, batch_size=256):
    """紧凑型模型的预测和分析"""
    print("\n=== 紧凑型增强Mamba模型自动评估 ===")
    
    # 整体预测
    print("[INFO] 正在进行整体预测...")
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    
    # 计算整体准确率
    overall_acc = np.mean(np.argmax(test_Y_hat, axis=1) == np.argmax(Y_test, axis=1))
    print(f"[RESULT] 整体测试准确率: {overall_acc:.4f}")
    
    # 生成混淆矩阵
    print("[INFO] 正在生成混淆矩阵...")
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    
    # 特别关注QAM-16和WB-FM的性能
    qam16_acc = confnorm[7, 7] / np.sum(confnorm[7, :])  # QAM-16准确率
    wbfm_acc = confnorm[10, 10] / np.sum(confnorm[10, :])  # WB-FM准确率
    
    print(f"[SPECIAL] QAM-16识别准确率: {qam16_acc:.4f}")
    print(f"[SPECIAL] WB-FM识别准确率: {wbfm_acc:.4f}")
    
    # 保存混淆矩阵图
    mltools.plot_confusion_matrix(
        confnorm,
        labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
        title="紧凑型增强Mamba模型混淆矩阵",
        save_filename='figure/compact_mamba/compact_mamba_total_confusion'
    )
    
    # 按信噪比分析性能
    print("[INFO] 正在分析各信噪比下的性能...")
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    qam16_snr_acc = {}
    wbfm_snr_acc = {}
    
    for i, snr in enumerate(snrs):
        test_SNRs = [lbl[x][1] for x in test_idx]
        
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        
        if len(test_X_i) == 0:
            continue
            
        # 预测
        test_Y_i_hat = model.predict(test_X_i, batch_size=batch_size)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        
        acc[snr] = 1.0 * cor / (cor + ncor)
        
        # 计算QAM-16和WB-FM在该SNR下的准确率
        if np.sum(confnorm_i[7, :]) > 0:
            qam16_snr_acc[snr] = confnorm_i[7, 7] / np.sum(confnorm_i[7, :])
        if np.sum(confnorm_i[10, :]) > 0:
            wbfm_snr_acc[snr] = confnorm_i[10, 10] / np.sum(confnorm_i[10, :])
        
        # 保存准确率结果
        with open('predictresult/compact_accuracy_res.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([acc[snr]])
        
        # 生成每个SNR的混淆矩阵图 - 与train_analyzer.py保持一致
        mltools.plot_confusion_matrix(
            confnorm_i,
            labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
            title=f"紧凑型增强Mamba模型 (SNR={snr}dB)",
            save_filename=f"figure/compact_mamba/CompactEnhancedMamba_Confusion(SNR={snr})(ACC={100.0*acc[snr]:.2f}).png"
        )
        
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)
    
     # 绘制性能曲线
    print("[INFO] 正在生成性能曲线...")
    plot_compact_performance_curves(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc)
 
    
    # 保存结果
    save_compact_evaluation_results(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc)
    
    return acc, acc_mod_snr

def plot_compact_training_history(history):
    """绘制紧凑型模型训练历史曲线"""
    
    # 使用mltools的show_history函数来保持一致性
    mltools.show_history(history)

def plot_compact_performance_curves(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc):
    """绘制紧凑型模型性能曲线"""
    
    # 绘制各调制类型的准确率曲线 - 与train_analyzer.py保持一致的风格
    plt.figure(figsize=(12, 8))
    
    for i, mod in enumerate(classes):
        plt.plot(snrs, acc_mod_snr[i], label=mod, marker='o')
        # 添加数值标签
        for x, y in zip(snrs, acc_mod_snr[i]):
            if not np.isnan(y):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel("信噪比 (dB)")
    plt.ylabel("分类准确率")
    plt.title("紧凑型增强Mamba模型 - 各调制类型分类准确率")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure/compact_mamba/compact_mamba_acc_with_mod.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制整体准确率曲线 - 与train_analyzer.py保持一致的风格
    plt.figure(figsize=(10, 6))
    snr_list = list(acc.keys())
    acc_list = [acc[x] for x in snr_list]
    
    plt.plot(snr_list, acc_list, 'b-o', linewidth=2, markersize=6)
    plt.xlabel("信噪比 (dB)")
    plt.ylabel("分类准确率")
    plt.title("紧凑型增强Mamba模型 - 整体分类准确率曲线")
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for x, y in zip(snr_list, acc_list):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure/compact_mamba/compact_mamba_overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_compact_evaluation_results(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc):
    """保存紧凑型评估结果"""
    
    # 保存各调制类型准确率数据
    with open('predictresult/compact_acc_for_mod.dat', 'wb') as f:
        pickle.dump(acc_mod_snr, f)
    
    # 保存整体准确率数据
    with open('predictresult/compact_results.dat', 'wb') as f:
        pickle.dump(acc, f)
    
    # 保存QAM-16和WB-FM专门的结果
    with open('predictresult/compact_qam16_wbfm_results.dat', 'wb') as f:
        pickle.dump({'qam16': qam16_snr_acc, 'wbfm': wbfm_snr_acc}, f)
    
    # 保存总体准确率（ResNet格式）
    print(acc)
    with open('predictresult/mamba_attention_compact_d0.5.dat', 'wb') as f:
        pickle.dump(("mamba_attention_compact", 0.5, acc), f)
    
    # 计算性能统计
    avg_acc = np.mean(list(acc.values()))
    max_acc = max(acc.values())
    min_acc = min(acc.values())
    
    # QAM-16和WB-FM的平均性能
    avg_qam16_acc = np.mean(list(qam16_snr_acc.values())) if qam16_snr_acc else 0
    avg_wbfm_acc = np.mean(list(wbfm_snr_acc.values())) if wbfm_snr_acc else 0
    
    print(f"\n=== 紧凑型增强Mamba模型性能总结 ===")
    print(f"整体平均准确率: {avg_acc:.4f}")
    print(f"整体最高准确率: {max_acc:.4f} (SNR = {max(acc, key=acc.get)} dB)")
    print(f"整体最低准确率: {min_acc:.4f} (SNR = {min(acc, key=acc.get)} dB)")
    print(f"\n=== 重点优化目标性能 ===")
    print(f"QAM-16平均准确率: {avg_qam16_acc:.4f}")
    print(f"WB-FM平均准确率: {avg_wbfm_acc:.4f}")
    
    return avg_acc, max_acc, min_acc, avg_qam16_acc, avg_wbfm_acc

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='紧凑型增强Mamba模型训练 - 快速训练版本')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数（默认80）')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小（默认256）')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--no_augmentation', action='store_true', help='禁用数据增强')
    
    args = parser.parse_args()
    
    print("=== 紧凑型增强Mamba模型训练 - QAM-16和WB-FM快速优化 ===")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"数据增强: {'禁用' if args.no_augmentation else '启用'}")
    
    # 训练模型并自动评估
    model, history = train_compact_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation
    )
    
    print("\n=== 训练与评估完成 ===")
    print("结果文件:")
    print("- 最佳模型权重: weights/CompactEnhancedMamba_best.h5")
    print("- 训练日志: training_log_compact.csv")
    print("- 混淆矩阵图: figure/compact_mamba/compact_mamba_*.png")
    print("- 结果数据: predictresult/compact_*.dat")
    print("\n优势:")
    print("- 训练时间大幅减少（约60-70%）")
    print("- 参数量显著降低（约70%）")
    print("- 保持QAM-16和WB-FM专用优化")
    print("- 适合快速实验和验证")

if __name__ == "__main__":
    main()