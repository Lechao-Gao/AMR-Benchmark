#!/usr/bin/env python3
"""
增强版Mamba模型训练脚本 - 专门优化QAM-16和WB-FM识别
使用专门的特征提取器和增强损失函数
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
import rmlmodels.EnhancedMambaModel as enhanced_mamba

class FocalLoss(BaseLoss):
    """焦点损失，专门处理难分类样本"""
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # 计算交叉熵
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 计算pt
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # 计算焦点损失
        focal_loss = self.alpha * tf.pow(1 - pt, self.gamma) * ce_loss
        
        return focal_loss

class QAMFMWeightedLoss(BaseLoss):
    """专门为QAM-16和WB-FM加权的损失函数"""
    def __init__(self, qam_weight=3.0, fm_weight=3.0, **kwargs):
        super(QAMFMWeightedLoss, self).__init__(**kwargs)
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

def create_data_augmentation():
    """创建数据增强函数"""
    def augment_qam_fm(X, Y, augment_ratio=0.5):
        """专门为QAM-16和WB-FM进行数据增强"""
        augmented_X = []
        augmented_Y = []
        
        for i in range(len(X)):
            # 获取标签
            label_idx = np.argmax(Y[i])
            
            # 只对QAM-16 (索引7) 和 WB-FM (索引10) 进行增强
            if label_idx == 7 or label_idx == 10:  # QAM-16 or WB-FM
                if np.random.random() < augment_ratio:
                    x_aug = X[i].copy()
                    
                    if label_idx == 7:  # QAM-16增强
                        # IQ不平衡增强
                        iq_imbalance = np.random.normal(0, 0.05)
                        x_aug[0, :, 0] *= (1 + iq_imbalance)  # I通道
                        x_aug[1, :, 0] *= (1 - iq_imbalance)  # Q通道
                        
                        # 相位偏移
                        phase_offset = np.random.uniform(-0.1, 0.1)
                        x_aug[1, :, 0] += phase_offset
                        
                    elif label_idx == 10:  # WB-FM增强
                        # 频率偏移
                        freq_offset = np.random.uniform(-0.05, 0.05)
                        t = np.arange(x_aug.shape[1])
                        x_aug[0, :, 0] += freq_offset * np.sin(0.1 * t)
                        x_aug[1, :, 0] += freq_offset * np.cos(0.1 * t)
                        
                        # 调制深度变化
                        mod_depth_factor = np.random.uniform(0.8, 1.2)
                        x_aug *= mod_depth_factor
                    
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
    
    return augment_qam_fm

def train_enhanced_model(epochs=100, batch_size=128, learning_rate=0.001, use_augmentation=True):
    """训练增强版Mamba模型"""
    
    print("=== 增强版Mamba模型训练 - 专门优化QAM-16和WB-FM ===" )
    print("特殊优化:")
    print("- QAM专用特征提取器: 星座图、IQ不平衡、符号速率")
    print("- FM专用特征提取器: 瞬时频率、调制深度、频偏检测")
    print("- 自适应特征融合: 动态权重分配")
    print("- 增强Mamba块: 双向处理 + 门控融合")
    print("- 多尺度注意力: 局部 + 全局特征")
    print("- 加权损失函数: QAM-16和WB-FM权重提升")
    
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
    
    # 数据增强
    if use_augmentation:
        print("\n正在进行数据增强...")
        augment_func = create_data_augmentation()
        X_train_aug, Y_train_aug = augment_func(X_train, Y_train)
        print(f"增强后训练数据形状: {X_train_aug.shape}")
        X_train, Y_train = X_train_aug, Y_train_aug
    
    # 创建增强模型
    print("\n正在构建增强版Mamba模型...")
    model = enhanced_mamba.EnhancedMambaModel(
        input_shape=[2, 128],
        classes=11,
        d_model=256,
        num_mamba_layers=3,
        num_attention_layers=2
    )
    
    # 创建加权损失函数
    weighted_loss = QAMFMWeightedLoss(qam_weight=3.0, fm_weight=3.0)
    
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(
        loss=weighted_loss,
        metrics=['accuracy'],
        optimizer=optimizer
    )
    
    print("\n增强版模型结构:")
    model.summary()
    print(f"模型总参数量: {model.count_params():,}")
    
    # 设置回调函数
    os.makedirs('weights', exist_ok=True)
    os.makedirs('figure/enhanced_mamba', exist_ok=True)
    os.makedirs('predictresult', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'weights/EnhancedMamba_best.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # 更激进的学习率衰减
            verbose=1,
            patience=8,
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,  # 更长的耐心等待
            verbose=1,
            mode='auto',
            restore_best_weights=True
        ),
        CSVLogger(
            'training_log_enhanced.csv',
            append=False
        )
    ]
    
    print("\n开始训练...")
    print("注意: 使用专门的QAM-16和WB-FM优化策略")
    
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
    predict_and_analyze_enhanced(model, X_test, Y_test, classes, snrs, test_idx, lbl, batch_size)
    generate_tsne_visualization(
        model=model,
        X_test=X_test,
        Y_test=Y_test,
        lbl=lbl,
        snrs=snrs,
        test_idx=test_idx,
        classes=classes,
        feature_layer='dense3',
        save_name='figure/enhanced_mamba/tsne_enhanced_mamba',
        weights_path='weights/EnhancedMamba_best.h5'
    )
    
    return model, history

def predict_and_analyze_enhanced(model, X_test, Y_test, classes, snrs, test_idx, lbl, batch_size=128):
    """增强版模型的预测和分析"""
    print("\n=== 增强版Mamba模型自动评估 ===")
    
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
        title="增强版Mamba模型混淆矩阵",
        save_filename='figure/enhanced_mamba/enhanced_mamba_total_confusion'
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
        with open('predictresult/enhanced_accuracy_res.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([acc[snr]])
        
        # 生成混淆矩阵图
        mltools.plot_confusion_matrix(
            confnorm_i,
            labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],
            title=f"增强版Mamba模型 (SNR={snr}dB)",
            save_filename=f"figure/enhanced_mamba/EnhancedMamba_Confusion(SNR={snr})(ACC={100.0*acc[snr]:.2f}).png"
        )
        
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)
    
    # 绘制性能曲线
    print("[INFO] 正在生成性能曲线...")
    plot_enhanced_performance_curves(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc)
    
    # 保存结果
    save_enhanced_evaluation_results(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc)
    
    return acc, acc_mod_snr

def plot_enhanced_performance_curves(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc):
    """绘制增强版性能曲线"""
    
    # 绘制各调制类型的准确率曲线，突出QAM-16和WB-FM
    plt.figure(figsize=(14, 10))
    
    for i, mod in enumerate(classes):
        if mod == '16-QAM':
            plt.plot(snrs, acc_mod_snr[i], label=mod, marker='o', linewidth=3, markersize=8, color='red')
        elif mod == 'WBFM':
            plt.plot(snrs, acc_mod_snr[i], label=mod, marker='s', linewidth=3, markersize=8, color='blue')
        else:
            plt.plot(snrs, acc_mod_snr[i], label=mod, marker='o', alpha=0.7)
    
    plt.xlabel("信噪比 (dB)")
    plt.ylabel("分类准确率")
    plt.title("增强版Mamba模型 - 各调制类型分类准确率\n")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure/enhanced_mamba/enhanced_mamba_acc_with_mod.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 专门绘制QAM-16和WB-FM的性能对比
    plt.figure(figsize=(12, 6))
    
    qam16_snrs = list(qam16_snr_acc.keys())
    qam16_accs = [qam16_snr_acc[snr] for snr in qam16_snrs]
    
    wbfm_snrs = list(wbfm_snr_acc.keys())
    wbfm_accs = [wbfm_snr_acc[snr] for snr in wbfm_snrs]
    
    plt.plot(qam16_snrs, qam16_accs, 'r-o', linewidth=3, markersize=8, label='QAM-16')
    plt.plot(wbfm_snrs, wbfm_accs, 'b-s', linewidth=3, markersize=8, label='WB-FM')
    
    # 添加数值标签
    for x, y in zip(qam16_snrs, qam16_accs):
        plt.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='red')
    for x, y in zip(wbfm_snrs, wbfm_accs):
        plt.text(x, y, f'{y:.3f}', ha='center', va='top', fontsize=10, color='blue')
    
    plt.xlabel("信噪比 (dB)")
    plt.ylabel("分类准确率")
    plt.title("QAM-16和WB-FM识别性能对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure/enhanced_mamba/qam16_wbfm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_enhanced_evaluation_results(acc, acc_mod_snr, classes, snrs, qam16_snr_acc, wbfm_snr_acc):
    """保存增强版评估结果"""
    
    # 保存各调制类型准确率数据
    with open('predictresult/enhanced_acc_for_mod.dat', 'wb') as f:
        pickle.dump(acc_mod_snr, f)
    
    # 保存整体准确率数据
    with open('predictresult/enhanced_results.dat', 'wb') as f:
        pickle.dump(acc, f)
    
    # 保存QAM-16和WB-FM专门的结果
    with open('predictresult/qam16_wbfm_results.dat', 'wb') as f:
        pickle.dump({'qam16': qam16_snr_acc, 'wbfm': wbfm_snr_acc}, f)
    
    # 保存总体准确率（ResNet格式）
    print(acc)
    with open('predictresult/mamba_attention_enhanced_d0.5.dat', 'wb') as f:
        pickle.dump(("mamba_attention_enhanced", 0.5, acc), f)
    
    # 计算性能统计
    avg_acc = np.mean(list(acc.values()))
    max_acc = max(acc.values())
    min_acc = min(acc.values())
    
    # QAM-16和WB-FM的平均性能
    avg_qam16_acc = np.mean(list(qam16_snr_acc.values())) if qam16_snr_acc else 0
    avg_wbfm_acc = np.mean(list(wbfm_snr_acc.values())) if wbfm_snr_acc else 0
    
    print(f"\n=== 增强版Mamba模型性能总结 ===")
    print(f"整体平均准确率: {avg_acc:.4f}")
    print(f"整体最高准确率: {max_acc:.4f} (SNR = {max(acc, key=acc.get)} dB)")
    print(f"整体最低准确率: {min_acc:.4f} (SNR = {min(acc, key=acc.get)} dB)")
    print(f"\n=== 重点优化目标性能 ===")
    print(f"QAM-16平均准确率: {avg_qam16_acc:.4f}")
    print(f"WB-FM平均准确率: {avg_wbfm_acc:.4f}")
    
    return avg_acc, max_acc, min_acc, avg_qam16_acc, avg_wbfm_acc

def generate_tsne_visualization(model, X_test, Y_test, lbl, snrs, test_idx, classes,
                                feature_layer='dense3', save_name='figure/enhanced_mamba/tsne_enhanced_mamba',
                                weights_path=None):
    """参考ResNet脚本生成t-SNE，可视化增强模型特征。"""
    print("\n" + "=" * 60)
    print("开始生成t-SNE可视化图...")
    print("=" * 60)

    test_SNRs = [lbl[x][1] for x in test_idx]

    selected_snrs_for_tsne = [-20, -10, 0, 10, 18]
    selected_snrs_for_tsne = [s for s in selected_snrs_for_tsne if s in snrs]
    if len(selected_snrs_for_tsne) < 3:
        additional_snrs = [s for s in snrs if s not in selected_snrs_for_tsne]
        selected_snrs_for_tsne.extend(additional_snrs[:5 - len(selected_snrs_for_tsne)])

    try:
        if weights_path and os.path.exists(weights_path):
            print(f"加载最佳权重: {weights_path}")
            model.load_weights(weights_path)

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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版Mamba模型训练 - 专门优化QAM-16和WB-FM')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--no_augmentation', action='store_true', help='禁用数据增强')
    
    args = parser.parse_args()
    
    print("=== 增强版Mamba模型训练 - QAM-16和WB-FM专门优化 ===")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"数据增强: {'禁用' if args.no_augmentation else '启用'}")
    
    # 训练模型并自动评估
    model, history = train_enhanced_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation
    )
    
    print("\n=== 训练与评估完成 ===")
    print("结果文件:")
    print("- 最佳模型权重: weights/EnhancedMamba_best.h5")
    print("- 训练日志: training_log_enhanced.csv")
    print("- 混淆矩阵图: figure/enhanced_mamba/enhanced_mamba_*.png")
    print("- QAM-16/WB-FM对比图: figure/enhanced_mamba/qam16_wbfm_comparison.png")
    print("- 结果数据: predictresult/enhanced_*.dat")

if __name__ == "__main__":
    main()