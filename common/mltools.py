"""
通用工具函数模块，用于无线电信号调制分类项目
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_chinese_init
import numpy as np
import pickle
import os

# 延迟导入 Keras（TensorFlow 版），避免在没有安装时出错
try:
    from tensorflow.keras.models import Model
except ImportError:
    Model = None

def is_gpu_available(verbose=False):
    """
    检查当前环境是否支持GPU训练
    
    参数:
        verbose: 是否打印详细信息，默认为False
        
    返回:
        bool: 如果GPU可用返回True，否则返回False
    """
    try:
        import tensorflow as tf
        
        # 检查是否有物理GPU设备
        gpus = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0
        
        if verbose:
            print("=" * 60)
            print("GPU 可用性检测")
            print("=" * 60)
            print(f"TensorFlow 版本: {tf.__version__}")
            print(f"检测到的 GPU 数量: {len(gpus)}")
            
            if gpus:
                print("✓ GPU 可用")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                    try:
                        details = tf.config.experimental.get_device_details(gpu)
                        if details:
                            print(f"    详情: {details}")
                    except:
                        pass
            else:
                print("✗ GPU 不可用")
            print("=" * 60)
        
        return gpu_available
        
    except ImportError:
        if verbose:
            print("错误: 未安装 TensorFlow")
        return False
    except Exception as e:
        if verbose:
            print(f"检测 GPU 时发生错误: {e}")
        return False

# # 全局设置中文字体，使用文泉驿微米黑
# plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
# # 解决负号显示问题，确保负号正确显示为"-"而不是方块
# plt.rcParams['axes.unicode_minus'] = False

# 确保figure目录存在
def ensure_figure_dir():
    """确保figure目录存在，如果不存在则创建"""
    if not os.path.exists('figure'):
        os.makedirs('figure')
    if not os.path.exists('predictresult'):
        os.makedirs('predictresult')

# 显示训练历史
def show_history(history, save_prefix=''):
    """
    显示训练历史曲线
    
    参数:
        history: Keras训练历史对象
        save_prefix: 保存文件名的前缀（可选），默认为空字符串
    """
    ensure_figure_dir()
    
    # 查找所有可能的准确率指标键名
    acc_keys = []
    val_acc_keys = []
    
    # 检查标准键名
    if 'accuracy' in history.history:
        acc_keys.append('accuracy')
    elif 'acc' in history.history:
        acc_keys.append('acc')
        
    if 'val_accuracy' in history.history:
        val_acc_keys.append('val_accuracy')
    elif 'val_acc' in history.history:
        val_acc_keys.append('val_acc')
    
    # 检查多输出模型的键名（如xc_accuracy, xc_acc等）
    for key in history.history.keys():
        if key.endswith('_accuracy') and key not in acc_keys and not key.startswith('val_'):
            acc_keys.append(key)
        elif key.endswith('_acc') and key not in acc_keys and not key.startswith('val_'):
            acc_keys.append(key)
        elif key.startswith('val_') and key.endswith('_accuracy') and key not in val_acc_keys:
            val_acc_keys.append(key)
        elif key.startswith('val_') and key.endswith('_acc') and key not in val_acc_keys:
            val_acc_keys.append(key)
    
    # 构建文件名
    loss_filename = f'figure/{save_prefix}total_loss.png' if save_prefix else 'figure/total_loss.png'
    acc_filename = f'figure/{save_prefix}total_acc.png' if save_prefix else 'figure/total_acc.png'
    
    # 绘制损失曲线
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig(loss_filename)
    plt.close()
    
    # 如果找到了准确率指标，绘制准确率曲线
    if acc_keys and val_acc_keys:
        # 使用第一个找到的准确率指标
        acc_key = acc_keys[0]
        val_acc_key = val_acc_keys[0]
        
        plt.figure()
        plt.title('Training accuracy performance')
        plt.plot(history.epoch, history.history[acc_key], label=f'train_{acc_key}')
        plt.plot(history.epoch, history.history[val_acc_key], label=f'val_{acc_key}')
        plt.legend()
        plt.savefig(acc_filename)
        plt.close()
        
        # 保存准确率和损失值到文本文件
        train_acc = history.history[acc_key]
        val_acc = history.history[val_acc_key]
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epoch = history.epoch
        np_train_acc = np.array(train_acc)
        np_val_acc = np.array(val_acc)
        np_train_loss = np.array(train_loss)
        np_val_loss = np.array(val_loss)
        np_epoch = np.array(epoch)
        
        # 使用前缀保存文本文件
        train_acc_filename = f'{save_prefix}train_acc.txt' if save_prefix else 'train_acc.txt'
        train_loss_filename = f'{save_prefix}train_loss.txt' if save_prefix else 'train_loss.txt'
        val_acc_filename = f'{save_prefix}val_acc.txt' if save_prefix else 'val_acc.txt'
        val_loss_filename = f'{save_prefix}val_loss.txt' if save_prefix else 'val_loss.txt'
        
        np.savetxt(train_acc_filename, np_train_acc)
        np.savetxt(train_loss_filename, np_train_loss)
        np.savetxt(val_acc_filename, np_val_acc)
        np.savetxt(val_loss_filename, np_val_loss)
    else:
        print("警告：未找到准确率指标。可用的键名：", list(history.history.keys()))

def plot_lstm2layer_output(a, modulation_type=None, save_filename=None):
    """
    绘制LSTM第2层输出
    
    参数:
        a: 输出数据
        modulation_type: 调制类型
        save_filename: 保存文件名
    """
    ensure_figure_dir()
    
    plt.figure(figsize=(4, 3), dpi=600)
    plt.plot(range(128), a[0], label=modulation_type)
    plt.legend()
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])
    plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

def plot_conv4layer_output(a, modulation_type=None):
    """
    绘制卷积第4层输出
    
    参数:
        a: 输出数据
        modulation_type: 调制类型
    """
    ensure_figure_dir()
    
    plt.figure(figsize=(4, 3), dpi=600)
    for i in range(100):
        plt.plot(range(124), a[0, 0, :, i])
        plt.xticks([])  # 去掉横坐标值
        plt.yticks(size=20)
        save_filename = './figure_conv4_output/output%d.png' % i
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.close()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    """
    绘制混淆矩阵
    
    参数:
        cm: 混淆矩阵数据
        title: 标题
        cmap: 颜色映射
        labels: 标签列表
        save_filename: 保存文件名
    """
    ensure_figure_dir()
    
    plt.figure(figsize=(4, 3), dpi=600)
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, size=12)
    plt.yticks(tick_marks, labels, size=12)
    
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i != j:
                text = plt.text(j, i, int(np.around(cm[i, j]*100)), ha="center", va="center", fontsize=10)
            elif i == j:
                if int(np.around(cm[i, j]*100)) == 100:
                    text = plt.text(j, i, int(np.around(cm[i, j]*100)), ha="center", va="center", fontsize=7, color='darkorange')
                else:
                    text = plt.text(j, i, int(np.around(cm[i, j]*100)), ha="center", va="center", fontsize=10, color='darkorange')

    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()

def calculate_confusion_matrix(Y, Y_hat, classes):
    """
    计算混淆矩阵
    
    参数:
        Y: 真实标签
        Y_hat: 预测标签
        classes: 类别列表
        
    返回:
        confnorm: 归一化混淆矩阵
        right: 正确预测数
        wrong: 错误预测数
    """
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    for k in range(0, Y.shape[0]):
        i = list(Y[k, :]).index(1)
        j = int(np.argmax(Y_hat[k, :]))
        conf[i, j] = conf[i, j] + 1

    for i in range(0, n_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong

def plot_tsne_visualization(model, X_data, Y_data, snr_data, classes, snrs, 
                            feature_layer_name='fc1', 
                            save_filename='figure/tsne_visualization.png',
                            selected_snrs=None,
                            n_samples_per_snr=500,
                            perplexity=30,
                            n_iter=1000):
    """
    使用t-SNE可视化模型在不同信噪比下对不同调制方式的分类能力
    
    参数:
        model: 训练好的Keras模型
        X_data: 输入数据 (n_samples, ...)
        Y_data: 真实标签 (n_samples, n_classes)
        snr_data: 每个样本对应的SNR值列表
        classes: 调制方式类别列表
        snrs: 所有SNR值列表
        feature_layer_name: 用于提取特征的层名称，默认为'fc1'
        save_filename: 保存文件名
        selected_snrs: 要可视化的SNR列表，如果为None则选择所有SNR
        n_samples_per_snr: 每个SNR选择的样本数量，用于加速计算
        perplexity: t-SNE的困惑度参数，默认为30
        n_iter: t-SNE迭代次数，默认为1000
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("错误: 需要安装scikit-learn库才能使用t-SNE可视化")
        print("请运行: pip install scikit-learn")
        return
    
    ensure_figure_dir()
    
    # 导入Model类
    # 使用全局变量检查，避免局部变量引用错误
    global Model
    if Model is None:
        try:
            from tensorflow.keras.models import Model
        except ImportError:
            print("错误: 需要安装 TensorFlow Keras 才能使用t-SNE可视化")
            print("请运行: pip install tensorflow")
            return
    
    # 检查是否为多输入模型
    is_multi_input = isinstance(X_data, (list, tuple))
    
    # 创建特征提取模型（从指定层提取特征）
    try:
        if feature_layer_name is None:
            # 如果 feature_layer_name 为 None，说明传入的是中间模型，直接使用
            feature_model = model
        else:
            feature_model = Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
    except (ValueError, AttributeError):
        # 如果找不到指定层，尝试使用倒数第二层（通常是全连接层）
        if feature_layer_name is not None:
            print(f"警告: 找不到层 '{feature_layer_name}'，尝试使用倒数第二层")
        try:
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        except:
            # 如果还是失败，可能是中间模型，直接使用
            feature_model = model
    
    # 如果selected_snrs为None，选择所有SNR
    if selected_snrs is None:
        selected_snrs = snrs
    
    # 为每个选定的SNR生成t-SNE图
    for snr in selected_snrs:
        # 筛选特定SNR的数据
        snr_indices = np.where(np.array(snr_data) == snr)[0]
        
        if len(snr_indices) == 0:
            print(f"警告: SNR={snr}dB 没有数据，跳过")
            continue
        
        # 如果样本太多，随机采样
        if len(snr_indices) > n_samples_per_snr:
            np.random.seed(42)  # 固定随机种子以便结果可复现
            snr_indices = np.random.choice(snr_indices, size=n_samples_per_snr, replace=False)
        elif len(snr_indices) <= max(2, perplexity):
            print(f"警告: SNR={snr}dB 的样本数 ({len(snr_indices)}) 不足以执行 t-SNE，已跳过")
            continue
        
        # 处理多输入模型的情况
        if is_multi_input:
            # 如果是多输入，对每个输入都进行索引
            X_snr = [x[snr_indices] for x in X_data]
        else:
            # 单输入模型
            X_snr = X_data[snr_indices]
        
        Y_snr = Y_data[snr_indices]
        
        # 提取特征
        print(f"正在提取SNR={snr}dB的特征...")
        features = feature_model.predict(X_snr, verbose=0, batch_size=400)

        # 将特征展平成二维矩阵，避免后续算法遇到高阶张量
        features = features.reshape(features.shape[0], -1)

        # 如果包含 NaN/Inf，先做替换，避免 scikit-learn 报警
        if np.any(~np.isfinite(features)):
            print(f"警告: SNR={snr}dB 的特征存在 NaN/Inf，已自动替换为 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 过滤掉方差过小的维度，避免PCA或TSNE出现 total_var=0 的情况
        variances = np.var(features, axis=0)
        valid_dims = variances > 1e-12
        if not np.any(valid_dims):
            print(f"警告: SNR={snr}dB 的特征方差为0，无法进行 t-SNE，已跳过")
            continue
        if np.count_nonzero(valid_dims) < features.shape[1]:
            features = features[:, valid_dims]

        # 对特征做零均值化，帮助t-SNE收敛
        features = features - np.mean(features, axis=0, keepdims=True)
        
        # 获取真实标签
        true_labels = np.argmax(Y_snr, axis=1)
        
        # 执行t-SNE降维
        print(f"正在对SNR={snr}dB执行t-SNE降维...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=n_iter,
            verbose=0,
            init='random',
            learning_rate='auto'
        )
        features_2d = tsne.fit_transform(features)

        # 如果降维结果退化到一条直线或一个点，直接跳过绘图
        if np.allclose(np.std(features_2d, axis=0), 0, atol=1e-9):
            print(f"警告: SNR={snr}dB 的 t-SNE 结果退化，可能样本过于相似，已跳过绘图")
            continue
        
        # 绘制t-SNE图
        plt.figure(figsize=(12, 10))
        
        # 为每个调制方式分配颜色和标记
        colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
        markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        
        for i, mod in enumerate(classes):
            mod_indices = np.where(true_labels == i)[0]
            if len(mod_indices) > 0:
                plt.scatter(features_2d[mod_indices, 0], 
                           features_2d[mod_indices, 1],
                           c=[colors[i]], 
                           marker=markers[i % len(markers)],
                           label=mod,
                           alpha=0.6,
                           s=30,
                           edgecolors='black',
                           linewidths=0.5)
        
        # 移除坐标轴标签和刻度，只展示聚集情况
        plt.xticks([])  # 移除x轴刻度
        plt.yticks([])  # 移除y轴刻度
        plt.xlabel('')  # 移除x轴标签
        plt.ylabel('')  # 移除y轴标签
        plt.title(f't-SNE  visualization- SNR = {snr} dB\n', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(False)  # 移除网格
        plt.tight_layout()
        
        # 保存图片 - 处理文件名（无论是否有扩展名）
        if save_filename.endswith('.png'):
            # 如果有.png扩展名，替换它
            base_name = save_filename[:-4]  # 移除 .png
            snr_filename = f"{base_name}_SNR_{snr}dB.png"
        else:
            # 如果没有扩展名，直接添加SNR和扩展名
            snr_filename = f"{save_filename}_SNR_{snr}dB.png"
        
        plt.savefig(snr_filename, dpi=300, bbox_inches='tight')
        print(f"已保存t-SNE图: {snr_filename}")
        plt.close()
    
    print("t-SNE可视化完成！")