# Mamba双注意力调制信号识别模型

## 模型概述

本模型是一个基于Mamba架构和双注意力机制的调制信号识别模型，专门针对RML2016.10a数据集设计。模型结合了状态空间模型的长序列建模能力和注意力机制的全局特征捕获能力，实现了对调制信号的高精度识别。

## 模型架构

```
多特征输入模块 → 小核卷积 → 特征融合 → Mamba主干结构[嵌入注意力块] → 分类输出
```

### 核心组件

1. **多特征输入模块**
   - **I/Q数据分支**: 直接处理原始I/Q信号
   - **幅度/相位分支**: 提取幅度和相位特征
   - **时频图分支**: 模拟短时傅里叶变换特征

2. **Mamba块 (MambaBlock)**
   - 基于状态空间模型的选择性扫描机制
   - 结合卷积和门控机制
   - 高效处理长序列依赖关系

3. **双注意力块 (DualAttentionBlock)**
   - 自注意力机制：捕捉序列内部依赖
   - 缩放点积注意力：增强全局特征建模
   - 残差连接和层归一化

4. **小核卷积层 (SmallKernelConv)**
   - 细粒度特征提取
   - 噪声抑制和局部特征增强

## 模型特点

- **多特征融合**: 充分利用I/Q数据、幅度相位、时频图等多种特征表示
- **全局建模**: Mamba架构实现高效的长序列全局特征建模
- **抗噪学习**: 双注意力机制增强对噪声的鲁棒性
- **细粒度识别**: 小核卷积捕捉调制类型的细微差异

## 支持的调制类型

模型支持识别以下11种调制类型：
- 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, 4-PAM, 16-QAM, 64-QAM, QPSK, WBFM

## 文件结构

```
MambaAttention/
├── main.py                           # 主训练脚本
├── rmlmodels/
│   └── MambaAttentionModel.py        # 模型定义
├── weights/                          # 训练权重保存目录
├── figure/                           # 结果图表保存目录
├── predictresult/                    # 预测结果保存目录
└── README.md                         # 说明文档
```

## 使用方法

### 1. 环境要求

```bash
# 主要依赖
tensorflow >= 2.8.0
keras >= 2.8.0
numpy
matplotlib
pickle
h5py
```

### 2. 训练模型

```bash
cd RML201610a/MambaAttention
python main.py
```

### 3. 模型参数配置

在 `main.py` 中可以调整以下参数：

```python
# 训练参数
nb_epoch = 1000          # 训练轮数
batch_size = 128         # 批次大小
learning_rate = 0.001    # 学习率

# 模型参数
d_model = 256            # 模型维度
num_mamba_layers = 4     # Mamba层数量
num_attention_layers = 2 # 注意力层数量
```

## 模型性能

模型在RML2016.10a数据集上的表现：

- **参数量**: 约1.5M参数
- **训练时间**: 根据硬件配置，约2-4小时
- **内存需求**: 建议8GB以上GPU内存

### 预期性能指标

- 高信噪比(>10dB): 准确率 > 90%
- 中信噪比(0-10dB): 准确率 > 80%
- 低信噪比(<0dB): 准确率 > 60%

## 输出文件

训练完成后会生成以下文件：

1. **权重文件**
   - `weights/MambaAttention_best.h5`: 最佳模型权重

2. **训练日志**
   - `training_log.csv`: 详细训练日志
   - `accuray_res.csv`: 各SNR准确率记录

3. **可视化结果**
   - `figure/mamba_attention_total_confusion.png`: 总体混淆矩阵
   - `figure/MambaAttention_Confusion(SNR=X).png`: 各SNR混淆矩阵
   - `figure/mamba_attention_acc_with_mod_*.png`: 各调制类型准确率曲线
   - `figure/mamba_attention_overall_accuracy.png`: 整体准确率曲线

4. **预测数据**
   - `predictresult/acc_for_mod_on_mamba_attention.dat`: 各调制类型准确率数据
   - `predictresult/MambaAttention_results.dat`: 整体预测结果

## 模型创新点

1. **多特征协同**: 首次将I/Q、幅度相位、时频图三种特征在Mamba架构中协同建模
2. **选择性注意**: 双注意力机制实现对不同调制特征的选择性关注
3. **长序列建模**: Mamba架构高效处理128长度的信号序列
4. **噪声鲁棒性**: 小核卷积和注意力机制联合实现噪声抑制

## 技术细节

### Mamba块实现
- 使用简化的状态空间模型
- 结合选择性扫描机制
- 门控和残差连接增强表达能力

### 双注意力设计
- 自注意力捕捉序列内依赖
- 缩放注意力增强全局建模
- 多头机制提升特征多样性

### 特征融合策略
- 加权融合多种特征表示
- 自适应特征选择
- 层次化特征提取

## 扩展建议

1. **数据增强**: 可添加噪声增强、频域变换等数据增强技术
2. **模型压缩**: 可使用知识蒸馏、量化等技术减小模型大小
3. **实时推理**: 可优化模型结构以支持实时调制识别
4. **迁移学习**: 可在其他调制识别数据集上进行微调

## 引用

如果您使用了本模型，请引用相关的Mamba和注意力机制论文：

```bibtex
@article{mamba2023,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{attention2017,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  journal={Advances in neural information processing systems},
  year={2017}
}
```

## 联系方式

如有问题或建议，请通过项目仓库提交Issue。