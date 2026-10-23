# SPIDER：等级盲的 Residual MLP++ DDS

本仓库整理了 SPIDER 项目中的最新紧凑型严重程度评分模块：一个不使用真实等级标签的 Residual MLP++，将冻结 VAE 的潜在特征映射为连续退变严重程度评分（DDS）。该模块仅负责下游评分；它不会训练 MRI VAE，也不会生成 21 帧 LCOVT 轨迹。

## 方法

- 将 4,480 维冻结 VAE 潜在特征映射为筛选后的 384 维表示。
- 使用残差 MLP 进行评分：384 → 128、两个残差块，再经 64 → 1 输出连续分数。
- 在生成的等级条件轨迹中，利用相邻样本对和抽样的长距离样本对学习相对排序。
- 根据验证集轨迹排序指标选择检查点。
- 使用训练集潜变量分数，将连续评分校准到 0–100 的 DDS 范围。
- 真实 Pfirrmann 等级不参与模型拟合、特征筛选、验证检查点选择或分数校准；可选的等级关联分析只在模型冻结后进行。

模型针对 384 维输入，共有 123,905 个参数。仓库提供的检查点包含推理所需的模型、特征归一化器和校准值。

## 目录结构

    ResidualMLP_PlusPlus_384/
    ├── residual_mlp_plusplus_dds.py
    ├── run_all.sh
    ├── requirements.txt
    └── checkpoints/
        ├── model_residual_mlp_plusplus.pt
        ├── feature_normalizer.npz
        └── calibration.json

## 环境安装

    python -m venv .venv
    # 激活虚拟环境后运行：
    pip install -r ResidualMLP_PlusPlus_384/requirements.txt

## 训练

训练需要另行准备元数据、VAE 潜变量缓存以及生成的 21 帧 LCOVT 轨迹。本仓库不包含这些文件。请将环境变量指向你有权使用的本地 SPIDER 数据目录：

    SPIDER_ROOT=/path/to/SPIDER-data bash ResidualMLP_PlusPlus_384/run_all.sh

训练脚本支持显式指定元数据、缓存和训练/验证轨迹目录。可查看命令帮助：

    python ResidualMLP_PlusPlus_384/residual_mlp_plusplus_dds.py train --help

训练流程会将新检查点、预测和分析文件写入输出目录；这些文件已由 Git 忽略。

## 基于潜在特征进行预测

预测命令需要一个 N × 4,480 的潜变量矩阵，以及包含 `image_path` 列、共 N 行的清单文件。潜变量需由相同的冻结 VAE 预处理流程生成：

    python ResidualMLP_PlusPlus_384/residual_mlp_plusplus_dds.py predict \
      --model-dir ResidualMLP_PlusPlus_384/checkpoints \
      --latents new_latents.npy \
      --manifest new_manifest.csv \
      --output predictions.csv \
      --device cuda

脚本输出 `raw_score` 和校准后的 `DDS`。它不会将 DICOM/JPEG 原始图像转换为 VAE 潜变量。

## 项目范围与结果解释

本仓库包含最终评分代码和紧凑推理文件，不包含上游生成模型、原始影像、掩膜、潜变量缓存、患者元数据、轨迹图像或实验报告。DDS 是锚定于训练集潜变量分布的研究评分；轨迹排序一致性本身不能证明临床有效性。该评分不能用于临床诊断。
