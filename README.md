# SPIDER：腰椎 MRI 条件生成与退变评分

SPIDER 是一个面向椎间盘 MRI 的研究代码项目，覆盖影像预处理、VAE 潜空间建模、Pfirrmann 等级条件扩散生成，以及生成轨迹上的下游退变严重程度评分。仓库保留核心训练、生成、评估代码和推理所需模型；原始 MRI、掩膜、患者元数据、生成图像和实验结果均不包含在内。

## 核心方法

### 1. 影像预处理与 VAE

预处理脚本从原始影像中提取椎间盘区域，并生成适配模型的二维灰度图。VAE 使用 ConvNeXt 风格卷积块和多尺度下采样/上采样结构，将单通道、约 160×224 的 MRI 映射到 8 通道潜变量。训练目标由重建误差、SSIM 项和 KL 正则组成，并通过 KL warm-up 与 latent scale 校准稳定潜空间。

### 2. Pfirrmann 等级条件潜空间扩散

`train_ordinal_latent_diffusion_v2.py` 在冻结 VAE 的潜空间中训练扩散模型，学习由 Pfirrmann 有序等级（G1–G5）与椎间盘节段共同条件化的潜变量分布。等级以连续有序坐标表示，减少把相邻等级当作互不相关类别的限制。

### 3. LCOVT 连续等级轨迹生成

`stage2_lcovt_v3_nlp_inspired_batchopt.py` 与配套运行脚本实现 Local Contrastive Ordinal Velocity Transport（LCOVT）推理：沿相邻等级条件下的扩散速度差进行连续控制，在源影像潜变量上生成从 G1 到 G5 的 21 帧轨迹，并保留五个整数等级锚点。该流程用于研究等级变化下的生成连续性和解剖结构保持。

### 4. 下游 DDS 严重程度评分

`ResidualMLP_PlusPlus_384/` 提供等级盲的 Residual MLP++ 评分模块。模型把冻结 VAE 的潜变量筛选为 384 维表示，通过残差 MLP 输出连续分数；训练使用生成轨迹上的相对排序约束。目录内保留一份模型权重、特征归一化参数和分数校准参数，供评分模块推理使用。

## 代码结构

```text
SPIDER/
├── data/                                  # 数据读取代码，不含患者数据
├── model/                                 # VAE、扩散调度器、等级条件网络与网络模块
├── utils/                                 # 扩散、I/O 与指标工具
├── extract_spider_ivd_2d_224x160_true_display_space_crop_v6_full.py
├── train_vae_v2.py
├── train_ordinal_latent_diffusion_v2.py
├── stage2_lcovt_v3_nlp_inspired_batchopt.py
├── run_lcovt_full_G1_G5_trajectory_21step_batchopt.py
├── evaluate_generated_6ai_newvae.py
└── ResidualMLP_PlusPlus_384/              # DDS 评分器及其推理参数
```

## 环境

建议使用 Python 3.10 或 3.11。先安装与本机 CUDA 对应的 PyTorch / torchvision，再安装其他依赖：

```bash
pip install -r requirements.txt
```

数据准备、训练和推理脚本均提供命令行参数；可先运行 `python <脚本名> --help` 查看当前参数，再把授权使用的数据路径传给脚本。训练和生成输出应写入本地 `runs/`、`results/` 或 `outputs/` 目录，这些目录由 `.gitignore` 排除。

## 本次发布的模型文件

- VAE 最佳检查点 `vae_best.pt`：作为 GitHub Release 附件提供。下载后放入 `weights/vae_best.pt`，用于编码和解码潜变量。
- [下载 VAE 权重](https://github.com/ali030202/SPIDER-Grade-Blind-DDS/releases/download/v1.0.0/vae_best.pt)。
- `ResidualMLP_PlusPlus_384/checkpoints/model_residual_mlp_plusplus.pt`：已保留在仓库中的 DDS 评分器权重；配套的 `feature_normalizer.npz` 与 `calibration.json` 一并保留。

源目录中没有找到已保存的 ordinal latent diffusion 检查点，因此本仓库提供该扩散模型的训练与生成代码，但不声称包含可直接加载的扩散权重。需要使用本地数据重新训练该部分。

## 数据与研究使用说明

仓库不包含 MRI 图像、掩膜、患者清单/标签 CSV、潜变量缓存、LCOVT 生成轨迹、预测结果、实验日志或报告。请自行准备有权使用的数据。该项目用于研究，不是临床诊断工具；生成轨迹排序或评分关联不能单独证明临床有效性。
