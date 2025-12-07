# Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-training

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2512.02652) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yhj137.github.io/pianist-transformer-demo/)

[English](/README.md) | 中文

这是论文 **《Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training》** 的官方实现。

我们的工作展示了，通过在百亿级 token 规模的无标签 MIDI 数据上进行自监督预训练，模型可以学习到深刻的音乐结构知识，并将其迁移到下游的表现力演奏生成任务中，生成在主观听力测试中与人类演奏家媲美的钢琴演奏。

## 主要特性

- **统一数据表示**: 无需区分乐谱MIDI和演奏MIDI，使得海量无标注的演奏数据可以直接用于预训练。
- **高效非对称架构**: 采用10层Encoder和2层Decoder的非对称结构，结合编码器序列压缩技术，在保证强大建模能力的同时，实现了`2.1x`的推理加速。可在CPU上高效推理。
- **可扩展的训练范式**: "预训练-微调"的两阶段训练流程，有效克服了监督数据稀缺的瓶颈。
- **媲美人类的生成效果**: 在客观和主观评测中，Pianist Transformer 的生成效果均达到SOTA，且在主观盲听测试中与人类钢琴家统计上无法区分。

## 环境依赖
本项目已在以下环境配置下通过测试。我们强烈建议您使用 `conda` 或 `venv` 创建独立的虚拟环境，以避免与您系统中已有的软件包产生冲突。

### 软件环境
-   **Python**: `3.11`
-   **PyTorch**: `2.7.1`
-   **Transformers**: `4.54.0`
-   **CUDA**: `11.8` 或更高版本

我们的核心依赖库包括 `PyTorch`, `Transformers`, `Accelerate` 和 `miditoolkit`等。完整的依赖列表请参见根目录下的 `requirements.txt` 文件。

### 硬件要求
-   **推理 (Inference)**: 普通CPU即可运行，GPU可加速推理。
-   **训练 (Training)**: 为了高效地复现我们的训练过程，我们推荐使用 **4 x NVIDIA GeForce RTX 4090** 或同等级别的GPU。

## 访问我们的模型

您可以直接从 HuggingFace 和 ModelScope 获取我们提供的两类模型权重：**预训练模型** 与 **微调模型（可直接用于推理）**。


| 模型 | HuggingFace | ModelScope | 参数量 |
|---------|----------|-------------|------------|
| pianist-transformer-base | [链接](https://huggingface.co/yhj137/pianist-transformer-base) | [链接](https://www.modelscope.cn/models/yhj137/pianist-transformer-base/) | 135M |
| pianist-transformer-rendering | [链接](https://huggingface.co/yhj137/pianist-transformer-rendering) | [链接](https://www.modelscope.cn/models/yhj137/pianist-transformer-rendering/) | 135M |


## 快速上手
跟随以下步骤，你可以在5分钟内配置好环境，并使用 Pianist Transformer 生成你的第一首富有表现力的钢琴演奏。
### 1. 克隆代码库
首先，将本仓库克隆到你的本地设备：
```bash
git clone https://github.com/yhj137/PianistTransformer.git
cd PianistTransformer
```
### 2. 配置运行环境
我们强烈建议使用 `conda` 创建一个独立的虚拟环境来运行此项目。

```bash
# 1. 创建并激活 conda 环境
conda create -n pianist-transformer python=3.11
conda activate pianist-transformer

# 2. 安装 PyTorch
# PyTorch的安装命令因您的操作系统和CUDA版本而异。以下是一些常见配置的示例命令：

# For Linux/Windows (NVIDIA GPU with CUDA 12.8):
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# For Linux/Windows (NVIDIA GPU with CUDA 11.8):
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# For macOS or CPU-only:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 3. 下载模型权重
我们已经将微调好的模型上传至HuggingFace和ModelScope，方便您下载使用。

```bash
# 从ModelScope下载 (国内用户)
python -m src.utils.download_model --source modelscope

# 从HuggingFace下载
# python -m src.utils.download_model --source huggingface
```
*提示：模型文件约 270 MB。下载完成后，请确保 `models/sft/` 目录下有 `generation_config.json`, `config.json`, `model.safetensors` 这三个文件。*

### 4.运行推理脚本

一切准备就绪！我们提供了一个便捷的推理脚本，它会自动处理示例乐谱。

```bash
sh script/inference.sh
```

渲染完成后，您可以在项目根目录下的 `data/midis/testset/inference` 目录中找到渲染的 MIDI 文件`0.mid`。您可以使用常用的DAW软件加载钢琴音源进行播放和编辑。

## 自己训练
本章节面向希望复现我们完整训练流程，或在自定义数据集上训练 Pianist Transformer 的研究者和开发者。

为了方便您验证和理解整个流程，**我们在仓库中内置了一份可以直接运行的最小化数据集**。您只需按照以下步骤执行相应脚本，即可完成从数据处理到最终模型微调的全过程。

### 1.数据预处理

我们提供的脚本将自动处理仓库中自带的最小数据集，将其转换为模型训练所需的格式。处理SFT数据需要安装音符对齐工具，请将 [Symbolic Music Alignment Tool](https://midialignment.github.io/demo.html)，放置在`./tools`下，目录结构为`./tools/AlignmentTool/*`，然后运行其中的`compile.sh`编译安装。完成后，运行如下脚本: 

```bash
sh script/data_process.sh
```
此过程完成后，处理好的数据将位于 `data/processed` 目录下，为后续的训练步骤做好准备。

### 2.自监督预训练

此阶段将在处理好的无标签数据上进行大规模预训练，让模型学习音乐的通用结构和知识。

```bash
sh script/pretrain.sh
```
⚠️ **注意**: 即使使用最小数据集，预训练也可能需要较长时间。训练日志和模型检查点将默认保存在 `models/pretrain/` 目录下。

### 3.监督微调

最后，我们使用预训练好的模型，在成对的（乐谱-演奏）数据上进行微调，教会模型如何生成富有表现力的演奏。

```bash
sh script/sft.sh
```
微调完成后，最终可用于推理的模型将保存在 `models/sft/` 目录下。

### 关于使用您自己的数据

如果您希望使用自己的数据集进行训练，请参考 `data/` 目录下的示例数据结构来组织您的文件。

您需要相应地修改 `script/data_process.sh` 脚本，或参考其中的代码逻辑来编写您自己的数据处理流程，以确保您的数据能被正确地加载和处理。

## 使用图形用户界面（GUI）
为了方便所有用户，尤其是那些不熟悉命令行的朋友，我们基于 PyQt 和 Pygame 开发了一个简单直观的图形用户界面 (GUI)。您无需编写任何代码，即可轻松使用 Pianist Transformer。

### 如何启动

首先，请确保您已经安装了包括 `PyQt5` 和 `pygame` 在内的所有依赖。然后运行：
```bash
python -m src.gui.ui
```
运行此命令后，GUI 窗口将自动弹出。

### 界面与功能介绍

<p align="center">
  <img src="/assets/gui.png" width="800"/>
  <br>
</p>

我们的GUI主要分为三个区域：**控制与参数区**、**状态显示区** 和 **结果操作区**。

**使用流程:**

1.  **加载乐谱 (`载入 midi`)**: 点击界面右侧的 `载入 midi` 按钮，选择您想要渲染的乐谱 MIDI 文件。

2.  **调整生成参数**:
    *   **`Temperature`**: 控制生成结果的随机性。值越高，演奏的“即兴”感越强；值越低，结果越稳定和重复。
    *   **`Top-p`**: 一种更先进的采样策略，用于控制结果的多样性。通常建议保持默认值。

3.  **开始渲染**:
    参数调整完毕后，点击`开始渲染`或者`再次渲染`即可开始渲染。中间的圆形进度条 (`正在渲染...`) 会实时显示渲染进度。您可以随时点击 `取消渲染` 来终止当前任务。

4.  **试听与对比**:
    渲染完成后，您可以使用顶部的播放器进行试听。您可以随时在`原乐谱`和不同的渲染版本 (`V1` - `V5`) 之间切换，以直观地对比效果差异。

5.  **保存结果**:
    我们提供两种实用的保存选项：
    *   **`保存渲染mid`**: 保存为标准的演奏MIDI文件，它记录了音符精确到毫秒的绝对时值信息。
    *   **`保存可编辑渲染mid`**: **(推荐)** 保存为DAW友好的MIDI文件。此文件利用了我们论文中提出的 "Expressive Tempo Mapping" 技术，将所有速度变化换为一个动态的速度轨。这意味着您可以将此文件导入任何专业音乐软件 (如 Logic Pro, Cubase, FL Studio) 中，它会自动对齐到节拍网格上，并保留所有的表现力，方便您进行二次编辑和创作。

## 引用
如果您在您的研究中发现我们的工作、代码或模型对您有帮助，我们不胜荣幸。请考虑引用我们的论文：

```bibtex
@misc{you2025pianisttransformerexpressivepiano,
      title={Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training}, 
      author={Hong-Jie You and Jie-Jing Shao and Xiao-Wen Yang and Lin-Han Jia and Lan-Zhe Guo and Yu-Feng Li},
      year={2025},
      eprint={2512.02652},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

