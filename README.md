# Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-training

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2512.02652) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yhj137.github.io/pianist-transformer-demo/)

English | [中文](/docs/README_zh.md)

This is the official implementation for the paper **"Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training"**.

Our work demonstrates that by pre-training on billions of tokens of unlabeled MIDI data, a model can learn a profound understanding of musical structures. This knowledge can then be transferred to the downstream task of expressive performance generation, producing piano performances that are statistically indistinguishable from those of human pianists in subjective listening tests.

## Key Features

- **Unified Data Representation**: Eliminates the distinction between score MIDI and performance MIDI, allowing massive amounts of unlabeled performance data to be directly used for pre-training.
- **Efficient Asymmetric Architecture**: Employs an asymmetric structure with a 10-layer Encoder and a 2-layer Decoder, combined with encoder sequence compression, to achieve a `2.1x` inference speedup while maintaining powerful modeling capabilities. Enables efficient inference on CPUs.
- **Scalable Training Paradigm**: A two-stage "pre-train, fine-tune" workflow that effectively overcomes the bottleneck of scarce supervised data.
- **Human-Level Generation Quality**: Pianist Transformer achieves State-Of-The-Art (SOTA) results in both objective and subjective evaluations, and is statistically indistinguishable from human pianists in subjective blind listening tests.

## Environment Dependencies
This project has been tested under the following environment configurations. We highly recommend using `conda` or `venv` to create an isolated virtual environment to avoid conflicts with existing packages on your system.

### Software Requirements
-   **Python**: `3.11`
-   **PyTorch**: `2.7.1`
-   **Transformers**: `4.54.0`
-   **CUDA**: `11.8` or higher

Our core dependencies include `PyTorch`, `Transformers`, `Accelerate`, and `miditoolkit`. For a complete list of dependencies, please see the `requirements.txt` file in the root directory.

### Hardware Requirements
-   **Inference**: A standard CPU is sufficient, though a GPU will accelerate the process.
-   **Training**: To efficiently reproduce our training process, we recommend using **4 x NVIDIA GeForce RTX 4090** GPUs or equivalent.

## Access Our Models

You can directly obtain our model checkpoints from **HuggingFace** and **ModelScope**.  
We provide two types of models: **pre-trained models** and **fine-tuned models (ready for inference)**.

| Model | HuggingFace | ModelScope | Parameters |
|-------|-------------|------------|------------|
| pianist-transformer-base | [Link](https://huggingface.co/yhj137/pianist-transformer-base) | [Link](https://www.modelscope.cn/models/yhj137/pianist-transformer-base/) | 135M |
| pianist-transformer-rendering | [Link](https://huggingface.co/yhj137/pianist-transformer-rendering) | [Link](https://www.modelscope.cn/models/yhj137/pianist-transformer-rendering/) | 135M |

## Quick Start
Follow the steps below to set up the environment and generate your first expressive piano performance with Pianist Transformer in under 5 minutes.

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/yhj137/PianistTransformer.git
cd PianistTransformer
```
### 2. Set Up the Environment
We highly recommend using `conda` to create an isolated virtual environment for this project.

```bash
# 1. Create and activate the conda environment
conda create -n pianist-transformer python=3.11
conda activate pianist-transformer

# 2. Install PyTorch
# The installation command for PyTorch may vary depending on your OS and CUDA version.
# Here are some examples for common configurations:

# For Linux/Windows (NVIDIA GPU with CUDA 12.8):
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# For Linux/Windows (NVIDIA GPU with CUDA 11.8):
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# For macOS or CPU-only:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# 3. Install other dependencies
pip install -r requirements.txt
```

### 3. Download the Model Weights
We have uploaded the fine-tuned model to the HuggingFace and ModelScope for easy access.

```bash
# from the HuggingFace
python -m src.utils.download_model --source huggingface

# or from ModelScope (for Chinese Users)
# python -m src.utils.download_model --source modelscope
```

*Note: The model file is approximately 270 MB. After downloading, please ensure the file `generation_config.json`, `config.json`, `model.safetensors` is located in the `models/sft/` directory.*

### 4. Run the Inference Script

You're all set! We provide a convenient inference script that will automatically process an example score.

```bash
sh script/inference.sh
```

After the rendering is complete, you will find the generated MIDI file `0.mid` in the `data/midis/testset/inference` directory. You can load it into your favorite DAW with a piano plugin for playback and editing.

## Training from Scratch
This section is for researchers and developers who wish to reproduce our full training pipeline or train the Pianist Transformer on a custom dataset.

To help you verify and understand the entire process, **we have included a minimal, ready-to-use dataset within this repository**. Simply follow the steps below to execute the scripts for the full pipeline, from data processing to final model fine-tuning.

### 1. Data Preprocessing

The provided script will automatically process the minimal dataset included in the repository, converting it into the format required for model training. To process the SFT data, you need to install the note-alignment tool. Please place the [Symbolic Music Alignment Tool](https://midialignment.github.io/demo.html)￼ under `./tools`, with the directory structure `./tools/AlignmentTool/*`, and then run the `compile.sh` script inside it to compile and install. After that, run the following script:

```bash
sh script/data_process.sh
```
Once this process is complete, the processed data will be located in the `data/processed` directory, ready for the subsequent training steps.

### 2. Self-Supervised Pre-training

In this stage, the model undergoes large-scale pre-training on the processed unlabeled data to learn general musical structures and knowledge.

```bash
sh script/pretrain.sh
```
⚠️ **Note**: Even with the minimal dataset, pre-training can be time-consuming. Training logs and model checkpoints will be saved to the `models/pretrain/` directory by default.

### 3. Supervised Fine-tuning

Finally, we take the pre-trained model and fine-tune it on paired (score-performance) data to teach the model how to generate expressive performances.

```bash
sh script/sft.sh
```
After fine-tuning is complete, the final model, ready for inference, will be saved in the `models/sft/` directory.

### Using Your Own Data

If you wish to train the model on your own dataset, please organize your files according to the example data structure found in the `data/` directory.

You will need to modify the `script/data_process.sh` script accordingly or write your own data processing workflow by referencing its logic to ensure your data is loaded and processed correctly.

## Graphical User Interface (GUI)
To make our tool accessible to everyone, especially users who are not familiar with the command line, we have developed a simple and intuitive Graphical User Interface (GUI) based on PyQt and Pygame. You can easily use the Pianist Transformer without writing any code.

### How to Launch

First, ensure you have installed all dependencies, including `PyQt5` and `pygame`. Then, run the following command:
```bash
python -m src.gui.ui
```
The GUI window will launch automatically.

### Interface and Features

<p align="center">
  <img src="assets/gui.png" width="800"/>
  <br>
</p>

Our GUI is divided into three main sections: **Control & Parameters**, **Status Display**, and **Result Actions**.

**Workflow:**

1.  **Load Score (`Load MIDI`)**: Click the `Load MIDI` button on the right side of the interface to select the score MIDI file you want to render.

2.  **Adjust Generation Parameters**:
    *   **`Temperature`**: Controls the randomness of the generated output. A higher value leads to a more "improvisational" feel, while a lower value makes the result more stable and deterministic.
    *   **`Top-p`**: A more advanced sampling strategy that controls the diversity of the output. It is generally recommended to keep the default value.

3.  **Start Rendering**:
    Once the parameters are set, click `Render` or `Render Again` to start the process. The circular progress bar in the center (`Rendering...`) will show the real-time progress. You can click `Cancel` at any time to terminate the current task.

4.  **Preview and Compare**:
    After rendering is complete, you can use the player at the top to preview the audio. You can switch between the `Original Score` and different rendered versions (`V1` - `V5`) at any time to visually compare the results.

5.  **Save the Results**:
    We offer two practical save options:
    *   **`Save Rendered MIDI`**: Saves a standard performance MIDI file, which records the absolute timing of notes with millisecond precision.
    *   **`Save Editable Rendered MIDI`**: **(Recommended)** Saves a DAW-friendly MIDI file. This option utilizes the "Expressive Tempo Mapping" technique proposed in our paper, converting all velocity and timing deviations into a dynamic tempo track. This means you can import the file into any professional music software (e.g., Logic Pro, Cubase, FL Studio), and it will automatically align to the beat grid while preserving all the expressive nuances, making it convenient for further editing and composition.

## Citation
If you find our work, code, or models helpful in your research, we would be grateful if you could cite our paper:

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
