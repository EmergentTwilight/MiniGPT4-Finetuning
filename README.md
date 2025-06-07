# MiniGPT4-Finetuning

**注意：**

1. 推荐使用linux服务器
2. 由于需要使用较大的CUDA显存，建议使用GPU配置较高的服务器
3. 下载预训练模型和数据集需要从huggingface使用git lfs下载大文件，请确保你有sudo权限或者git-lfs可用

## conda环境配置

```bash
git clone https://github.com/RukawaY/MiniGPT4-Finetuning.git
cd MiniGPT4-Finetuning
conda env create -f environment.yml
conda activate minigptv
```

## 下载预训练的Vicuna V0 7B模型

先切换到根目录下，然后执行以下命令：

```bash
git clone https://huggingface.co/Vision-CAIR/vicuna-7b/
cd vicuna-7b
git lfs pull
```

之后修改`MiniGPT-4/minigpt4/configs/models/minigpt4_vicuna0.yaml`中第18行的`llama_model`值为模型的路径。

## 下载预训练的MiniGPT-4 7B模型

点击[这里](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)下载预训练的MiniGPT-4 7B模型，并将其放在`MiniGPT-4/prerained_minigpt4_7b.pth`

之后修改`minigpt4_eval.yaml `中第8行的`ckpt`值为模型的路径。

## DEMO

在根目录下执行：

```bash
python demo.py --cfg-path minigpt4_eval.yaml  --gpu-id 0
```

会在`127.0.0.1:7860`启动一个网页，打开后即可看到DEMO。

## 下载微调数据集

同样在根目录下执行：

```bash
git clone https://huggingface.co/datasets/nlphuji/flickr30k/
cd flickr30k
git lfs pull
```

## 基于Flickr30k数据集的指令微调

在Flickr30k数据集上进行指令微调，主要训练连接视觉模块和语言模型的线性映射层。可以运行`finetuning.py`脚本查看如何运行。

```bash
python finetuning.py
```

**1. 准备数据集**

首先，确保按照“下载微调数据集”部分的说明，下载了Flickr30k数据集。然后在 `MiniGPT4-Finetuning` 根目录下运行数据准备脚本，以生成训练所需的注解文件：

```bash
python prepare_flickr30k.py
```

该脚本会在 `flickr30k` 目录下生成一个 `flickr30k_grounded_detail.json` 文件。

**2. 检查配置文件**

已经准备好了Flickr30k微调的配置文件。在开始训练前，请打开并检查 `train_configs/minigpt4_flickr_finetune.yaml` 文件和`minigpt4/configs/datasets/flickr30k/finetune.yaml`文件。

确保：
- `minigpt4_flickr_finetune.yaml`文件中`model.ckpt` 的值是预训练MiniGPT-4 7B模型 (`prerained_minigpt4_7b.pth`) 的正确路径。
- `finetune.yaml`文件中`datasets.flickr30k_grounded_detail.build_info` 中的 `ann_path` 和 `image_path` 指向了正确的数据集路径（推荐使用绝对路径）。

**3. 开始训练**

确认数据集和配置无误后，即可开始微调。运行以下命令将使用单个GPU开始训练。如果有多张GPU，可以修改 `nproc-per-node` 的值。

```bash
torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigpt4_flickr_finetune.yaml
```

训练完成后，微调模型的检查点将保存在配置文件中 `output_dir` 指定的目录下（默认为 `output/minigpt4_flickr_finetune`）。

## 评估

TODO