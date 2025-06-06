# MiniGPT4-Finetuning

**注意：**

1. 推荐使用linux服务器
2. 由于需要使用较大的CUDA显存，建议使用GPU配置较高的服务器
3. 下载预训练模型和数据集需要从huggingface使用git lfs下载大文件，请确保你有sudo权限或者git-lfs可用

## conda环境配置

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
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

## 训练

## 评估
