#import "ZJU-Project-Report-Template/template.typ": *

#show: project.with(
  theme: "project",
  course: "自然语言处理导论",
  title: "MiniGPT4-Finetuning 项目报告",
  semester: "2025 春夏",
  author: "夏子渊 金裕涵 林滨 程韬 潘越",
  name: "夏子渊 金裕涵 林滨 程韬 潘越",
  date: "2025-06-20",
  college: "计算机科学与技术学院",
  teacher: "汤斯亮",
  place: "浙江大学",
  language: "zh",
  table_of_contents: true,
)

= 1.摘要
MiniGPT4-Finetuning 项目旨在基于视觉语言模型 MiniGPT-4，对 #strong[Flickr30k] 数据集开展指令微调，以增强图像-文本理解与生成能力。本文档总结了项目背景、环境配置、模型下载、微调流程、评估结果与未来工作等内容，为后续复现与迭代提供参考。

= 2. Project Introduction
MiniGPT-4 通过结合视觉编码器与 Vicuna 语言模型，实现了图像到文本的高质量对齐。为了进一步提升其在中文场景下的表现，我们开展了面向 Flickr30k 数据集的指令微调。

实验环境:
- 操作系统: Ubuntu 18.04 LTS
- 显卡: NVIDIA GeForce RTX 3090 $times$ 7
- Python: 3.10.13
- CUDA: 11.7
- PyTorch: 2.0.1


= 3. Technical Details
== 理论知识
=== 指令微调
指令微调是一种在预训练语言模型上进行微调的方法，旨在让模型学习到更多的指令相关知识。

=== 视觉语言模型
视觉语言模型是一种结合了视觉和语言的模型，它能够将图像和文本进行联合建模，从而实现图像到文本的生成。
=== 评估指标
==== BLEU
BLEU (Bilingual Evaluation Understudy) 是一种广泛使用的机器翻译评估指标。它通过比较生成文本与参考文本之间的n-gram重叠程度来计算分数。BLEU分数范围从0到1,1表示完全匹配。主要特点:
- 计算n-gram精确率
- 使用简短惩罚因子
- 支持多个参考文本
- 对词序敏感

==== CIDEr 
CIDEr (Consensus-based Image Description Evaluation) 是专门为图像描述任务设计的评估指标。它的主要特点包括:
- 使用TF-IDF加权来强调重要词汇
- 考虑n-gram的共识程度
- 对罕见但准确的描述给予更高权重
- 分数范围从0到10,越高越好

==== ROUGE-L
ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence) 是一种基于最长公共子序列的评估指标。其特点:
- 不要求严格连续匹配
- 考虑词序信息
- 计算召回率和精确率
- 对句子结构敏感
- 分数范围从0到1



== 技术细节
=== 转换数据集
我们使用 `prepare_flickr30k.py` 脚本将 Flickr30k 数据集转换为适合指令微调的格式。该脚本读取原始的 `flickr_annotations_30k.csv` 文件，并生成一个 JSON 文件，其中每个样本包含图像 ID 和对应的文本描述。

=== 定义测评指标
我们使用上述 BLEU、CIDEr 和 ROUGE-L 等指标评估模型性能。

在自定义的 `eval_scripts/eval_flickr30k.py` 中，我们利用 `COCO API` 实现了上述评估指标的计算。


= 4. Experiment Results
==  Conda 环境
```bash
conda env create -f environment.yml
conda activate minigptv
```

== 预训练模型获取
下载 Vicuna-7B 语言模型：
```bash
git clone https://huggingface.co/Vision-CAIR/vicuna-7b
cd vicuna-7b && git lfs pull
```
下载 MiniGPT-4 视觉-语言模型权重，并配置 `train_configs/minigpt4_flickr_finetune.yaml`。

== 数据准备
执行脚本生成注解文件：
```bash
python prepare_flickr30k.py
```

== 指令微调
使用单卡 GPU 训练线性映射层：
```bash
torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigpt4_flickr_finetune.yaml
```

训练过程部分如图所示：
#image("../train_result/terminal_1.png",width: 70%)
#image("../train_result/terminal_2.png",width: 70%)
#image("../train_result/terminal_3.png",width: 70%)
#image("../train_result/terminal_4.png",width: 70%)


训练结果部分如图所示：
#table(
  columns: 2,
  [epoch0: ],[#image("../train_result/epoch_0.png")],
  [epoch1: ],[#image("../train_result/epoch_1.png")],
  [epoch2: ],[#image("../train_result/epoch_2.png")],
  [epoch3: ],[#image("../train_result/epoch_3.png")],
  [epoch4: ],[#image("../train_result/epoch_4.png")],
)

== 评估
运行如下脚本在多模型间对比性能：
```bash
bash evaluate.sh
```

== 评估结果可视化
下表汇总 `eval_result` 目录中的全部评估图片：  
#table(
  columns: 4,
  [pretrained: ],[#image("../eval_result/pretrained_metric.png")],
  [epoch0: ],[#image("../eval_result/epoch0_metric.png")],
  [epoch1: ],[#image("../eval_result/epoch1_metric.png")],
  [epoch2: ],[#image("../eval_result/epoch2_metric.png")],
  [epoch3: ],[#image("../eval_result/epoch3_metric.png")],
  [epoch4: ],[#image("../eval_result/epoch4_metric.png")],
)

// 生成折线图后，取消下一行注释即可显示
以直观的折线图表示为：
 #image("../eval_result/metric_trends.png", width: 80% )

== 结果与分析

=== 模型性能持续提升
随着训练轮次（Epoch）的增加，所有评价指标均呈现稳定上升趋势，表明模型通过迭代学习有效捕捉了文本生成任务的核心规律。其中：
- ​​语义与连贯性优化显著​​：CIDEr和ROUGE-L的增速远超其他指标（详见图表斜率），说明模型在生成内容的语义相关性、上下文连贯性上提升最为突出，逐渐接近人类语言表达模式。
- ​​局部一致性稳步改进​​：BLEU系列指标增长平缓但持续（BLEU-1至BLEU-4增幅约50%-80%），反映模型在局部词汇匹配和短语结构的准确性上逐步完善。
=== 训练动态揭示关键拐点
- 早期快速收敛​​：pretrained至epoch1阶段所有指标快速跃升，验证预训练权重提供了高质量初始化。
- 中后期差异化优化​​：epoch2后CIDEr与ROUGE-L仍保持陡峭上升，而BLEU系列进入平缓增长期，表明模型后期更侧重于语义整体性而非局部词序精确度，符合文本生成任务的本质目标。
- 持续训练价值​​：截至epoch4，各曲线仍未出现平台期，建议扩展训练轮次（如至epoch6）以挖掘性能潜力。 
- 重点优化方向​​：可针对性设计长依赖文本和抽象语义的增强训练模块（如注意力机制改进），进一步发挥CIDEr与ROUGE-L的优势。
=== 局限性与后续工作
- 评估维度补充​​：需增加人工评价或SPICE等细粒度指标，验证模型在视觉语义对齐上的表现（若为多模态任务）。
- 泛化能力检验​​：当前结果基于单一数据集，需在跨领域数据上验证鲁棒性。
- 探索更大的语言模型后端。
- 优化推理速度与显存占用。

= References
#link("https://arxiv.org/abs/2304.10592")[[1]MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models]  // 添加超链接

#link("https://github.com/Vision-CAIR/MiniGPT-4")[[2]minigpt4 github仓库]  // 添加超链接


#link("https://huggingface.co/datasets/nlphuji/flickr30k/tree/main")[[3]Flickr30k数据集]  // 添加超链接
