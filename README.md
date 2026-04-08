<div align="center">

<h1> MoRE: Mixture-of-Retrieval Experts for Reasoning-Guided Multimodal Knowledge Exploitation </h1>


<h5 align="center"> 

<a href='https://arxiv.org/abs/2505.22095'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/hmhm1229/MoRE'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='https://huggingface.co/hmhm1229/MoRE-3B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>

Chunyi Peng<sup>1</sup>,
Zhipeng Xu<sup>1</sup>,
Zhenghao Liu<sup>1</sup>,
Yishan Li<sup>3</sup>,
Yukun Yan<sup>2</sup>,
Yu Gu<sup>1</sup>
Minghe Yu<sup>1</sup>
Ge Yu<sup>1</sup>
Maosong Sun<sup>2</sup>

<sup>1</sup>Northeastern University, <sup>2</sup>Tsinghua University, <sup>3</sup>ModleBest Inc.

<h5 align="center"> If you find this project useful, please give us a star🌟.
</h5>
</div>

## News
2026.04.03 Our work is accepted by SIGIR 2026 🎉🎉🎉!

2025.08.22 We upload [MoRE-3B](https://huggingface.co/hmhm1229/MoRE-3B).

## Environment
For training, answer generation, and evaluation processes:
```bash
conda create -n router python=3.11
conda activate router
pip install requirements_router.txt
```
For retriever and corpus construction processes:
```bash
conda create -n retriever python=3.11
conda activate retriever
pip install requirements_retriever.txt
```

## Corpora Construction
For the text corpus, you can download `enwiki-20241020` from [Huggingface](https://huggingface.co/datasets/hmhm1229/enwiki-20241020). Then preprocess, and index it with the following commands:
```bash
7z x enwiki-20241020-pages-articles-multistream.xml.zip.001 
conda activate retriever
wikiextractor enwiki-20241020-pages-articles-multistream.xml.bz2 -o wiki_extracted
python wiki_preprocess.py
```
For the image corpus, you can directly download [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR). To embed and index it you can follow the [repository](https://github.com/TIGER-AI-Lab/UniIR)

For the table corpus, you can download, embed, and index Open-WikiTable following the [repository](https://github.com/sean0042/Open_WikiTable), or you can download directly the one we have already preprocessed from [here](https://huggingface.co/hmhm1229/table-retriever). 

## Retrievers Preparation
For the Text-Image Retriever, you can directly download [UniIR](https://huggingface.co/TIGER-Lab/UniIR)

For the Table Retriever, you can train it with the help of [repository](https://github.com/sean0042/Open_WikiTable), or you can download it directly from [here](https://huggingface.co/hmhm1229/table-retriever). 

## Datasets
We have prepared all the text datasets in `./datasets`, for images you need to download them from:
- `InfoSeek:` InfoSeek images can be downloaded from [OVEN](https://github.com/open-vision-language/oven/tree/main/image_downloads)
- `Dyn-VQA:` Dynamic VQA images can be downloaded from [DynVQA_en.202412](https://github.com/Alibaba-NLP/OmniSearch/blob/main/dataset/DynVQA_en/DynVQA_en.202412.jsonl)
- `WebQA:` WebQA images can be downloaded from [Google Drive](https://drive.google.com/drive/folders/19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb)

## Training
If you do not want to train the model, you can download [MoRE](https://huggingface.co/hmhm1229/R1-Router) and skip this section to [Evaluation](#evaluation)
### Data Synthesis
If you want to use the ready-to-use synthetic data directly, you can skip this section to [Step-GRPO Training](#step-grpo-training)

First, we need to synthesize the data step by step:
```bash
bash src/data_synthesis/data_synthesis.sh
```
### Step-GRPO Training
Our training framework is based on [EasyR1](https://github.com/hiyouga/EasyR1). All you need to do is download it and replace some files with the files in `./Easy-R1`.
Then start training with the command:
```bash
conda activate router
bash examples/run_qwen2_5_vl_7b_stepgrpo.sh
```
## Evaluation
We provide the evaluation pipeline for the R1-Router:
```bash
bash evaluation.sh
```
or, you can just evaluate the results we have provided by:
```bash
conda activate router
cd src
python evaluate.py --dataset_name all --method "r1-router3"
```

## Acknowledgement 
Our work is built on the following codebases, and we are deeply grateful for their contributions.
- [EasyR1](https://github.com/hiyouga/EasyR1)
- [UniIR](https://huggingface.co/TIGER-Lab/UniIR)
- [Open-WikiTable](https://github.com/sean0042/Open_WikiTable)
- [OmniSearch](https://github.com/Alibaba-NLP/OmniSearch)

## Citation
We appreciate your citations if you find our paper relevant and useful to your research!
```
@article{peng2025mixture,
  title={Mixture-of-Retrieval Experts for Reasoning-Guided Multimodal Knowledge Exploitation},
  author={Peng, Chunyi and Xu, Zhipeng and Liu, Zhenghao and Li, Yishan and Yan, Yukun and Wang, Shuo and Gu, Yu and Yu, Minghe and Yu, Ge and Sun, Maosong},
  year={2025}
  url={https://arxiv.org/abs/2505.22095}, 
}
```

## Contact Us
If you have questions, suggestions, or bug reports, please email us. We will try our best to help you.
```
hm.cypeng@gmail.com
```
