<div align="center">

# LAABC: Label-Aware Adversarial Margin Learning via Soft Prompting for Robust Short Text Classification

Official PyTorch implementation of **Label-Aware Adversarial Margin Learning (AML)** for robust short-text classification. which is accepted by Expert Systems with Applications(ESWA) [paper](https://doi.org/10.1016/j.eswa.2026.133946).

[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://github.com/huggingface/transformers)
[![Paper](https://img.shields.io/badge/Paper-link-orange)](https://doi.org/10.1016/j.eswa.2026.133946)

</div>

## Overview

Short texts often provide limited semantic context, making their representations sensitive to small input perturbations. This repository implements **Adversarial Margin Learning (AML)**, which combines label-aware soft prompting with adversarial training to improve the discriminability and robustness of short-text representations.

The framework:

1. prepends natural-language label tokens to each input as soft label-aware prompts;
2. encodes the label tokens and input text with a pretrained language model;
3. applies Fast Gradient Method (FGM) perturbations to the word embeddings during training; and
4. optimizes a joint classification and adversarial margin objective that pulls the text representation toward its correct label representation and separates it from competing labels.

## Framework

<p align="center">
  <img src="./Adversarial%20Margin%20Learning/assets/model.jpg" width="100%" alt="Framework of Label-Aware Adversarial Margin Learning">
</p>

<p align="center"><em>Overview of the label-aware adversarial margin learning framework.</em></p>

## Repository Structure

```text
Adversarial-Margin-Learning/
|-- README.md
`-- Adversarial Margin Learning/
    |-- assets/
    |   `-- model.jpg
    |-- data/
    |   |-- SST2_Train.json
    |   |-- SST2_Test.json
    |   `-- ...
    |-- ad.py              # FGM and PGD adversarial perturbations
    |-- config.py          # Command-line arguments and experiment settings
    |-- data_utils.py      # Dataset loading and label-aware prompt construction
    |-- loss_func.py       # Adversarial margin loss
    |-- main.py            # Training and evaluation entry point
    |-- model.py           # Transformer classifier
    `-- requirements.txt
```

## Requirements

The author-reported experimental environment is:

- Python 3.9
- PyTorch 2.1.0
- NumPy 1.23.5
- Hugging Face Transformers 4.34.1
- CUDA-capable GPU recommended; CPU execution is also supported

> **Version note:** the bundled `requirements.txt` currently pins PyTorch 2.0.0 and leaves Transformers unpinned, while the original project README reports the versions above. Use the author-reported versions for strict reproduction, or update `requirements.txt` so the two records are consistent.

## Installation

```bash
git clone https://github.com/sushun176/Adversarial-Margin-Learning.git
cd Adversarial-Margin-Learning/"Adversarial Margin Learning"

conda create -n aml python=3.9 -y
conda activate aml
pip install torch==2.1.0 numpy==1.23.5 transformers==4.34.1 tqdm
```

Alternatively, install the repository's dependency file with `pip install -r requirements.txt` if exact replication of the author-reported environment is not required.

The pretrained `bert-base-uncased` or `roberta-base` checkpoint is downloaded automatically on first use. The current `main.py` configures a Hugging Face mirror; remove or change `HF_ENDPOINT` there if that mirror is unavailable in your environment.

## Datasets

Processed versions of the following datasets are included in the `data/` directory:

| Argument | Dataset | Task | Classes |
|---|---|---|---:|
| `sst2` | SST-2 | Sentiment classification | 2 |
| `subj` | SUBJ | Subjectivity classification | 2 |
| `trec` | TREC | Question classification | 6 |
| `pc` | Pros and Cons | Sentiment classification | 2 |
| `cr` | Customer Reviews | Sentiment classification | 2 |

Each JSON file contains records in the following format:

```json
{
  "text": "an example short text",
  "label": "positive"
}
```

To use another dataset, convert its train and test splits to this format and add the corresponding label mapping in `data_utils.py` and class count in `config.py`.

## Training and Evaluation

Run the default experiment on SST-2 with BERT:

```bash
python main.py --dataset sst2 --model_name bert 
```

Run an experiment with RoBERTa:

```bash
python main.py --dataset sst2 --model_name roberta 
```

Examples for the other included datasets:

```bash
python main.py --dataset subj --model_name bert 
python main.py --dataset trec --model_name bert 
python main.py --dataset pc   --model_name bert 
python main.py --dataset cr   --model_name bert 
```

Training and evaluation are performed in the same run. The best checkpoint is saved as `best_model.pth`, and experiment logs are written to `logs/`.


## Reproducibility Notes

- Run commands from the `Adversarial Margin Learning/` directory so the default `data/` path resolves correctly.
- Fix random seeds before reporting final results if deterministic comparisons are required.
- The implementation saves only the best-performing checkpoint from each run; rename or move it before starting another experiment if multiple checkpoints must be retained.
- Dataset licensing and original citations should be checked before redistributing processed data.

## Citation

If you find this repository useful, please cite the corresponding paper. 



```bibtex
@article{su2026laabc,
  title={LAABC: label-aware adversarial margin learning via soft prompting for robust short text classification},
  author={Su, Shun and Shao, Dangguo and Liu, Jianjian and Yu, Zhengtao and Ma, Lei},
  journal={Expert Systems with Applications},
  pages={133946},
  year={2026},
  publisher={Elsevier}
}
```


## Acknowledgements

This implementation is built with [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers), using the pretrained [BERT](https://huggingface.co/bert-base-uncased) and [RoBERTa](https://huggingface.co/roberta-base) encoders. thanks to [DualCL](https://arxiv.org/abs/2201.08702)for their great help to our codes and research. 

## Contact

For questions or issues, please use the repository's [issue tracker](https://github.com/sushun176/Adversarial-Margin-Learning/issues).
