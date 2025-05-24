# Reproducible scaling laws for contrastive language-image learning [[arXiv]](https://arxiv.org/abs/2212.07143)

*by Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, Jenia Jitsev* [[arXiv:2212.07143]](https://arxiv.org/abs/2212.07143) (Accepted at [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Cherti_Reproducible_Scaling_Laws_for_Contrastive_Language-Image_Learning_CVPR_2023_paper.html))

In this repository, we provide the code for reproducing the experiments on large-scale CLIP pre-training and transfer to various downstream tasks for the paper "Reproducible scaling laws for contrastive language-image learning", together with pointers to open artefacts from this study (pre-trained models and all the intermediate checkpoints).

## Artefacts
Pre-trained openCLIP models across various scales, final checkpoints: https://huggingface.co/laion/scaling-laws-openclip/tree/main \
All intermediate checkpoints for the pre-trained models (useful for studying training dynamics): https://huggingface.co/laion/scaling-laws-openclip/tree/main/full_checkpoints 

## Related resources
You may also check other related resources:

- the [OpenCLIP](https://github.com/mlfoundations/open_clip) repository that points to the pre-trained models used in this study
- the [LAION-400m](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md) and [LAION-5B](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md) composition instructions, the datasets used for openCLIP pre-training in this study
- The instructions are also valid for the updated safety revision of those datasets, [Re-LAION](https://laion.ai/blog/relaion-5b/). Follow [Re-LAION-research](https://huggingface.co/collections/laion/re-laion-5b-research-67e312387d2a4f879c4920b1) or [Re-LAION-research-safe](https://huggingface.co/collections/laion/re-laion-5b-research-safe-67e311013ba899a938569e32) to obtain [Re-LAION](https://laion.ai/blog/relaion-5b/) datasets for openCLIP experiments. 
- [CLIP Benchmarking](https://github.com/LAION-AI/CLIP_benchmark), transfer evaluation used in this study

## Introduction

## Scaling plots

To reproduce scaling plots from the paper, see the [figures](figures.ipynb) notebook.

## Download pre-trained models

First, you need to clone the repo and install the requirements.

```bash
git clone https://github.com/LAION-AI/scaling-laws-openclip
cd scaling-laws-openclip
pip install -r requirements.txt
```

We provide a script, `download_models.py`, to download all pre-trained models used in the paper.
To download all the 29 models used in the paper, use :

```bash
python download_models.py
```

You can also download a subset of the models. For instance:

```bash
python download_models.py --samples_seen 3B 13B --model ViT-B-32 --data 80M 400M 2B
```

will only download ViT-B/32 models with samples seen of 3B or 13B, trained on any of 80M/400M/2B LAION datasets.

## Using pre-training models in OpenCLIP

Once you download the pre-trained models, you can also use them in OpenCLIP.
Following is an example with ViT-H/14.

First, you need to download the model:

```bash
> python download_models.py --samples_seen 34B --model ViT-H-14 --data 2B

'Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt' downloaded.
```

Once the model is downloaded, it is possible to directly use it in OpenCLIP:

```python
import torch
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt')
```

For a complete example, see the [inference](inference.ipynb) notebook.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2818--2829},
  year={2023}
}
```
## Acknowledgements
