# Reproducible scaling laws for contrastive language-image learning [[arXiv]](https://arxiv.org/abs/2212.07143)

Work still in progress. In this repository, we will provide the code for reproducing the experiments on large-scale CLIP pre-training and transfer to various downstream tasks for the paper "Reproducible scaling laws for contrastive language-image learning".

Stay tuned.

Until finalized, you may check

- the [openCLIP](https://github.com/mlfoundations/open_clip) repository that points to the pre-trained models used in this study
- the [LAION-400m](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md) and [LAION-5B](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md) composition instructions, the datasets used for openCLIP pre-training in this study
- [CLIP Benchmarking](https://github.com/LAION-AI/CLIP_benchmark), transfer evaluation used in this study

## Introduction


# Download pre-trained models

We provide a script, `download.py` to download all pre-trained models used in the paper.
To download all the 29 models used in the paper, use :

`python download_models.py`


You can also download a subset of the models. For instance:

`python download_models.py --samples_seen 3B 13B --model ViT-B-32 --data 80M 400M 2B`

will only download ViT-B/32 models with samples seen of 3B or 13B, trained on any of 80M/400M/2B LAION datasets.

# Using pre-training models in OpenCLIP

Once you download the pre-trained models, you can also use them in OpenCLIP.
Following is an example.

```bash
> python download_models.py --samples_seen 34B --model ViT-H-14 --data 2B

Downloading https://huggingface.co/laion/scaling-laws-openclip/resolve/main/Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt to Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt
```

Once the model is downloaded, it is possible to directly use it in OpenCLIP:

```python
import torch
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt')
```
See https://github.com/mlfoundations/open_clip#usage for more detailed usage of OpenCLIP.

## Citation

If you find this work helpful, please cite our paper:

```
@article{cherti2022reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  journal={arXiv preprint arXiv:2212.07143},
  year={2022}
}
```
## Acknowledgements
