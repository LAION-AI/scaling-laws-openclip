import os
import pandas as pd
import argparse
from huggingface_hub import hf_hub_download
df = pd.read_csv("trained_models_info.csv")
parser = argparse.ArgumentParser(description="Script to download pre-trained models")
parser.add_argument("--samples_seen", type=str, nargs="+", default=["3B", "13B", "34B"])
parser.add_argument("--dataset", type=str, nargs="+",default=["80M", "400M", "2B"])
parser.add_argument("--model", type=str, nargs="+", default=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "ViT-g-14"])
args = parser.parse_args()
for model in args.model:
    for samples_seen in args.samples_seen:
        for dataset in args.dataset:
            res = df[(df.arch==model) & (df.samples_seen_pretty==samples_seen)&(df.data==dataset)]
            if len(res) == 1:
                filename = res.name.tolist()[0]
                hf_hub_download("laion/scaling-laws-openclip", filename, cache_dir=".", force_filename=filename)

