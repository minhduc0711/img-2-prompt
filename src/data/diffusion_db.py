import numpy as np
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
import clip
from torch.utils.data import Dataset
import torch

class DiffusionDBDataset(Dataset):
    def __init__(self, img_transform=None,
                 bert_tokenizer=None,
                 subset_name="large_first_50k"):
       self.img_transform = img_transform
       self.bert_tokenizer = bert_tokenizer
       self.hf_dataset = load_dataset('poloclub/diffusiondb', subset_name)["train"]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        res = {"global_idx": idx}
        img = self.hf_dataset[idx]["image"]

        if self.img_transform:
            img = self.img_transform(img)
        else:
            img = pil_to_tensor(img)
        res["img"] = img
        prompt = self.hf_dataset[idx]["prompt"]
        clip_tokens = clip.tokenize(prompt, truncate=True).squeeze()
        res["clip_tokens"] = clip_tokens
        if self.bert_tokenizer is not None:
            bert_tokens = self.bert_tokenizer.encode(prompt,
                    padding="max_length", max_length=77, truncation=True)
            res["bert_tokens"] = torch.tensor(bert_tokens)

        return res
