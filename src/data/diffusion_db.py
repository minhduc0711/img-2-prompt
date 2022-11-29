import numpy as np
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
import clip
from torch.utils.data import Dataset


class DiffusionDBDataset(Dataset):
    def __init__(self, img_transform=None,
                 subset_name="large_first_50k"):
       self.img_transform = img_transform
       self.hf_dataset = load_dataset('poloclub/diffusiondb', subset_name)["train"]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        img = self.hf_dataset[idx]["image"]

        if self.img_transform:
            img = self.img_transform(img)
        else:
            img = pil_to_tensor(img)

        prompt = self.hf_dataset[idx]["prompt"]
        tokens = clip.tokenize(prompt, truncate=True).squeeze()

        return img, tokens
