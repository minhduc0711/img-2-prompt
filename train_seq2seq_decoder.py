import clip
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.models import Seq2SeqDecoder
from src.data import DiffusionDBDataset

clip_model, preprocess = clip.load("ViT-B/32")
# For some reason, the weights in the CLIP model are automatically converted
# to float16. We convert it back to float32
clip_model = clip_model.float()

ds = DiffusionDBDataset(img_transform=preprocess, subset_name="large_first_1k")
train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)

model = Seq2SeqDecoder(clip_model)

trainer = pl.Trainer(max_epochs=5, accelerator="gpu")
trainer.fit(model=model, train_dataloaders=train_dataloader)
