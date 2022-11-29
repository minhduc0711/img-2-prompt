import clip

from src.models import Seq2SeqDecoder
from src.data import DiffusionDBDataset

clip_model, preprocess = clip.load("ViT-B/32")

ds = DiffusionDBDataset(img_transform=preprocess, subset_name="large_first_1k")
train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)

model = Seq2SeqDecoder(clip_model)

trainer = pl.Trainer(max_epochs=5, accelerator="gpu")
trainer.fit(model=model, train_dataloaders=train_dataloader)
