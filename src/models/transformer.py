import itertools
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src * math.sqrt(self.d_model)
        # add positional encoding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class TransformerImg2Prompt(pl.LightningModule):
    def __init__(self, clip_model, bert_model, emsize: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 learning_rate=5e-4,
                 warm_up_step=4000):
        super().__init__()
        self.save_hyperparameters(ignore=["clip_model", "bert_model"])

        self.clip_model = clip_model
        self.bert_model = bert_model

        self.transformer = Transformer(30522,  # Hardcode the vocab size of BERT
                                       emsize, nhead,
                                       d_hid, nlayers, dropout)
        # To match CLIP img vec dimension with BERT embeds
        self.fc_img = nn.Linear(512, 768)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).cuda()

    def construct_input_seq(self, img, target_bert_tokens=None):
        if target_bert_tokens is not None:
            bert_embeds = self.bert_model(target_bert_tokens[:, :-1])[0]
        else:
            bert_embeds = torch.Tensor().to(img)  # empty
        img_embed = self.clip_model.encode_image(img)
        img_embed = self.fc_img(img_embed)

        # [batch, seqlen, embed_dim]
        input_seq = torch.concat([img_embed.unsqueeze(1), bert_embeds], dim=1)
        # [seqlen, batch, embed_dim]
        input_seq = input_seq.permute(1, 0, 2)
        return input_seq

    def forward(self, input_seq):
        src_mask = self.generate_square_subsequent_mask(input_seq.shape[0])
        return self.transformer(input_seq, src_mask)

    def training_step(self, batch, batch_idx):
        imgs, target_bert_tokens = batch["img"], batch["bert_tokens"]

        input_seq = self.construct_input_seq(imgs, target_bert_tokens)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = self(input_seq)
        # shape: (batch_size, vocab_size, seq_len)
        preds = torch.permute(preds, (1, 2, 0))
        targets = target_bert_tokens.long()

        loss_vals = F.cross_entropy(preds, targets, reduction="none")
        loss_vals *= targets != 0
        loss = loss_vals.mean()
        self.log("train/CE_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, target_bert_tokens = batch["img"], batch["bert_tokens"]

        input_seq = self.construct_input_seq(imgs, target_bert_tokens)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = self(input_seq)
        # shape: (batch_size, vocab_size, seq_len)
        preds = torch.permute(preds, (1, 2, 0))
        targets = target_bert_tokens.long()

        # mask loss values for [PAD] tokens
        loss_vals = F.cross_entropy(preds, targets, reduction="none")
        loss_vals *= targets != 0
        loss = loss_vals.mean()
        self.log("val/CE_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
    # def predict(self, batch):
        imgs = batch["img"]
        predictions = []
        for img in imgs:
            img = img.unsqueeze(0)
            input_seq = self.construct_input_seq(img)

            pred_tokens = []
            pred_next_token = None
            while pred_next_token != 102 and len(pred_tokens) < 77:
                next_token_proba = self(input_seq)[-1, ...]

                pred_next_token = torch.argmax(next_token_proba.squeeze())
                pred_tokens.append(pred_next_token.item())

                bert_embeds = self.bert_model(pred_next_token.view(1, 1))[0]
                input_seq = torch.concat([input_seq, bert_embeds],dim=0)

            predictions.append(pred_tokens)
        return predictions

    def configure_optimizers(self):
        train_layers = [self.transformer, self.fc_img]
        params = itertools.chain(*[layer.parameters() for layer in train_layers])
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)

        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
        using_native_amp=None
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        if self.trainer.global_step < self.hparams.warm_up_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warm_up_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
