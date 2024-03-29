import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class StepDecoder(nn.Module):
    def __init__(self, bert_model, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        word_embed_dim = 768
        self.rnn = nn.LSTM(word_embed_dim, hid_dim, n_layers)
            #, dropout = dropout)
        # self.embedding = nn.Embedding(output_dim, word_embed_dim)
        self.bert_model = bert_model
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, bert_tokens_t, h0, c0):
        batch_size = input.shape[0]

        input = input.unsqueeze(0)


        bert_tokens_t = bert_tokens_t.unsqueeze(0)
        word_embeds = self.bert_model(bert_tokens_t)[0]
        # print(word_embeds.shape)
        # word_embeds = self.dropout(self.embedding(input))

        output, (hn, cn) = self.rnn(word_embeds, (h0, c0))

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hn, cn

class Decoder(pl.LightningModule):
    def __init__(self, vocab_size, hid_dim, n_layers, dropout,
                 bert_model=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.step_decoder = StepDecoder(
            bert_model, vocab_size, hid_dim, n_layers, dropout)

    def forward(self, img_embed, target_words_ids=None,
                target_bert_tokens=None,
                teacher_forcing_ratio = 1.0):
        #TODO: use clip img embeds for h0 or c0?
        batch_size = img_embed.shape[0]
        h = torch.zeros(self.step_decoder.n_layers, batch_size,
                         self.step_decoder.hid_dim).to(img_embed)
        c = img_embed.unsqueeze(0).repeat(self.step_decoder.n_layers, 1, 1)

        # TODO: hardcoded 4 now
        seq_len = 77#target_words_ids.shape[0]
        # TODO: concat tensors instead of predefinining the shape of outputs
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size).to(img_embed)
        # the first token is always <startoftext>
        input = torch.ones(batch_size).to(device=self.device, dtype=torch.int) * 49406
        if target_words_ids is not None:
            target_words_ids = target_words_ids.T  # [trg len, batch size]

        bert_tokens_t = target_bert_tokens[:, 0]
        for t in range(1, seq_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, h, c = self.step_decoder(input, bert_tokens_t, h, c)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            if target_words_ids is not None and teacher_force:
                input = target_words_ids[t]
            else:
                input = top1
            bert_tokens_t = target_bert_tokens[:, t]
        # print(outputs)

        return outputs

class Seq2SeqDecoder(pl.LightningModule):
    def __init__(self, clip_model, bert_model):
        super().__init__()
        self.clip_model = clip_model
        # for param in clip_model.parameters():
        #     param.requires_grad = False

        self.decoder = Decoder(vocab_size=clip_model.vocab_size,
                        bert_model=bert_model,
                        hid_dim=512,
                        n_layers=1,
                        dropout=0.5)

    def forward(self, imgs, target_words_ids=None,
                target_bert_tokens=None):
        # Extract CLIP image embeddings
        img_embeds = self.clip_model.encode_image(imgs)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = self.decoder(img_embeds, target_words_ids,
                target_bert_tokens)
        return preds


    def training_step(self, batch, batch_idx):
        imgs, target_clip_tokens, target_bert_tokens = batch

        preds = self(imgs, target_clip_tokens, target_bert_tokens)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = preds[1:, :, :]  # ignore the <startoftext> token
        preds = torch.permute(preds, (1, 2, 0))
        targets = target_clip_tokens[:, 1:].long()

        loss = F.cross_entropy(preds, targets)
        self.log("train/CE_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, target_clip_tokens, target_bert_tokens = batch

        preds = self(imgs, target_clip_tokens, target_bert_tokens)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = preds[1:, :, :]  # ignore the <startoftext> token
        preds = torch.permute(preds, (1, 2, 0))
        targets = target_clip_tokens[:, 1:].long()

        loss = F.cross_entropy(preds, targets)
        self.log("val/CE_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=0.001)
