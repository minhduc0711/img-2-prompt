import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class StepDecoder(nn.Module):
    def __init__(self, word_embed_dim, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(word_embed_dim, hid_dim, n_layers, dropout = dropout)
        self.embedding = nn.Embedding(output_dim, word_embed_dim)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, h0, c0):
        batch_size = input.shape[0]

        input = input.unsqueeze(0)
        input = self.dropout(self.embedding(input))

        output, (hn, cn) = self.rnn(input, (h0, c0))

        prediction = self.fc_out(output.squeeze(0))

        return prediction

class Decoder(pl.LightningModule):
    def __init__(self, word_embed_dim, vocab_size, hid_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.step_decoder = StepDecoder(
            word_embed_dim, vocab_size, hid_dim, n_layers, dropout)

    def forward(self, img_embed, target_words_ids=None, teacher_forcing_ratio = 1.0):
        #TODO: use clip img embeds for h0 or c0?
        batch_size = img_embed.shape[0]
        h0 = torch.zeros(self.step_decoder.n_layers, batch_size,
                         self.step_decoder.hid_dim).to(self.device)
        c0 = img_embed.unsqueeze(0).repeat(self.step_decoder.n_layers, 1, 1)

        target_words_ids = target_words_ids.T  # [trg len, batch size]
        seq_len = target_words_ids.shape[0]
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size).to(self.device)
        # First token
        # TODO: if target_words_ids is not provided, use <startoftext> token
        input = target_words_ids[0,:]

        for t in range(1, seq_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output = self.step_decoder(input, h0, c0)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = target_words_ids[t] if teacher_force else top1

        return outputs

class Seq2SeqDecoder(pl.LightningModule):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

        self.decoder = Decoder(vocab_size=clip_model.vocab_size,
                        word_embed_dim=128,
                        hid_dim=512,
                        n_layers=2,
                        dropout=0.5)

    def forward(self, imgs, target_words_ids=None):
        # Extract CLIP image embeddings
        img_embeds = self.clip_model.encode_image(imgs)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = self.decoder(img_embeds, target_words_ids)
        # shape: (seq_len, batch_size)
        pred_words_ids = torch.argmax(preds, 2)
        # Trim the words after <endoftext> token (id = 49407)
        eot_idxs = torch.argmax((pred_words_ids == 49407).to(dtype=torch.int32), 0)
        for j in range(pred_words_ids.shape[1]):
            pred_words_ids[eot_idxs[j] ,j] = 0

        pred_words_ids = pred_words_ids.T
        pred_text_embeds = self.clip_model.encode_text(pred_words_ids)
        true_text_embeds = self.clip_model.encode_text(target_words_ids)
        return pred_words_ids, pred_text_embeds, true_text_embeds

    def training_step(self, batch, batch_idx):
        imgs, texts = batch

        pred_words_ids, pred_text_embeds, true_text_embeds = \
            self(imgs, texts)

        loss = F.mse_loss(pred_text_embeds, true_text_embeds)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
