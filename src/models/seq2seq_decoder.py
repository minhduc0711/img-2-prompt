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
        # x = self.fc_in(word_embeds)
        # print(x.shape)
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
        self.fc_img = nn.Linear(512, hid_dim)

    def forward(self, img_embed, target_words_ids=None,
                target_bert_tokens=None,
                teacher_forcing_ratio = 1.0):
        #TODO: use clip img embeds for h0 or c0?
        batch_size = img_embed.shape[0]
        h = torch.zeros(self.step_decoder.n_layers, batch_size,
                         self.step_decoder.hid_dim).to(img_embed)
        img_embed = self.fc_img(img_embed)
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
                        hid_dim=1024,
                        n_layers=10,
                        dropout=0.5)

    def forward(self, imgs, target_words_ids=None,
                target_bert_tokens=None):
        # Extract CLIP image embeddings
        img_embeds = self.clip_model.encode_image(imgs)
        # shape: (seq_len, batch_size, self.vocab_size)
        preds = self.decoder(img_embeds, target_words_ids,
                target_bert_tokens)

        # shape: (seq_len, batch_size)
        pred_words_ids = torch.argmax(preds, 2)
        # Hardcode the <startoftext> token into the outputs
        pred_words_ids[0, :] = 49406
        #pred_words_ids[10, 0] = 49407
        # Trim the words after <endoftext> token (id = 49407)
        eot_idxs = torch.where(pred_words_ids == 49407)
        for seq_idx, eot_idx in zip(*eot_idxs):
            pred_words_ids[seq_idx, eot_idx+1 :] = 0

        pred_words_ids = pred_words_ids.T

        # pred_text_embeds = self.clip_model.encode_text(pred_words_ids)

        # Giulio approach: feed probas directly into CLIP's word embedding layer
        # instead of token IDs
        x = preds.permute(1, 0, 2) @ self.clip_model.token_embedding.weight

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), pred_words_ids.argmax(dim=-1)] @ self.clip_model.text_projection
        pred_text_embeds = x

        true_text_embeds = self.clip_model.encode_text(target_words_ids) \
            if target_words_ids is not None else None

        return pred_words_ids, pred_text_embeds, true_text_embeds

    def training_step(self, batch, batch_idx):
        imgs, target_clip_tokens, target_bert_tokens = batch

        pred_words_ids, pred_text_embeds, true_text_embeds = \
            self(imgs, target_clip_tokens, target_bert_tokens)

        loss = F.mse_loss(pred_text_embeds, true_text_embeds)
        self.log("train/mse_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, target_clip_tokens, target_bert_tokens = batch

        pred_words_ids, pred_text_embeds, true_text_embeds = \
                self(imgs, target_clip_tokens, target_bert_tokens)

        loss = F.mse_loss(pred_text_embeds, true_text_embeds)
        self.log("val/mse_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=0.001)
