import argparse
import csv
import os
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)


class DialogueDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        src_texts = []
        tgt_texts = []
        with open(os.path.join(data_dir, split + '.tsv')) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                src_texts.append(row[0])
                tgt_texts.append(row[1])
        
        self.batch = tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return self.batch['input_ids'].size(0)

    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        attention_mask = self.batch['attention_mask'][index]
        labels = self.batch['labels'][index]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''From fairseq'''
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class MT5Trainer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
        self.model = MT5ForConditionalGeneration.from_pretrained(
            'google/mt5-small'
        )

        # dataset
        self.train_dataset = DialogueDataset(
            data_dir=self.hparams.data_dir,
            split='train',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )

    def training_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        decoder_input_ids = self.model._shift_right(labels)

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs.logits
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=lprobs,
            target=labels,
            epsilon=self.hparams.label_smoothing,
            ignore_index=pad_token_id
        )

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        decoder_input_ids = self.model._shift_right(labels)

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs.logits
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=lprobs,
            target=labels,
            epsilon=self.hparams.label_smoothing,
            ignore_index=pad_token_id
        )

        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # https://huggingface.co/blog/how-to-generate
        beam_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        preds = [
            self.tokenizer.decode(beam_output, skip_special_tokens=True)
            for beam_output in beam_outputs
        ]

        with open(os.path.join(self.hparams.output_dir, 'preds.txt'), 'a') as f:
            f.write('\n'.join(preds) + '\n')

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate
        )

        # scheduler
        num_training_steps = (
            (
                (len(self.train_dataset) - 1)
                // (self.hparams.train_batch_size * self.hparams.gpus)
            )
            // self.hparams.accumulate_grad_batches
            * self.hparams.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return [optimizer], [scheduler]

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=4
        )
        return loader

    def val_dataloader(self):
        dataset = DialogueDataset(
            data_dir=self.hparams.data_dir,
            split='val',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=4
        )
        return loader

    def test_dataloader(self):
        dataset = DialogueDataset(
            data_dir=self.hparams.data_dir,
            split='test',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=4
        )
        return loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir')
    parser.add_argument('output_dir')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)

    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)

    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=args.output_dir
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
    )

    model = MT5Trainer(args)

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
