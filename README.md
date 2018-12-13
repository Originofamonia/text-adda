# PyTorch-ADDA-for-Text
A PyTorch implementation of [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464) for amazon review dataset.

## Environment
- Python 3.6
- PyTorch 0.2.0

## run main

```
python main.py --seqlen 200 --patience 5 --num_epochs_pre 200 ^
	       --log_step_pre 32 --eval_step_pre 10 --save_step_pre 100 ^
	       --num_epochs 100 --log_step 32 --save_step 100
```

### arguments

```
python main.py -h

usage: main.py [-h] [--seqlen SEQLEN] [--patience PATIENCE]
               [--num_epochs_pre NUM_EPOCHS_PRE] [--log_step_pre LOG_STEP_PRE]
               [--eval_step_pre EVAL_STEP_PRE] [--save_step_pre SAVE_STEP_PRE]
               [--num_epochs NUM_EPOCHS] [--log_step LOG_STEP]
               [--save_step SAVE_STEP]

Specify Params for Experimental Setting

optional arguments:
  -h, --help            show this help message and exit
  --seqlen SEQLEN       Specify maximum sequence length (default: 200)
  --patience PATIENCE   Specify patience of early stopping for pretrain (default: 5)
  --num_epochs_pre NUM_EPOCHS_PRE
                        Specify the number of epochs for pretrain (default: 200)
  --log_step_pre LOG_STEP_PRE
                        Specify log step size for pretrain (default: 32)
  --eval_step_pre EVAL_STEP_PRE
                        Specify eval step size for pretrain (default: 10)
  --save_step_pre SAVE_STEP_PRE
                        Specify save step size for pretrain (default: 100)
  --num_epochs NUM_EPOCHS
                        Specify the number of epochs for adaptation (default: 100)
  --log_step LOG_STEP   Specify log step size for adaptation (default: 32)
  --save_step SAVE_STEP
                        Specify save step size for adaptation (default: 100)
```

## Network

In this experiment, I use three types of network. They are very simple.

- BERT encoder

```
BERTEncoder(
  (encoder): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): BertLayerNorm()
      (dropout): Dropout(p=0.1)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
	⋮
        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
)
```

- BERT classifier

```
BERTClassifier(
  (classifier): Sequential(
    (0): Dropout(p=0.1)
    (1): Linear(in_features=768, out_features=96, bias=True)
    (2): LeakyReLU(negative_slope=0.01)
    (3): Linear(in_features=96, out_features=2, bias=True)
  )
)
```

- Discriminator

```
Discriminator(
  (layer): Sequential(
    (0): Linear(in_features=768, out_features=96, bias=True)
    (1): ReLU()
    (2): Linear(in_features=96, out_features=96, bias=True)
    (3): ReLU()
    (4): Linear(in_features=96, out_features=2, bias=True)
  )
)
```
