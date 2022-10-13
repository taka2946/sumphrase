# SumPhrase
Contains code and dataset from the paper [Phrase-Level Localization of Inconsistency Errors in Summarization by Weak Supervision](https://aclanthology.org/2022.coling-1.537/), Masato Takatsuka, Tetsunori Kobayashi, Yoshihiko Hayashi, COLING 2022.

Environment base is Python 3.8. Also see requirements.txt. We used Stanford CoreNLP version 4.1.0.
# Model and Data
datasets are available at:
https://drive.google.com/drive/folders/1DKQwAVwU8HZyougcq5HUPAOUd-xbQiDI?usp=sharing

All models are available at:
https://drive.google.com/drive/folders/12afZCQnHPH7gYU4yRGhS8o8ZYoZzNCUy?usp=sharing

## Datasets
The 'train' folder contains synthetic datasets used to train sumphrase models.

'train_dae.jsonl': This file contains arc-level labels.
'train_phrase.jsonl': This file contains phrase-level labels created from the arc-level labels.

Each dataset contains four types of data - [para, fusion, comp, ref]. 
'para' means that it is created by paraphrasing the reference summary (original data provided in [this paper](https://github.com/tagoyal/factuality-datasets#factuality-models-and-data)). 'fusion' means that it is created by sentence fusion of two sentences in the original document. 'comp' means that it is created by sentence compression of the sentence in the original document. 'ref' means that it is the reference summary.

The 'test' and 'val' folder contains K2020 dataset provided in [this paper](https://github.com/salesforce/factCC).
The 'test_rerank' folder contains reranking summary dataset provided in [this paper](https://aclanthology.org/P19-1213.pdf).


# Running Code
Download the datasets from the google drive. The data has been preprocessed. 
You can train 4 types of models by setting the 'model_type' argument: 'electra_dae' (arc-level error detection which proposed in [this paper](https://github.com/tagoyal/factuality-datasets#factuality-models-and-data)), 'electra_dae_multi' (electra-dae + corresponding sentence detection task), 'sumphrase' (phrase-level error detection) or 'sumphrase_multi' (sumphrase+ corresponding sentence detection task).

## Training
Run the following command to train models:
```
 python3 train.py \
   --model_type sumphrase_multi \
   --model_name_or_path google/electra-base-discriminator \
   --do_train \
   --do_eval \
   --train_data_file /path/to/train_phrase.jsonl \
   --eval_data_file /path/to/val_phrase.jsonl \
   --per_gpu_eval_batch_size 10 \
   --per_gpu_train_batch_size 10 \
   --num_train_epochs 3.0 \
   --learning_rate 2e-5 \
   --output_dir /path/to/output_dir \
   --save_step 40 \
   --multitask_loss_weight 1.0 0.5
```

## Evaluating Models
Run the following command to use K2020 dataset:

```
python3 train.py 
  --model_type sumphrase_multi \
  --model_name_or_path /path/to/model-best/ \ 
  --do_eval \
  --eval_data_file /path/to/test_phrase.jsonl \
  --per_gpu_eval_batch_size 24 \
  --output_dir /path/to/output_dir \
  --train_data_file None
```

Run the following command to use Reranking Summary dataset:
```
python3 test_reranking_summary.py \ 
  --model_type sumphrase_multi \
  --model_name_or_path /path/to/model-best/ \
  --do_eval \
  --eval_data_file /path/to/val_sentence_pairs_phrase.jsonl \ 
  --output_dir /path/to/output_dir
```
If you want to train arc-level error detection models, use '*_dae.jsonl' data.
