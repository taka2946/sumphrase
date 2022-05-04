import torch
import os
import csv
from torch.utils.data import TensorDataset
import json
from tqdm import tqdm
from logging import getLogger, StreamHandler, Formatter, DEBUG

from my_dataset import MyDataset


def setup_logger(modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


logger = setup_logger(__name__)


def read_jsonl(input_file):
    lines = []
    with open(input_file) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def pad_1d(input, max_length, pad_token):
    padding_length = max_length - len(input)
    if padding_length < 0:
        input = input[:max_length]
        padding_length = 0
    input = input + ([pad_token] * padding_length)
    return input


class InputFeatures(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def convert_examples_to_features(examples, tokenizer, max_length=128, pad_token=None, num_deps_per_ex=20, input_type='arc', evaluate=False):
    features = []
    rejected_ex = 0
    
    logger.info(input_type)
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)
        tokens_input = []
        sent_idx = []
        sent_basis_label = example['sent_basis_label']
        tokens_input.extend(tokenizer.tokenize('[CLS]'))
        
        for sent in example['text_sentences']:
            index_now = len(tokens_input)
            start_idx = index_now
            for (word_index, word) in enumerate(sent.split(' ')):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens_input.extend(word_tokens)
                    index_now += len(word_tokens)
            sent_idx.append([i for i in range(start_idx, index_now)])
        
        tokens_input_more = []
        tokens_input_more.extend(tokenizer.tokenize('[SEP]'))
        tokens_input_more.extend(tokenizer.tokenize('[CLS]'))
        index_now += 2
        hypo_cls_idx = index_now - 1

        index_map = {}
        for (word_index, word) in enumerate(example['context'].split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input_more.extend(word_tokens)
                index_map[word_index] = [index_now + index for index in range(len(word_tokens))]
                index_now += len(word_tokens)
        tokens_input_more.extend(tokenizer.tokenize('[SEP]'))

        if len(tokens_input) + len(tokens_input_more) > max_length:
            extra_len = len(tokens_input) + len(tokens_input_more) - max_length
            tokens_input = tokens_input[:-extra_len]
            for w in index_map:
                index_map[w] = [index - extra_len for index in index_map[w]]
            hypo_cls_idx = hypo_cls_idx - extra_len
            
            over = False
            for i in range(len(sent_idx)):
                for j in range(len(sent_idx[i])):
                    if sent_idx[i][j] == len(tokens_input):
                        extra_idx = i
                        extra_token_idx = j
                        over = True
                        break
                if over:
                    break
            if over:
                if extra_token_idx != 0:
                    sent_idx = sent_idx[:extra_idx+1]
                    sent_idx[extra_idx] = sent_idx[extra_idx][:extra_token_idx]
                    if not evaluate:
                        sent_basis_label = sent_basis_label[:extra_idx+1]
                else:
                    sent_idx = sent_idx[:extra_idx]
                    if not evaluate:
                        sent_basis_label = sent_basis_label[:extra_idx]

        tokens_input = tokens_input + tokens_input_more

        child_indices = [[] for _ in range(num_deps_per_ex)]
        head_indices = [[] for _ in range(num_deps_per_ex)]

        mask_entail = [0] * num_deps_per_ex
        mask_cont = [0] * num_deps_per_ex
        num_dependencies = 0

        input_arcs = [[0] * 100 for _ in range(num_deps_per_ex)]
        sentence_label = int(example['sentlabel'])
        
        for i in range(num_deps_per_ex):
            if input_type == 'arc':                
                if example['dep_idx' + str(i)] == '':
                    break

                child_idx, head_idx = example['dep_idx' + str(i)].split(' ')
                child_idx = int(child_idx)
                head_idx = int(head_idx)

                label = int(example['dep_label' + str(i)])

                # take first word
                child_indices[i] = [index_map[child_idx][0]]
                head_indices[i] = [index_map[head_idx][0]]
                w1 = example['dep_words' + str(i)].split(' ')[0]
                w2 = example['dep_words' + str(i)].split(' ')[1]
            else:
                if example['child_idx' + str(i)] == '':
                    break

                child_idx = example['child_idx' + str(i)]
                head_idx = example['head_idx' + str(i)]

                label = int(example['phrase_label' + str(i)])
                child_indices_tmp = []
                head_indices_tmp = []
                
                for index in child_idx:
                    child_indices_tmp += index_map[index]
                for index in head_idx:
                    head_indices_tmp += index_map[index]
                child_indices[i] = child_indices_tmp
                head_indices[i] = head_indices_tmp
                w1 = example['child_words' + str(i)]
                w2 = example['head_words' + str(i)]

            num_dependencies += 1

            if label == 1:
                mask_entail[i] = 1
            else:
                mask_cont[i] = 1

            arc_text = w1 + ' [SEP] ' + w2
            arc = tokenizer.encode(arc_text)
            input_arcs[i] = pad_1d(arc, 100, pad_token)

        if num_dependencies == 0:
            rejected_ex += 1
            continue

        input_ids = tokenizer.convert_tokens_to_ids(tokens_input)

        assert len(tokens_input) == len(input_ids), 'length mismatched'
        padding_length_a = max_length - len(tokens_input)
        input_ids = input_ids + ([pad_token] * padding_length_a)
        input_attention_mask = [1] * len(tokens_input) + ([0] * padding_length_a)

        features.append(InputFeatures(data_type=example["type"] if not evaluate else "test",
                                      input_ids=input_ids,
                                      input_attention_mask=input_attention_mask,
                                      sentence_label=sentence_label,
                                      child_indices=child_indices,
                                      head_indices=head_indices,
                                      mask_entail=mask_entail,
                                      mask_cont=mask_cont,
                                      num_dependencies=num_dependencies,
                                      arcs=input_arcs,
                                      sent_indices=sent_idx,
                                      sent_basis_label=sent_basis_label,
                                      hypo_cls_idx=hypo_cls_idx
                                       ))
    logger.info("no labels examples: {}".format(rejected_ex))
    return features



def load_and_cache_examples(args, tokenizer, evaluate):
    if evaluate:
        data_dir = '/'.join(args.eval_data_file.split('/')[:-1])
        filename = os.path.basename(args.eval_data_file).split('.')[0]
        data_path = args.eval_data_file
    else:
        data_dir = '/'.join(args.train_data_file.split('/')[:-1])
        filename = os.path.basename(args.train_data_file).split('.')[0]
        data_path = args.train_data_file

    model_type = 'electra'
    if 'dae' in args.model_type:
        input_type = 'arc'
    else:
        input_type = 'phrase'

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            filename,
            model_type,
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = read_jsonl(data_path)
            
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            input_type=input_type,
            evaluate=evaluate
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info("Saved")

    if not evaluate and args.dataset is not None:
        features = [f for f in features if f.data_type in args.dataset]

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    input_attention_mask = [f.input_attention_mask for f in features]

    child_indices = [f.child_indices for f in features]
    head_indices = [f.head_indices for f in features]

    mask_entail = [f.mask_entail for f in features]
    mask_cont = [f.mask_cont for f in features]

    num_dependencies = [f.num_dependencies for f in features]
    arcs = [f.arcs for f in features]

    sentence_label = [f.sentence_label for f in features]
    
    sent_indices = [f.sent_indices for f in features]
    sent_basis_label = [torch.tensor(f.sent_basis_label, dtype=torch.long) for f in features]
    hypo_cls_idx = [f.hypo_cls_idx for f in features]
    
    dataset = MyDataset(all_input_ids, input_attention_mask, child_indices, head_indices, mask_entail, mask_cont, 
                            num_dependencies, arcs, sentence_label, sent_indices, sent_basis_label, hypo_cls_idx, args.max_seq_length)
    return dataset