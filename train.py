# Copyright (c) 2019, Salesforce.com, Inc.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# * Neither the name of Salesforce.com nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import json
import logging
import os
import random
from typing import Dict, Tuple
import numpy as np
from train_utils import load_and_cache_examples
import model
from my_dataset import collate_fn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from sklearn.utils.extmath import softmax
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from train_utils import setup_logger
from transformers import (
    AdamW,
    ElectraConfig,
    ElectraTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

logger = setup_logger(__name__)

MODEL_CLASSES = {
    "electra_dae": (ElectraConfig, model.ElectraDependencyModel, ElectraTokenizer),
    "electra_dae_multi": (ElectraConfig, model.ElectraDependencyMultiTaskModel, ElectraTokenizer),
    "sumphrase": (ElectraConfig, model.SumPhraseModel, ElectraTokenizer),
    "sumphrase_multi": (ElectraConfig, model.SumPhraseMultiTaskModel, ElectraTokenizer),
    }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_checkpoints(args, output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def compute_metrics_balanced(preds, golds):
    n_0 = 0.
    d_0 = 0.
    n_1 = 0.
    d_1 = 0.
    for p, g in zip(preds, golds):
        if g == 0:
            if p == 0:
                n_0 += 1
            d_0 += 1
        elif g == 1:
            if p == 1:
                n_1 += 1
            d_1 += 1

    acc_0 = n_0 / d_0
    acc_1 = n_1 / d_1

    return {'acc': (acc_0 + acc_1) / 2}


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="", save_file=False) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids_sent = None
    sent_basis_preds = None
    model.eval()
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
            mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
            sent_labels = batch[8]
            sent_indices, sent_basis_label, hypo_cls_idx =  batch[9],  batch[10], batch[11]

            inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                      'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                      'num_dependency': num_dependency, 'sent_label': sent_labels, 
                      'sent_indices': sent_indices, 'sent_basis_label': sent_basis_label,
                      'hypo_cls_idx':  hypo_cls_idx, 'device': args.device}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids_sent = sent_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids_sent = np.append(out_label_ids_sent, sent_labels.detach().cpu().numpy(), axis=0)
        if len(outputs) > 2:
            if sent_basis_preds is None:
                sent_basis_preds = outputs[2].detach().cpu().tolist()
            else:
                sent_basis_preds.extend(outputs[2].detach().cpu().tolist())

    if save_file:
        f_out = open(os.path.join(eval_output_dir, 'dev_out.txt'), 'w')
    else:
        f_out = None
    k = 0
    sent_pred = []
    nb_eval_steps = 0
    for batch in eval_dataloader:
        nb_eval_steps += 1
        for inp, p_mask, arc_list, sent_indices in zip(batch[0], batch[4], batch[7], batch[9]):
            if f_out is not None:
                padding_idx = len(inp)

                tokens = tokenizer.convert_ids_to_tokens(inp)
                article_len = tokens.index('[SEP]') + 1
                text_article = tokens[1:article_len - 1]  # removing [CLS] and [SEP]


                summary = tokens[article_len+1:]  # has all the pad tokens also
                if '[PAD]' in summary:
                    summary_len = summary.index('[PAD]')
                    summary = summary[:summary_len - 1]
                else:
                    summary = summary[:-1]

                if 'multi' in args.model_type:
                    sents = []
                    for sent_idx in sent_indices:
                        if sent_idx[0] == padding_idx:
                            break
                        sent_idx -= 1
                        sent_idx = sent_idx.detach().cpu().tolist()
                        if padding_idx-1 in sent_idx:
                            sent_len = sent_idx.index(padding_idx-1)
                            sent_idx = sent_idx[:sent_len]
                        sent_tokens = np.array(text_article)[np.array(sent_idx)]
                        sents.append(' '.join(sent_tokens).replace(' ##', ''))
                    text_article_cleaned = '\n'.join(sents)
                    summary_cleaned = ' '.join(summary).replace(' ##', '')
                else:
                    text_article_cleaned = ' '.join(text_article).replace(' ##', '')
                    summary_cleaned = ' '.join(summary).replace(' ##', '')

                f_out.write('article\n' + text_article_cleaned + '\n')
                f_out.write('summary\n' + summary_cleaned + '\n')

                if 'multi' in args.model_type:
                    sent_pred_curr_prob = softmax(sent_basis_preds[k])[:len(sents)]
                    f_out.write('sentence basis prob\n')
                    for prob in sent_pred_curr_prob:
                        f_out.write(str(prob[0]) + '\t' + str(prob[1]) + '\n\n')
                        
            num_negative = 0
            for j, arc in enumerate(arc_list):
                arc_text = tokenizer.decode(arc)
                arc_text = arc_text.replace(tokenizer.pad_token, '').strip()

                if arc_text == '':  # for bert
                    break

                pred_temp = softmax([preds[k][j]])
                pred = np.argmax(pred_temp)
                if pred == 0:
                    num_negative += 1
                if f_out is not None:
                    f_out.write(arc_text + '\n')
                    f_out.write('pred:\t' + str(pred) + '\n')
                    f_out.write(str(pred_temp[0][0]) + '\t' + str(pred_temp[0][1]) + '\n\n')
            if f_out is not None:
                f_out.write('sent gold:\t' + str(out_label_ids_sent[k]) + '\n')
            if num_negative > 0:
                sent_pred.append(0)
                if f_out is not None:
                    f_out.write('sent_pred:\t0\n\n')
            else:
                sent_pred.append(1)
                if f_out is not None:
                    f_out.write('sent_pred:\t1\n\n')
            k += 1
    if f_out is not None:
        f_out.close()

    balanced_acc = balanced_accuracy_score(y_true=out_label_ids_sent, y_pred=sent_pred)
    cm = confusion_matrix(y_true=out_label_ids_sent, y_pred=sent_pred)
    f1 = f1_score(y_true=out_label_ids_sent, y_pred=sent_pred, average='macro')
    result = {'acc': balanced_acc, 'cm': cm, 'macro f1': f1}
    result_dep = {}
    # print(result_dep)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result_dep.keys()):
            logger.info("dep level %s = %s", key, str(result_dep[key]))
            writer.write("dep level  %s = %s\n" % (key, str(result_dep[key])))
        for key in sorted(result.keys()):
            logger.info("sent level %s = %s", key, str(result[key]))
            writer.write("sent level  %s = %s\n" % (key, str(result[key])))
        writer.write('\n')

    return result


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, writer) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size


    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * (args.stop_lr_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    if 'multi' in args.model_type:
        logger.info("Multi Task loss weight = %f, %f", args.multitask_loss_weight[0], args.multitask_loss_weight[1])

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    acc_prev = 0.

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        pred_labels = []
        labels = []
        corr = 0
        num_label = 0
        for step, batch in enumerate(epoch_iterator):
            
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
            mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
            sent_labels = batch[8]
            sent_indices, sent_basis_label, hypo_cls_idx =  batch[9],  batch[10], batch[11]

            inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                      'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                      'num_dependency': num_dependency, 'sent_label': sent_labels, 
                      'sent_indices': sent_indices, 'sent_basis_label': sent_basis_label,
                      'hypo_cls_idx':  hypo_cls_idx, 'device': args.device, 'multitask_loss_weight': args.multitask_loss_weight}

            model.train()
            outputs = model(**inputs)
            loss = outputs[0].mean()
            

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            now_loss = loss.item()
            tr_loss += loss.item()
            
            if 'multi' in args.model_type:
                loss_dep, loss_sent = outputs[3].mean().item(), outputs[4].mean().item()
                sent_basis_logits = outputs[2]
                pred_label = torch.argmax(sent_basis_logits, dim=2)
                corr += (sent_basis_label==pred_label).sum()
                num_label += (sent_basis_label!=-1).sum()
                
                pred_labels += pred_label.view(-1).tolist()
                labels += sent_basis_label.view(-1).tolist()
                
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if not scheduler.get_last_lr()[0] < args.min_lr:
                    scheduler.step()
                model.zero_grad()
                global_step += 1

                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('Loss/train', now_loss, global_step)
                
                if 'multi' in args.model_type:
                    writer.add_scalar('loss_sent/train', loss_sent, global_step)
                    writer.add_scalar('loss_dep/train', loss_dep, global_step)
                    if num_label != 0:
                        writer.add_scalar('sent_acc/train', corr/num_label, global_step)
                        writer.add_scalar('sent_bal_acc/train', compute_metrics_balanced(pred_labels, labels)['acc'], global_step)
                    else:
                        writer.add_scalar('sent_acc/train', 0, global_step)
                    
                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    logs = {}
                    loss_scalar_dep = (tr_loss - logging_loss) / args.save_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar_dep
                    if 'multi' in args.model_type:
                        logs["sent_acc"] = (corr/num_label).item()
                        logs['sent_bal_acc'] = compute_metrics_balanced(pred_labels, labels)['acc']
                        logs["loss_sent"] = loss_sent
                        logs["loss_dep"] = loss_dep
                    logging_loss = tr_loss

                    # print(json.dumps({**logs, **{"step": global_step, 'epoch': epoch_iterator.n}}))
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # Evaluation
                    result = evaluate(args, model, tokenizer, eval_dataset)
                    # save_checkpoints(args, args.output_dir, model, tokenizer)
                    writer.add_scalar('acc/val', result['acc'], global_step)
                    writer.add_scalar('macroF1/val', result['macro f1'], global_step)

                    if result['acc'] > acc_prev:
                        acc_prev = result['acc']
                        # Save model checkpoint best
                        logger.info('best model saved, step: {}'.format(global_step))
                        output_dir = os.path.join(args.output_dir, "model-best")
                        save_checkpoints(args, output_dir, model, tokenizer)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

    evaluate(args, model, tokenizer, eval_dataset)
    save_checkpoints(args, args.output_dir, model, tokenizer)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size evaluation.", )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument("--stop_lr_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument("--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--seed", type=int, default=100, help="random seed for initialization")
    parser.add_argument("--min_lr", type=float, default=5e-8)
    parser.add_argument("--dataset", nargs='+', default=['fusion', 'comp', 'ref', 'para'])
    parser.add_argument("--same_size", action="store_true")
    parser.add_argument("--multitask_loss_weight", nargs='+', type=float, default=[1.0, 1.0])
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()
    
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.n_gpu = 1  # no multi gpu support right now.
    #device = torch.device("cuda", args.gpu_device)
    args.device = "cuda"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'model.log')
    )

    # Set seed
    set_seed(args)

    if args.deterministic:
        logger.info("set cudnn deterministic mode")
        torch.backends.cudnn.deterministic = True

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.input_dir is not None:
        logger.info('loading model')
        tokenizer = tokenizer_class.from_pretrained(args.input_dir)
        model = model_class.from_pretrained(args.input_dir)
    else:
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

    model.to(args.device)
    logger.info(model)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    logger.info('eval dataset size: {}'.format(len(eval_dataset)))
    evaluate(args, model, tokenizer, eval_dataset, save_file=True)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir))
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        logger.info('train dataset size: {}'.format(len(train_dataset)))

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset, writer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
