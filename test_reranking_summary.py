import argparse
import logging
import os
from typing import Dict
import numpy as np
from train_utils import load_and_cache_examples
from train import MODEL_CLASSES, set_seed
from my_dataset import collate_fn
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from sklearn.utils.extmath import softmax
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from train_utils import setup_logger
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = setup_logger(__name__)


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=2, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
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

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids_sent = sent_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids_sent = np.append(out_label_ids_sent, sent_labels.detach().cpu().numpy(), axis=0)
        if len(outputs) > 2:
            if sent_basis_preds is None:
                sent_basis_preds = outputs[2].detach().cpu().tolist()
                sent_basis_labels = sent_basis_label.detach().cpu().tolist()
            else:
                sent_basis_preds.extend(outputs[2].detach().cpu().tolist())
                sent_basis_labels.extend(sent_basis_label.detach().cpu().tolist())

    f_out = open(os.path.join(eval_output_dir, 'dev_out.txt'), 'w')
    k = 0
    correct = 0
    sent_basis_pred = []

    for batch in eval_dataloader:
        first = True
        for inp, arc_list, sent_indices in zip(batch[0], batch[7], batch[9]):
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
            if first:
                f_out.write('article\n' + text_article_cleaned + '\n')
            f_out.write('summary\n' + summary_cleaned + '\n')

            if 'multi' in args.model_type:
                sent_pred_curr_prob = softmax(sent_basis_preds[k])[:len(sents)]
                f_out.write('sentence basis prob\n')
                for prob in sent_pred_curr_prob:
                    f_out.write(str(prob[0]) + '\t' + str(prob[1]) + '\n\n')
                
                sent_basis_pred_curr = np.argmax(sent_pred_curr_prob, axis=1).tolist()

                if -1 in sent_basis_labels[k]:
                    neg_index = sent_basis_labels[k].index(-1)
                    sent_basis_labels[k] = sent_basis_labels[k][:neg_index]
                
                num_add = len(sent_basis_labels[k]) - len(sent_basis_pred_curr)
                
                if num_add > 0:
                    sent_basis_pred_curr = sent_basis_pred_curr + [0 for _ in range(num_add)]
                elif num_add < 0:
                    print('error')
                    print(k)
                    print(num_add)

                sent_basis_pred.extend(sent_basis_pred_curr)

            pred_neg_probs = []
            for j, arc in enumerate(arc_list):
                arc_text = tokenizer.decode(arc)
                arc_text = arc_text.replace(tokenizer.pad_token, '').strip()
                if arc_text == '':  # for bert
                    break

                pred_temp = softmax([preds[k][j]])
                pred_neg_probs.append(pred_temp[0][0])
                pred = np.argmax(pred_temp)

                f_out.write(arc_text + '\n')
                f_out.write('pred:\t' + str(pred) + '\n')
                f_out.write(str(pred_temp[0][0]) + '\t' + str(pred_temp[0][1]) + '\n\n')
            f_out.write('sent gold:\t' + str(out_label_ids_sent[k]) + '\n')
            
            max_neg_pred = np.max(pred_neg_probs)
            
            if not first:
                if max_neg_pred < max_neg_pred_first:
                    f_out.write('sent_pred:\t0 1\n')
                    if out_label_ids_sent[k] == 1:
                        correct += 1
                else:
                    f_out.write('sent_pred:\t1 0\n\n')
                    if out_label_ids_sent[k] == 0:
                        correct += 1
                f_out.write(str(max_neg_pred_first) + '\t' + str(max_neg_pred) + '\n\n')
            k += 1
            first = False
            max_neg_pred_first = max_neg_pred
    f_out.close()


    acc = correct / (len(eval_dataset)/2)
    result = {'acc': acc, 'correct': correct}
    result_basis = {}
    if 'multi' in args.model_type:
        sent_basis_labels = np.array(sum(sent_basis_labels, []))
        sent_basis_pred = np.array(sent_basis_pred)
        sent_basis_pred = sent_basis_pred[sent_basis_labels!=-1]
        sent_basis_labels = sent_basis_labels[sent_basis_labels!=-1]

        sent_basis_balanced_acc = balanced_accuracy_score(y_true=sent_basis_labels, y_pred=sent_basis_pred)
        sent_basis_f1 = f1_score(y_true=sent_basis_labels, y_pred=sent_basis_pred, average='macro')
        sent_basis_cm = confusion_matrix(y_true=sent_basis_labels, y_pred=sent_basis_pred)
        result_basis = {'acc': sent_basis_balanced_acc, 'macro-f1': sent_basis_f1, 'cm': sent_basis_cm}

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("reranking %s = %s", key, str(result[key]))
            writer.write("reranking  %s = %s\n" % (key, str(result[key])))
        writer.write('\n')
        for key in sorted(result_basis.keys()):
            logger.info("sent basis %s = %s", key, str(result_basis[key]))
            writer.write("sent basis  %s = %s\n" % (key, str(result_basis[key])))
        writer.write('\n')

    return result


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
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int, help="Batch size evaluation.", )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument("--stop_lr_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--include_sentence_level", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--seed", type=int, default=100, help="random seed for initialization")
    parser.add_argument("--min_lr", type=float, default=5e-8)
    parser.add_argument("--loss_weight", action="store_true")
    parser.add_argument("--dataset", nargs='+', default=['fusion', 'comp', 'ref', 'para'])
    parser.add_argument("--same_size", action="store_true")
    parser.add_argument("--multitask_loss_weight", nargs='+', type=float, default=[1.0, 1.0])

    args = parser.parse_args()
    
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
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
    evaluate(args, model, tokenizer, eval_dataset)


if __name__ == "__main__":
    main()
