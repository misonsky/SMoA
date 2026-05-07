import os
import pickle

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, EarlyStoppingCallback

from dataset.dataset_hg import HGDataset,HFDataset,GLUEDataset
from dataset.format_inputs import TASK_TYPE, format_causal_input, gen_max_new_token_map, token_length_map, dataset_map, \
    task_map

from transformers import DataCollatorForTokenClassification

from utils import forward_wrap_with_option_len,DataCollatorWithPaddingAndNesting,SIGUSR1Callback
from dataset.tasks import *

from datetime import datetime
import jsonlines
import torch
import transformers
from pytictoc import TicToc
from models.get_models import (
    print_trainable_parameters,
    get_tokenizer,
    get_prefix_tuning_models,
    get_fft_models,
    get_lora_models,
    get_smoa_models,
    get_qora_models,
    get_melora_models)
import argparse
from dataset.tasks import get_task
#trainer
from transformers import (
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer)
from customized_trainer import Seq2SeqTrainer
from transformers.generation import GenerationConfig
# from customized_trainer import customized_trainer
from utils import EvaluateAfterTrainForGeneration,evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--peft_type', type=str,default='hime',choices=['prefix', 'lora', 'smoa', 'fft','qora','melora'])
parser.add_argument('--task_type',type=str,default="CAUSAL_LM")
parser.add_argument('--enable_grad_ckpt', action='store_true')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--num_train', default=1000, type=int)
parser.add_argument('--num_eval', default=100, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--grad_acc', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--warmup', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--epoch', type=float, default=10)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default="facebook/opt-125m")
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--dataset', type=str, default='sst2',
                    choices=['e2e_nlg','e2e_cleaned','dailydialog','mmlu',
                            'samsum','boolq','common_170k','piqa',
                            'siqa','hellas','winog','obqa','gsm8k',
                            'convai2','arce','arcc','meta_math','common_all',
                            'sst2','copa','multirc','cb','wic','wsc','record',
                            'rte','squad','drop'])
parser.add_argument('--dataset_analysis', action='store_true')
parser.add_argument('--dataset_ratio', type=float, default=1.0)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--ds_config', type=str, default=None)
parser.add_argument('--output_folder', type=str, default='outputs')
parser.add_argument('--load_bit', type=int, default=16)
parser.add_argument('--r_ab', type=int, default=4)
parser.add_argument('--l_num', type=int, default=4)
parser.add_argument('--smoa_num_blocks', type=int, default=None, help='Number of SMoA spectral blocks; defaults to --l_num.')
parser.add_argument('--smoa_branch_ranks', type=str, default=None, help='Optional comma-separated SMoA per-block ranks, e.g. 8,8.')
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--bias', type=str, default="none")
parser.add_argument('--target_modules', type=str, default='q_proj,v_proj,k_proj,up_proj,down_proj')
parser.add_argument('--eval_strategy', type=str, default='steps', choices=['no', 'steps', 'epoch'])
parser.add_argument('--eval_steps', type=float, default=1000)
parser.add_argument('--logging_steps', type=float, default=1000)
parser.add_argument('--max_length', type=int, default=None)
parser.add_argument('--max_new_tokens', type=int, default=None)
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--virtual_tokens', type=int, default=8)
parser.add_argument('--compute_rank', action='store_true')
parser.add_argument('--compute_norm', action='store_true')
parser.add_argument('--load_order', type=int, default=-1)
parser.add_argument('--init_ab', type=str, default='kaiming,zero')
parser.add_argument('--train_ab', type=str, default='yy', help='y means yes, n means no')
parser.add_argument('--do_sample', default='false', type=str)
parser.add_argument('--top_k', default=40, type=int)
parser.add_argument('--top_p', default=0.95, type=float)
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--rand_R', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--decoding', type=str, default='default', choices=['default', 'greedy'])
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--early_stop_patience', type=int, default=0)
COMPUTE_DS_LENGTH = False
args = parser.parse_args()
if args.compute_rank or args.compute_norm:
    assert args.ckpt is not None

output_name = 'output_{}_{}'.format(args.load_order, args.dataset)
if args.dataset == 'mmlu':
    output_name += '_prob'
if args.max_new_tokens is not None:
    output_name += '_maxT={}'.format(args.max_new_tokens)
if args.beam_size is not None:
    output_name += '_beam={}'.format(args.beam_size)
output_path = '{}/{}_eval.jsonl'.format(args.ckpt, output_name)

dataset_name = args.dataset

MAX_NEW_TOKEN_LENGTH = gen_max_new_token_map[dataset_name]
MAX_TOKEN_LENGTH = token_length_map[dataset_name]

if args.max_new_tokens is not None:
    MAX_NEW_TOKEN_LENGTH = args.max_new_tokens

metric_for_best_model={}
# keyword_map = {
#     'e2e_nlg': 'TARGET: '
# }

# convert args to dict
args_dict = vars(args)
model_name = args.model_name
peft_type = args.peft_type
train_ab = args.train_ab
# create a directory by time
exp_name = f"{args.output_folder}/{model_name.split('/')[-1]}-{dataset_name}-{peft_type}-lr={format(args.lr, '.2e')}-"


if args.seed is not None:
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    exp_name = exp_name + f'seed={args.seed}-'
    seed_everything(args.seed)
output_dir_by_time = exp_name + args.exp_name + '-' + datetime.now().strftime(
    "%Y-%m-%d-%H-%M-%S")


if args.ckpt is None:
    os.makedirs(output_dir_by_time, exist_ok=True)


task = get_task(dataset_name)
if task is not None:
    task = task()

if task:
    task.load_dataset(split='train')
    task.load_dataset(split='validation')
    # -1 returan all examples
    train_samples = task.sample_subset(data_split="train", seed=args.seed, num=args.num_train)
    valid_samples = task.sample_subset(data_split="valid", seed=args.seed, num=args.num_eval)

else:
    train_dataset = HGDataset(dataset_map[dataset_name], 'train', task_map[dataset_name], training_ratio=args.dataset_ratio)
    valid_dataset = HGDataset(dataset_map[dataset_name], 'valid', task_map[dataset_name],training_ratio=args.dataset_ratio)
    test_dataset = HGDataset(dataset_map[dataset_name], 'test', task_map[dataset_name], training_ratio=args.dataset_ratio)

if peft_type == 'lora':
    model, tokenizer, model_config = get_lora_models(model_name=model_name,
                    enable_checkpoint=args.enable_grad_ckpt,
                    load_bit=args.load_bit,
                    r=args.r_ab,
                    bias = args.bias,
                    lora_alpha=args.lora_alpha,
                    lora_dropout = args.lora_dropout,
                    target_modules=args.target_modules,
                    task_type=args.task_type)
elif peft_type == 'smoa':
    smoa_num_blocks = args.smoa_num_blocks if args.smoa_num_blocks is not None else args.l_num
    smoa_branch_ranks = None
    if args.smoa_branch_ranks is not None:
        smoa_branch_ranks = [int(rank) for rank in args.smoa_branch_ranks.split(',') if rank]
    model, tokenizer, model_config = get_smoa_models(model_name=model_name,
                    enable_checkpoint=args.enable_grad_ckpt,
                    load_bit=args.load_bit,
                    r=args.r_ab,
                    num_blocks=smoa_num_blocks,
                    branch_ranks=smoa_branch_ranks,
                    bias = args.bias,
                    lora_alpha=args.lora_alpha,
                    lora_dropout = args.lora_dropout,
                    target_modules=args.target_modules,
                    task_type=args.task_type)
elif peft_type == 'melora':
    rank = [args.r_ab] * args.l_num
    lora_alpha = [args.lora_alpha] * args.l_num
    model, tokenizer, model_config = get_melora_models(model_name=model_name,
                    enable_checkpoint=args.enable_grad_ckpt,
                    load_bit=args.load_bit,
                    r=rank,
                    bias = args.bias,
                    lora_alpha=lora_alpha,
                    lora_dropout = args.lora_dropout,
                    target_modules=args.target_modules,
                    task_type=args.task_type)
elif peft_type == 'qora':
    model, tokenizer, model_config = get_qora_models(model_name=model_name,
                    enable_checkpoint=args.enable_grad_ckpt,
                    load_bit=args.load_bit,
                    r=args.r_ab,
                    bias = args.bias,
                    lora_alpha=args.lora_alpha,
                    lora_dropout = args.lora_dropout,
                    target_modules=args.target_modules,
                    task_type=args.task_type)

elif peft_type == 'fft':
    model, tokenizer, model_config = get_fft_models(load_bit=args.load_bit,
                                                    model_name=model_name,
                                                    enable_checkpoint=args.enable_grad_ckpt)

elif peft_type == 'prefix':
    model, tokenizer, model_config = get_prefix_tuning_models(load_bit=args.load_bit, 
                                                            model_name=model_name,
                                                            enable_checkpoint=args.enable_grad_ckpt,
                                                            virtual_tokens=args.virtual_tokens,
                                                            task_type=args.task_type)
else:
    raise NotImplementedError('Not supported model!')

trainable_params = print_trainable_parameters(model)

def get_parameter_dict(model):
    return dict(model.named_parameters())

tokenizer_left = get_tokenizer(model_name=model_name)
tokenizer_left.padding_side = 'left'

tokenizer_right = get_tokenizer(model_name=model_name)
tokenizer_right.padding_side = 'right'

if tokenizer_left.pad_token_id is None and 'llama-3' in model_name.lower():
    tokenizer_left.pad_token = tokenizer_left.bos_token
    tokenizer_right.pad_token = tokenizer_right.bos_token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = tokenizer.bos_token_id
elif tokenizer_left.pad_token_id is None and 'llama3' in model_name.lower():
    tokenizer_left.pad_token = tokenizer_left.bos_token
    tokenizer_right.pad_token = tokenizer_right.bos_token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = tokenizer.bos_token_id
elif tokenizer_left.pad_token_id is None:
    tokenizer_left.pad_token = tokenizer_left.unk_token
    tokenizer_right.pad_token = tokenizer_right.unk_token
    tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.unk_token_id
if "opt" in model_name.lower():
    tokenizer.bos_token_id = 0

if task_map[dataset_name] in [TASK_TYPE.SST2,TASK_TYPE.COPA,TASK_TYPE.MULTIRC,TASK_TYPE.CB,TASK_TYPE.WIC,TASK_TYPE.WSC,TASK_TYPE.RECORD,TASK_TYPE.RTE,TASK_TYPE.SQUAD,TASK_TYPE.DROP]:
    train_as_classification = True
    if task_map[dataset_name] in [TASK_TYPE.COPA,TASK_TYPE.RECORD,TASK_TYPE.SQUAD,TASK_TYPE.DROP]:
        train_as_classification = False
    train_dataset=HFDataset(GLUEDataset(task=task,
                template=task.get_template(), 
                demonstrations=[],
                samples=train_samples,
                tokenizer=tokenizer_left, 
                max_length=MAX_TOKEN_LENGTH, 
                train_as_classification=train_as_classification,
                generation=task.generation, 
                generation_with_gold=True,
                max_new_tokens=MAX_NEW_TOKEN_LENGTH))

    valid_dataset = HFDataset(GLUEDataset(task=task,
                template=task.get_template(), 
                demonstrations=[],
                samples=valid_samples,
                tokenizer=tokenizer_left, 
                max_length=MAX_TOKEN_LENGTH, 
                train_as_classification=train_as_classification,
                generation=task.generation, 
                generation_with_gold=True,
                max_new_tokens=MAX_NEW_TOKEN_LENGTH))

    model.original_forward = model.forward
    model.forward = forward_wrap_with_option_len.__get__(model, type(model))

test_steps = 0

def data_collator_e2e(features, return_tensors="pt"):
    batchfied_features = {}
    keys = features[0].keys()
    for key in keys:
        batchfied_features[key] = [f[key] for f in features]
    split = batchfied_features['split'][0]
    for_inference = (split == 'test')
    template_type = 0
    if dataset_name in ['boolq', 'gsm8k', 'common_170k', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa','common_all']:
        template_type = 4
    if dataset_name in ['mmlu']:
        template_type = 7
    
    lm_input, lm_target = format_causal_input(batch = batchfied_features,
                                                   left_tokenizer=tokenizer_left, 
                                                   right_tokenizer=tokenizer_right,
                                                    template_type=template_type,
                                                    max_token_length=MAX_TOKEN_LENGTH,
                                                    for_test=for_inference,
                                                    shift_target=False,
                                                    target_length=MAX_NEW_TOKEN_LENGTH)

    # Replace target pad to -100
    lm_target_ce = lm_target.clone()
    lm_target_ce[lm_target_ce == tokenizer_left.pad_token_id] = -100
    if peft_type in ['prefix']:
        lm_input['attention_mask'] = None

    batch = {**lm_input, 'labels': lm_target_ce}
    if for_inference:
        batch = lm_input
    return batch
data_collator_function = None

if task_map[dataset_name] in [TASK_TYPE.SST2,TASK_TYPE.COPA,TASK_TYPE.MULTIRC,TASK_TYPE.CB,TASK_TYPE.WIC,TASK_TYPE.WSC,TASK_TYPE.RECORD,TASK_TYPE.RTE,TASK_TYPE.SQUAD,TASK_TYPE.DROP]:
    if task_map[dataset_name] in [TASK_TYPE.COPA, TASK_TYPE.RECORD, TASK_TYPE.SQUAD, TASK_TYPE.DROP]:
        data_collator_function = DataCollatorForTokenClassification(tokenizer_left, pad_to_multiple_of=8)
    else:
        data_collator_function = DataCollatorWithPaddingAndNesting(tokenizer_left, pad_to_multiple_of=8)
else:
    data_collator_function = data_collator_e2e

generation_config = GenerationConfig(
    max_length=MAX_TOKEN_LENGTH,
    num_beams=1,
)
callbacks = []
if args.early_stop_patience > 0:
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))
eval_bz = max(1, int(args.batch / 4))
if dataset_name == 'common_170k':
    eval_bz = args.batch

if task_map[dataset_name] in [TASK_TYPE.SST2,TASK_TYPE.COPA,TASK_TYPE.MULTIRC,TASK_TYPE.CB,TASK_TYPE.WIC,TASK_TYPE.WSC,TASK_TYPE.RECORD,TASK_TYPE.RTE,TASK_TYPE.SQUAD,TASK_TYPE.DROP]:
    trainer_args = TrainingArguments(
        deepspeed=args.ds_config,
        local_rank=args.local_rank,
        dataloader_num_workers=args.num_workers,
        resume_from_checkpoint=args.ckpt,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=eval_bz,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=args.enable_grad_ckpt,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        bf16=True if torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
        # fp16=True if not torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
        metric_for_best_model='eval_loss',
        remove_unused_columns=False,
        save_on_each_node=False,
        save_safetensors=peft_type == 'fft',
        output_dir=output_dir_by_time,
        do_eval=True,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=['tensorboard'],
        eval_steps=args.eval_steps,
        eval_accumulation_steps=1,
        load_best_model_at_end=True
    )
else:
    trainer_args = Seq2SeqTrainingArguments(
        deepspeed=args.ds_config,
        local_rank=args.local_rank,
        dataloader_num_workers=args.num_workers,
        resume_from_checkpoint=args.ckpt,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=eval_bz,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=args.enable_grad_ckpt,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        bf16=True if torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
        # fp16=True if not torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
        metric_for_best_model='eval_loss',
        remove_unused_columns=False,
        save_on_each_node=False,
        save_safetensors=peft_type == 'fft',
        output_dir=output_dir_by_time,
        do_eval=True,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=['tensorboard'],
        eval_steps=args.eval_steps,
        eval_accumulation_steps=1,
        generation_config=GenerationConfig(
            max_length=MAX_TOKEN_LENGTH,
            num_beams=1,
        ),
        load_best_model_at_end=True,
        predict_with_generate=True,
    )

if task_map[dataset_name] in [TASK_TYPE.SST2,TASK_TYPE.COPA,TASK_TYPE.MULTIRC,TASK_TYPE.CB,TASK_TYPE.WIC,TASK_TYPE.WSC,TASK_TYPE.RECORD,TASK_TYPE.RTE,TASK_TYPE.SQUAD,TASK_TYPE.DROP]:
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=callbacks,
        args=trainer_args,
        data_collator=data_collator_function
    )
else:
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=callbacks,
        args=trainer_args,
        data_collator=data_collator_function
    )

if args.ckpt is not None:
    trainer._load_best_model(order=args.load_order)
    train_seconds = -1

else:
    train_tic = TicToc()
    train_tic.tic()
    trainer.train()
    train_seconds = train_tic.tocvalue()


if task_map[dataset_name] in [TASK_TYPE.SST2,TASK_TYPE.COPA,TASK_TYPE.MULTIRC,TASK_TYPE.CB,TASK_TYPE.WIC,TASK_TYPE.WSC,TASK_TYPE.RECORD,TASK_TYPE.RTE,TASK_TYPE.SQUAD,TASK_TYPE.DROP]:
    
    metrics = evaluation(eval_samples=valid_samples,
                        task=task,
                        args = args,
                        model = model,
                        tokenizer=tokenizer,
                        dataset_name=dataset_name,
                        train_as_classification=train_as_classification)
    print(metrics)
else:
    EvaluateAfterTrainForGeneration(args,trainer,model,model_config,tokenizer,tokenizer_left,tokenizer_right,dataset_name,test_dataset,output_path,output_dir_by_time,train_seconds)
