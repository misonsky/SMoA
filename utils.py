import json
import os
import contextlib
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
import logging
import time
import jsonlines
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import transformers
from typing import Optional, Union, List, Dict, Any
import signal
from subprocess import call
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dataset.dataset_hg import HGDataset,HFDataset,single_sample
from tqdm import tqdm
from dataset.format_inputs import format_causal_input, dataset_map, task_map,token_length_map,gen_max_new_token_map
from models.get_models import print_trainable_parameters

from metrics import calculate_metric

logger = logging.getLogger(__name__)

metrics_map={"sst2":"accuracy",
            "rte":"accuracy",
            "cb":"accuracy",
            "wsc":"accuracy",
            "wic":"accuracy",
            "multirc":"accuracy",
            "copa":"accuracy",
            "record":"accuracy",
            "squad":"f1",
            "drop":"f1"}
def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens 
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    # print("\nlabel-0",labels.size())
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits
    # print("\nlogits-0",logits.size())
    # print("\n num_options",num_options)
    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None: 
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # Option part
        shift_labels[~mask] = 0 # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1) # (bsz x num_options)
        # print(selected_log_probs.size())
        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0) # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0) # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0] # Labels repeat so we only take the first one
            # print(selected_log_probs.size())
            # print("labels",labels.size())
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        
        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0, "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature
            
        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != "gold"} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]
        
        return batch
        

class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)



def EvaluateAfterTrainForGeneration(args,trainer,model,model_config,tokenizer,tokenizer_left,tokenizer_right,dataset_name,test_dataset,output_path,output_dir_by_time,train_seconds):
    kwgenargs = {}
    args_dict = vars(args)
    MAX_TOKEN_LENGTH = token_length_map[dataset_name]
    MAX_NEW_TOKEN_LENGTH = gen_max_new_token_map[dataset_name]
    if args.max_new_tokens is not None:
        MAX_NEW_TOKEN_LENGTH = args.max_new_tokens
    trainable_params = print_trainable_parameters(model)
    if args.do_sample is not None:
        if args.do_sample.lower() in ['yes', 'true']:
            do_sample = True
        else:
            do_sample = False
        kwgenargs['do_sample'] = do_sample
    if args.decoding == 'greedy':
        eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False)
    elif dataset_name == 'common_170k':
        eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False)
    elif dataset_name == 'common_all':
        beam = 4
        eval_result = {}
        for _dataset_name in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa']:
            print(f'decoding {dataset_name}')
            test_dataset = HGDataset(dataset_map[_dataset_name], 
                                    'test',
                                    task_map[_dataset_name],
                                    training_ratio=args.dataset_ratio)
            _output_path = output_path.replace(dataset_name, _dataset_name)
            if os.path.exists(_output_path):
                print(f'{_output_path} exists, skipping')
                continue
            if args.beam_size is not None:
                beam = args.beam_size
            _eval_result = trainer.predict(test_dataset,
                                       max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                       pad_token_id=tokenizer.pad_token_id,
                                       temperature=args.temperature,
                                       top_p=args.top_p,
                                       top_k=args.top_k,
                                       num_beams=beam,
                                       **kwgenargs)
            eval_result[_output_path] = _eval_result
    elif dataset_name in ['mmlu']:
        eval_results = []
        pbar = tqdm(test_dataset)
        id_a, id_b, id_c, id_d = tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D'])
        options = ['A', 'B', 'C', 'D']
        context = []
        text_result = []
        ground_truth = []
        for row in pbar:
            model.eval()
            with torch.no_grad():
                keys = row.keys()
                batchfied_features = {}
                for key in keys:
                    batchfied_features[key] = [row[key]]
                lm_input, lm_target = format_causal_input(batchfied_features, 
                                                      tokenizer_left, 
                                                      tokenizer_right,
                                                      template_type=7, 
                                                      max_token_length=MAX_TOKEN_LENGTH,
                                                      for_test=True, 
                                                      shift_target=False,
                                                      target_length=MAX_NEW_TOKEN_LENGTH)
                lm_input = lm_input.to('cuda')
                with torch.autocast('cuda'):
                    prob = model(**lm_input).logits
                id_probs = prob[0][-1][[id_a, id_b, id_c, id_d]]
                prob_pred = options[id_probs.argmax().item()]
                answer = row['target']
                eval_results.append(prob_pred == answer)
                acc = np.asarray(eval_results).mean()
                pbar.set_postfix_str(f'Current ACC: {acc * 100}')
                context.append(row['input'])
                text_result.append(prob_pred)
                ground_truth.append(answer)
        print(f'ACC: {acc * 100}')
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write(args_dict)
            writer.write(model_config.to_dict())
            writer.write({"acc": acc * 100})
            writer.write(trainable_params)
            for c, p, g in zip(context, text_result, ground_truth):
                writer.write({
                    'context': c,
                    'pred': p,
                    'gt': g,
                })
        exit(0)
    elif dataset_name in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'gsm8k']:
        beam = 4
        if args.beam_size is not None:
            beam = args.beam_size
        eval_result = trainer.predict(test_dataset,
                                max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                pad_token_id=tokenizer.pad_token_id,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                num_beams=beam,
                                **kwgenargs)
    else:
        eval_result = trainer.predict(test_dataset,
                                max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=False, num_beams=4,
                                length_penalty=0.9, no_repeat_ngram_size=4)
    if not isinstance(eval_result, dict):
        if args.ckpt is None:
            output_path = '{}/output.jsonl'.format(output_dir_by_time)
        eval_result = {output_path: eval_result}
    for _output_path, _eval_result in eval_result.items():
        if args.local_rank in [-1, 0]:
            logits = _eval_result.predictions
            logits[logits == -100] = tokenizer.pad_token_id
            raw_text_result = tokenizer.batch_decode(logits)
            text_result = []
            for tt in raw_text_result:
                tt = tt.replace(tokenizer.pad_token, '')
                keywords = [tokenizer.eos_token, 'Q:', 'R:']
                for keyword in keywords:
                    if keyword in tt:
                        tt = tt[:tt.index(keyword)]
                text_result.append(tt)
            context = [test_dataset.__getitem__(i)['input'] for i in range(test_dataset.__len__())]
            ground_truth = [test_dataset.__getitem__(i)['target'] for i in range(test_dataset.__len__())]
            if args.ckpt is not None:
                if os.path.exists(output_dir_by_time):
                    os.removedirs(output_dir_by_time)
            else:
                _output_path = '{}/output.jsonl'.format(output_dir_by_time)
            mem_used = torch.cuda.mem_get_info()[1] / 1024 / 1024 - torch.cuda.mem_get_info()[0] / 1024 / 1024

            with jsonlines.open(_output_path, mode='w') as writer:
                writer.write(args_dict)
                if args.peft_type != 'fft':
                    writer.write(model_config.to_dict())
                else:
                    writer.write('\n')
                writer.write({"mem_used": mem_used, "train_seconds": train_seconds})
                writer.write(trainable_params)
                for c, p, g in zip(context, text_result, ground_truth):
                    writer.write({
                        'context': c,
                        'pred': p,
                        'gt': g
                    })


### glue
def one_step_pred_(example,reference,candidates,model,tokenizer,args,verbose=False,generation=False):
    if generation:
        input_ids = example["input_ids"]
        input_ids = torch.tensor([input_ids]).to('cuda')
        outputs = model.generate(
                input_ids, 
                do_sample=args.sampling, 
                temperature=args.temperature, 
                num_beams=args.num_beams, 
                top_p=args.top_p, 
                top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, 
                eos_token_id=[tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], tokenizer.eos_token_id],
            )
        # For generation, directly return the text output
        output_text = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
        return Prediction(correct_candidate=reference, predicted_candidate=output_text)
    else:
        outputs = []
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {candidates}")
            logger.info(f"Correct candidate: {reference}")
        for candidate_id, example in enumerate(example):
            input_ids = example["input_ids"]
            input_ids = torch.tensor([input_ids]).to('cuda')
            option_len = example["option_len"]
            
            with torch.inference_mode():
                model.eval()
                logits = model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            selected_log_probs = selected_log_probs[-option_len:]
            outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": None})
            if verbose:
                if candidate_id == 0:
                    logger.info("=== Candidate %d ===" % candidate_id)
                    logger.info(self.tokenizer.decode(input_ids))
                else:
                    logger.info("=== Candidate %d (without context)===" % candidate_id)
                    logger.info(self.tokenizer.decode(input_ids))
                logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")
        scores = [x['log_probs'].mean().item() for x in outputs]
        if verbose:
            logger.info(f"Prediction scores: {scores}")
        if isinstance(reference, list):
            # For some datasets there are multiple correct answers
            correct_candidate_id = [candidates.index(c) for c in reference]
        else:
            correct_candidate_id = candidates.index(reference)

        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))
def forward(input_ids, model,args,option_len=None, generation=False):
    input_ids = torch.tensor([input_ids]).to(model.device)
    if generation:
        # Autoregressive generation
        outputs = model.generate(
            input_ids, 
            do_sample=args.sampling, 
            temperature=args.temperature, 
            num_beams=args.num_beams, 
            top_p=args.top_p, 
            top_k=args.top_k, 
            max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
            num_return_sequences=1, 
            eos_token_id=[tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], tokenizer.eos_token_id],
            )
        output_text = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
        return output_text
    else:
        with torch.inference_mode():
            model.eval()
            logits = model(input_ids=input_ids).logits
        labels = input_ids[0, 1:]
        logits = logits[0, :-1] 
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
        selected_log_probs = selected_log_probs.cpu().detach()
        # Only return the option (candidate) part
        return selected_log_probs[-option_len:]
def one_step_pred(eval_sample,task,args,model,tokenizer,dataset_name,train_as_classification,verbose=False):
    verbose = verbose
    if verbose:
        logger.info("========= Example =========")
        logger.info(f"Candidate: {eval_sample.candidates}")
        logger.info(f"Correct candidate: {eval_sample.correct_candidate}")
    MAX_TOKEN_LENGTH = token_length_map[dataset_name]
    MAX_NEW_TOKEN_LENGTH = gen_max_new_token_map[dataset_name]
    encoded_candidates, option_lens=single_sample(task=task,
                template=task.get_template(), 
                demonstrations=[],
                sample=eval_sample,
                tokenizer=tokenizer,
                max_length=MAX_TOKEN_LENGTH,
                generation=task.generation, 
                generation_with_gold=True,
                max_new_tokens=MAX_NEW_TOKEN_LENGTH)
    outputs = []
    if task.generation:
        output_text = forward(input_ids=encoded_candidates[0],
                                    model=model,
                                    args=args,
                                    generation=True)
        if verbose:
            logger.info("=== Prompt ===")
            logger.info(self.tokenizer.decode(encoded_candidates[0]))
            logger.info(f"Output: {output_text}") 
        return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
    else:
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = forward(input_ids=encoded_candidate, 
                                        model=model,
                                        args=args,
                                        option_len=option_lens[candidate_id])
            if verbose:
                if candidate_id == 0:
                    logger.info("=== Candidate %d ===" % candidate_id)
                    logger.info(tokenizer.decode(encoded_candidate))
                else:
                    logger.info("=== Candidate %d (without context)===" % candidate_id)
                    logger.info(tokenizer.decode(encoded_candidate).split(task.train_sep)[-1])
                logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")
            outputs.append({"log_probs": selected_log_probs, "sfc_log_probs":  None})
        scores = [x['log_probs'].mean().item() for x in outputs]
        if verbose:
            logger.info(f"Prediction scores: {scores}")

        if isinstance(eval_sample.correct_candidate, list):
            # For some datasets there are multiple correct answers
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

def evaluation(eval_samples, task, args, model, tokenizer,dataset_name,train_as_classification):
    # Prediction loop
    predictions = []
    for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
        predictions.append(one_step_pred(eval_sample=eval_sample, 
                                        task=task,
                                        args=args,
                                        model=model,
                                        tokenizer=tokenizer,
                                        dataset_name=dataset_name,
                                        train_as_classification=train_as_classification,
                                        verbose=(eval_id < 3)))
    
    # Calculate metrics 
    metric_name = metrics_map[args.dataset]
    metrics = {metric_name: calculate_metric(predictions, metric_name)}
    return metrics

    # predictions = []
    # eval_candidates = []
    # for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
    #     if generation:
    #         reference = references[eval_id]
    #         predictions.append(
    #             one_step_pred(eval_sample,reference,candidates,model,tokenizer,args,generation)
    #         )
    #     else:
    #         num_options = eval_sample["num_options"]
    #         # option_len = eval_sample["option_len"]
    #         eval_candidates.append(eval_sample)
    #         if len(eval_candidates) == num_options:
    #             reference = references[eval_id // num_options]
    #             ref_candidates = candidates[eval_id // num_options]
    #             predictions.append(
    #                 one_step_pred(eval_candidates,reference,ref_candidates,model,tokenizer,args,generation)
    #             )
    #             eval_candidates=[]
    # metric_name = metrics_map[args.dataset]
    # # getattr(args.task, "metric_name", "accuracy")
    # metrics = {metric_name: calculate_metric(predictions, metric_name)}
    # return metrics
# def EvaluateAfterTrainForGlue():

                


