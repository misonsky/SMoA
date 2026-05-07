import os
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from .metrics import calculate_metric


def _freeze_for_peft(model, enable_checkpoint=False):
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)
    return model

def one_step_pred(example,model,tokenizer,device,args,generation=False):

    logger.info("========= Example =========")
    logger.info(f"Candidate: {example.candidates}")
    logger.info(f"Correct candidate: {example.correct_candidate}")
    outputs = []
    if generation:
        input_ids = example["input"]
        torch.tensor([input_ids]).to(device)
        model.generate(
                input_ids, 
                do_sample=args.sampling, 
                temperature=args.temperature, 
                num_beams=args.num_beams, 
                top_p=args.top_p, top_k=args.top_k, 
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, 
                eos_token_id=[tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], tokenizer.eos_token_id],
            )
        # For generation, directly return the text output
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
        return selected_log_probs[-example["option_len"]:]
def evaluation(eval_samples, model, tokenizer, device, args ,generation):
    # Prediction loop
    predictions = []  
    for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
        predictions.append(
            one_step_pred(eval_sample,model,tokenizer,device,args,generation)
        )
    metric_name = getattr(args.task, "metric_name", "accuracy")
    metrics = {metric_name: calculate_metric(predictions, metric_name)}
    return metrics

def get_tokenizer(model_name="facebook/opt-1.3b"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def get_models(model_name="facebook/opt-1.3b", enable_checkpoint=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, low_cpu_mem_usage=True
    )

    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer

def get_fft_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        #         device_map='auto',
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    print_trainable_parameters(model)
    return model, tokenizer, None

def get_lora_models(model_name="facebook/opt-1.3b",
                    enable_checkpoint=False,
                    load_bit=16,
                    r=16,
                    bias=None,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    task_type = "CAUSAL_LM",
                    target_modules=None):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = _freeze_for_peft(model, enable_checkpoint=enable_checkpoint)
    from cpeft import LoraConfig, get_peft_model
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


def get_smoa_models(model_name="facebook/opt-1.3b",
                    enable_checkpoint=False,
                    load_bit=16,
                    r=16,
                    num_blocks=2,
                    branch_ranks=None,
                    bias=None,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    task_type="CAUSAL_LM",
                    target_modules=None):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = _freeze_for_peft(model, enable_checkpoint=enable_checkpoint)

    from cpeft import SMoAConfig, get_peft_model
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    config = SMoAConfig(
        r=r,
        num_blocks=num_blocks,
        branch_ranks=branch_ranks,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config

def get_melora_models(model_name="facebook/opt-1.3b",
                    enable_checkpoint=False,
                    load_bit=16,
                    r=16,
                    bias=None,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    task_type = "CAUSAL_LM",
                    target_modules=None):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = _freeze_for_peft(model, enable_checkpoint=enable_checkpoint)
    from cpeft import MELoraConfig, get_peft_model
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    config = MELoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config

def get_qora_models(model_name="facebook/opt-1.3b",
                    enable_checkpoint=False,
                    load_bit=16,
                    r=16,
                    bias=None,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    task_type = "CAUSAL_LM",
                    target_modules=None):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = _freeze_for_peft(model, enable_checkpoint=enable_checkpoint)
    from cpeft import QoraConfig, get_peft_model
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    config = QoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config
def get_prefix_tuning_models(model_name="facebook/opt-1.3b", 
                            enable_checkpoint=False, 
                            load_bit=8, 
                            virtual_tokens=8,
                            task_type = "CAUSAL_LM"):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    from cpeft import PrefixTuningConfig, get_peft_model

    config = PrefixTuningConfig(
        num_virtual_tokens=virtual_tokens,
        task_type=task_type)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return {"trainable": trainable_params, "all": all_param, "trainable%": 100 * trainable_params / all_param}
