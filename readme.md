# PEFTZoo

PEFTZoo is a local research codebase for parameter-efficient fine-tuning (PEFT) experiments on language models. It includes several LoRA-style tuners and training scripts for commonsense reasoning, dialogue generation, and mathematical reasoning tasks.

## Supported PEFT Methods

The current training entry supports:

- `lora`: standard low-rank adaptation
- `melora`: multi-branch LoRA variant
- `qora`: Qora tuner
- `smoa`: Spectrum Modulation Adapter
- `hira`: hira tuning
- `ia3`: ia3 tuning
- `p_tuning`: p_tuning tuning
- `prefix`: prefix tuning
- `prompt`: prompt tuning
- `fft`: full fine-tuning

## SMoA

SMoA, Spectrum Modulation Adapter, is implemented under:

```text
cpeft/tuners/smoa.py
```

The implementation follows the paper design:

1. Compute a one-time SVD of each frozen target weight matrix.
2. Use the frozen spectral structure to reorder input and output coordinates.
3. Split the reordered coordinates into aligned spectral blocks.
4. Attach one local low-rank branch to each diagonal block.
5. Apply Hadamard modulation with the frozen block anchor:

```text
Delta M_k = (B_k A_k) * M_k
```

where `*` denotes element-wise multiplication. The block updates are assembled into a global update and can be merged into the base weight for inference.

## Main Files

```text
cpeft/tuners/smoa.py      # SMoA config, tuner model, and linear adapter layer
cpeft/config.py           # PEFT type registry, including SMOA
cpeft/mapping.py          # Maps SMOA to SMoAConfig and SMoAModel
cpeft/tuners/__init__.py  # Exports SMoA classes
models/get_models.py      # get_smoa_models helper
train.py             # Main training entry
run_train.sh         # Convenience script for SMoA training
```

## Running SMoA

A quick example:

```bash
bash run_train.sh
```

You can override common settings with environment variables:

```bash
DATASET=boolq \
MODEL_NAME=/path/to/llama \
RANK_BUDGET=32 \
SMOA_NUM_BLOCKS=2 \
LORA_ALPHA=32 \
TARGET_MODULES=q_proj,v_proj,k_proj,up_proj,down_proj \
bash run_train_smoa.sh
```

Equivalent manual command:

```bash
torchrun --nproc_per_node=1 train.py \
  --dataset boolq \
  --peft_type smoa \
  --model_name /path/to/llama \
  --r_ab 32 \
  --smoa_num_blocks 2 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,v_proj,k_proj,up_proj,down_proj
```

## SMoA Arguments

SMoA reuses the common LoRA-style arguments and adds SMoA-specific options:

- `--peft_type smoa`: selects SMoA.
- `--r_ab`: total reference rank budget `r`.
- `--smoa_num_blocks`: number of spectral blocks `K`; defaults to `--l_num` when omitted.
- `--smoa_branch_ranks`: optional comma-separated per-block ranks, such as `8,8`; the values must sum to `--r_ab`.
- `--lora_alpha`: adapter scaling.
- `--lora_dropout`: dropout before the adapter update.
- `--target_modules`: comma-separated target module names.

Example with custom per-block ranks:

```bash
torchrun --nproc_per_node=1 train.py \
  --dataset boolq \
  --peft_type smoa \
  --r_ab 16 \
  --smoa_num_blocks 2 \
  --smoa_branch_ranks 8,8
```

## Notes

- SMoA currently supports standard `torch.nn.Linear` and Hugging Face `Conv1D` target layers.
- SVD preprocessing is done once when adapters are injected; it is not part of the inference path.
- The adapter can be merged into the base model through the existing PEFT merge flow.
