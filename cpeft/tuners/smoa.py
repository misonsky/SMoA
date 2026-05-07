# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..config import PeftConfig, PeftType
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
    transpose,
)
from .tuners_utils import BaseTuner, BaseTunerLayer


def _split_rank(total_rank: int, num_blocks: int) -> List[int]:
    if num_blocks <= 0:
        raise ValueError("`num_blocks` must be positive.")
    if total_rank < 0:
        raise ValueError("`r` must be non-negative.")

    base_rank = total_rank // num_blocks
    remainder = total_rank % num_blocks
    return [base_rank + (1 if idx < remainder else 0) for idx in range(num_blocks)]


def _sanitize_adapter_name(adapter_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", adapter_name)


def _split_sizes(total_size: int, num_blocks: int) -> List[int]:
    base_size = total_size // num_blocks
    remainder = total_size % num_blocks
    return [base_size + (1 if idx < remainder else 0) for idx in range(num_blocks)]


@dataclass
class SMoAConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Total SMoA rank budget."})
    num_blocks: int = field(default=2, metadata={"help": "Number of spectral blocks."})
    branch_ranks: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Optional per-block ranks. Must sum to `r` when provided."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with SMoA."
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Global SMoA scaling."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Dropout applied before the SMoA update."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for SMoA. Can be 'none', 'all' or 'lora_only'."})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Modules apart from SMoA layers to set as trainable and save."},
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the low-rank factors."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={"help": "Optional layer indexes to transform."},
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Layer pattern used together with `layers_to_transform`."},
    )

    def __post_init__(self):
        if self.num_blocks <= 0:
            raise ValueError("`num_blocks` must be positive.")
        if self.branch_ranks is not None:
            if len(self.branch_ranks) != self.num_blocks:
                raise ValueError("`branch_ranks` must have `num_blocks` elements.")
            if sum(self.branch_ranks) != self.r:
                raise ValueError("`branch_ranks` must sum to `r`.")
        self.peft_type = PeftType.SMOA


class SMoALayer(BaseTunerLayer):
    def __init__(self, in_features: int, out_features: int, fan_in_fan_out: bool = False):
        self.r = {}
        self.num_blocks = {}
        self.branch_ranks = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.block_buffer_names = {}
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out

    def update_layer(
        self,
        adapter_name,
        r,
        num_blocks,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        branch_ranks=None,
    ):
        ranks = branch_ranks if branch_ranks is not None else _split_rank(r, num_blocks)
        if len(ranks) != num_blocks:
            raise ValueError("`branch_ranks` must have one entry per block.")
        if sum(ranks) != r:
            raise ValueError("The sum of `branch_ranks` must equal the total rank `r`.")

        self.r[adapter_name] = r
        self.num_blocks[adapter_name] = num_blocks
        self.branch_ranks[adapter_name] = ranks
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout.update(
            nn.ModuleDict({adapter_name: nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()})
        )

        in_sizes = _split_sizes(self.in_features, num_blocks)
        out_sizes = _split_sizes(self.out_features, num_blocks)
        lora_a_modules = nn.ModuleList()
        lora_b_modules = nn.ModuleList()
        for rank, in_size, out_size in zip(ranks, in_sizes, out_sizes):
            if rank > 0:
                lora_a_modules.append(nn.Linear(in_size, rank, bias=False))
                lora_b_modules.append(nn.Linear(rank, out_size, bias=False))
            else:
                lora_a_modules.append(nn.Identity())
                lora_b_modules.append(nn.Identity())

        self.lora_A.update(nn.ModuleDict({adapter_name: lora_a_modules}))
        self.lora_B.update(nn.ModuleDict({adapter_name: lora_b_modules}))

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        for rank, lora_a, lora_b in zip(
            self.branch_ranks[adapter_name],
            self.lora_A[adapter_name],
            self.lora_B[adapter_name],
        ):
            if rank <= 0:
                continue
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_b.weight)

    def _clear_block_buffers(self, adapter_name: str):
        for names in self.block_buffer_names.get(adapter_name, []):
            for name in names:
                if name in getattr(self, "_buffers", {}):
                    self._buffers.pop(name)
                if hasattr(self, name):
                    delattr(self, name)
        self.block_buffer_names[adapter_name] = []

    @staticmethod
    def _coordinate_order(left_vectors: torch.Tensor, singular_values: torch.Tensor, size: int) -> torch.Tensor:
        if left_vectors.numel() == 0:
            return torch.arange(size, device=left_vectors.device)
        spectral_ids = torch.arange(left_vectors.shape[1], device=left_vectors.device, dtype=left_vectors.dtype)
        energy = left_vectors.pow(2) * singular_values.to(left_vectors.dtype).unsqueeze(0).pow(2)
        denom = energy.sum(dim=1).clamp_min(torch.finfo(energy.dtype).eps)
        score = (energy * spectral_ids.unsqueeze(0)).sum(dim=1) / denom
        return torch.argsort(score)

    def rebuild_spectral_blocks(self, adapter_name: str):
        if adapter_name not in self.num_blocks:
            return

        self._clear_block_buffers(adapter_name)
        num_blocks = self.num_blocks[adapter_name]
        effective_weight = transpose(self.weight.detach(), self.fan_in_fan_out).to(torch.float32)
        device = effective_weight.device
        q = min(effective_weight.shape)
        sanitized_adapter_name = _sanitize_adapter_name(adapter_name)

        if q == 0:
            row_order = torch.arange(self.out_features, device=device)
            col_order = torch.arange(self.in_features, device=device)
        else:
            try:
                u, singular_values, vh = torch.linalg.svd(effective_weight, full_matrices=False)
            except RuntimeError:
                u, singular_values, vh = torch.linalg.svd(effective_weight.cpu(), full_matrices=False)
                u = u.to(device)
                singular_values = singular_values.to(device)
                vh = vh.to(device)
            row_order = self._coordinate_order(u, singular_values, self.out_features)
            col_order = self._coordinate_order(vh.transpose(0, 1), singular_values, self.in_features)

        row_blocks = torch.tensor_split(row_order, num_blocks)
        col_blocks = torch.tensor_split(col_order, num_blocks)
        block_names = []
        for idx, (row_idx, col_idx) in enumerate(zip(row_blocks, col_blocks)):
            row_name = f"smoa_rows_{sanitized_adapter_name}_{idx}"
            col_name = f"smoa_cols_{sanitized_adapter_name}_{idx}"
            anchor_name = f"smoa_anchor_{sanitized_adapter_name}_{idx}"
            anchor = effective_weight[row_idx][:, col_idx].contiguous()
            self.register_buffer(row_name, row_idx.to(dtype=torch.long), persistent=False)
            self.register_buffer(col_name, col_idx.to(dtype=torch.long), persistent=False)
            self.register_buffer(anchor_name, anchor, persistent=False)
            block_names.append((row_name, col_name, anchor_name))

        self.block_buffer_names[adapter_name] = block_names

    def _ensure_spectral_blocks(self, adapter_name: str):
        if adapter_name not in self.block_buffer_names or not self.block_buffer_names[adapter_name]:
            self.rebuild_spectral_blocks(adapter_name)

    def _get_delta_weight_matrix(self, adapter_name: str, dtype: torch.dtype, device: torch.device):
        if adapter_name not in self.lora_A:
            return torch.zeros(self.out_features, self.in_features, dtype=dtype, device=device)

        self._ensure_spectral_blocks(adapter_name)
        delta_weight = torch.zeros(self.out_features, self.in_features, dtype=dtype, device=device)
        for rank, lora_a, lora_b, names in zip(
            self.branch_ranks[adapter_name],
            self.lora_A[adapter_name],
            self.lora_B[adapter_name],
            self.block_buffer_names[adapter_name],
        ):
            if rank <= 0:
                continue
            row_name, col_name, anchor_name = names
            row_idx = getattr(self, row_name).to(device=device)
            col_idx = getattr(self, col_name).to(device=device)
            anchor = getattr(self, anchor_name).to(device=device, dtype=lora_a.weight.dtype)
            branch_weight = lora_b.weight @ lora_a.weight
            block_delta = (branch_weight * anchor).to(dtype)
            delta_weight[row_idx[:, None], col_idx[None, :]] = block_delta

        return delta_weight * self.scaling[adapter_name]

    def get_delta_weight(self, adapter_name: str):
        delta_weight = self._get_delta_weight_matrix(
            adapter_name,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        return transpose(delta_weight, self.fan_in_fan_out)


class SMoAModel(BaseTuner):
    prefix = "lora_"

    def _check_new_adapter_config(self, config: SMoAConfig) -> None:
        if len(self.peft_config) > 1 and config.bias != "none":
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(smoa_config, key):
        if isinstance(smoa_config.target_modules, str):
            target_module_found = re.fullmatch(smoa_config.target_modules, key)
        else:
            target_module_found = any(
                re.match(f".*\\.{target_key}$", key) for target_key in smoa_config.target_modules
            ) or any(target_key == key for target_key in smoa_config.target_modules)
            is_using_layer_indexes = getattr(smoa_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(smoa_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\\.(\\d+)\\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(smoa_config.layers_to_transform, int):
                            target_module_found = layer_index == smoa_config.layers_to_transform
                        else:
                            target_module_found = layer_index in smoa_config.layers_to_transform
                        break
                    target_module_found = False

        return target_module_found

    def _create_and_replace(
        self,
        smoa_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        if optionnal_kwargs.get("loaded_in_8bit", False) or optionnal_kwargs.get("loaded_in_4bit", False):
            raise ValueError("This SMoA implementation currently supports only standard torch Linear/Conv1D layers.")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": smoa_config.r,
            "num_blocks": smoa_config.num_blocks,
            "branch_ranks": smoa_config.branch_ranks,
            "lora_alpha": smoa_config.lora_alpha,
            "lora_dropout": smoa_config.lora_dropout,
            "fan_in_fan_out": smoa_config.fan_in_fan_out,
            "init_lora_weights": smoa_config.init_lora_weights,
            "bias": bias,
        }

        if isinstance(target, SMoALayer):
            target.update_layer(
                adapter_name,
                smoa_config.r,
                smoa_config.num_blocks,
                smoa_config.lora_alpha,
                smoa_config.lora_dropout,
                smoa_config.init_lora_weights,
                branch_ranks=smoa_config.branch_ranks,
            )
            target.rebuild_spectral_blocks(adapter_name)
            return

        new_module = self._create_new_module(smoa_config, adapter_name, target, **kwargs)
        self._replace_module(parent, target_name, new_module, target)
        new_module.rebuild_spectral_blocks(adapter_name)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if hasattr(child, "bias") and child.bias is not None:
            new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state

        new_module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self) -> None:
        active_adapter = self._get_active_adapter()
        bias = self.peft_config[active_adapter].bias

        for name, parameter in self.model.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad = False

        if bias == "none":
            return
        if bias == "all":
            for name, parameter in self.model.named_parameters():
                if "bias" in name:
                    parameter.requires_grad = True
            return
        if bias == "lora_only":
            for module in self.model.modules():
                if isinstance(module, SMoALayer) and hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = True
            return

        raise NotImplementedError

    @staticmethod
    def _create_new_module(smoa_config, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = smoa_config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = smoa_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                "Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )

        return Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, SMoALayer):
                module.disable_adapters = not enabled
            elif isinstance(module, ModulesToSaveWrapper):
                module.disable_adapters = not enabled

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.peft_config:
            if self.peft_config[active_adapter].bias != "none":
                warnings.warn(
                    "Careful, disabling adapter layers with bias configured to be "
                    f"'{self.peft_config[active_adapter].bias}' does not restore the exact base model output."
                )
        self._set_adapter_layers(enabled=False)

    def _get_active_adapter(self) -> str:
        active_adapter = None
        for module in self.model.modules():
            if isinstance(module, SMoALayer):
                active_adapter = module.active_adapter

        return active_adapter

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, SMoALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def merge_and_unload(self):
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if isinstance(target, SMoALayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


class Linear(nn.Linear, SMoALayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        num_blocks: int = 2,
        branch_ranks: Optional[List[int]] = None,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SMoALayer.__init__(self, in_features=in_features, out_features=out_features, fan_in_fan_out=fan_in_fan_out)
        self.weight.requires_grad = False

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name,
            r,
            num_blocks,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            branch_ranks=branch_ranks,
        )
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A:
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter).to(self.weight.dtype)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A:
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter).to(self.weight.dtype)
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = self.lora_dropout[self.active_adapter](x.to(self.weight.dtype))
            delta_weight = self._get_delta_weight_matrix(
                self.active_adapter,
                dtype=x.dtype,
                device=x.device,
            )
            result = result + F.linear(x, delta_weight, bias=None)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        return result.to(previous_dtype)
