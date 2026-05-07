import json
import os
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging
from .templates import (
    SST2Template,
    CopaTemplate,
    BoolQTemplate,
    BoolQTemplateV2,
    BoolQTemplateV3,
    MultiRCTemplate,
    CBTemplate,
    WICTemplate,
    WSCTemplate,
    ReCoRDTemplate,
    ReCoRDTemplateGPT3,
    SGLUERTETemplate,
    SQuADv2Template,
    DROPTemplate,
    COLATemplate,
    MNLITemplate,
    MRPCTemplate,
    QNLITemplate,
    QQPTemplate
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name):
    task_class={"sst2":SST2Dataset,
                "copa":COPADataset,
                "multirc":MULTIRCDataset,
                "cb":CBDataset,
                "wic":WICDataset,
                "wsc":WSCDataset,
                "record":RECORDDataset,
                "rte":RTEDataset,
                "squad":SQUADDataset,
                "drop":DROPDataset
                }
    if task_name in task_class:
        return task_class[task_name]
    else:
        return None

@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
        self.samples = {"train": None, "valid": None,"test":None}
    def get_task_name(self):
        return self.subtask
        
    def load_dataset():
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       templates = {0: Template}
       return templates[template_version]
    def get_reference(self,split):
        examples = self.samples[split]
        references = [example.correct_candidate for example in examples]
        return references
    def get_candidates(self,split):
        examples = self.samples[split]
        candidates = [example.candidates for example in examples]
        return candidates
    def sample_subset(self, data_split="train", seed=42, num=100):
        samples = self.samples[data_split]
        if num <=0:
            return samples
        lens = len(samples)
        np.random.seed(seed)
        index = np.random.permutation(lens).tolist()[:num]
        sub_samples = []
        for i in index:
            sub_samples.append(samples[i])
        return sub_samples
    def build_sample(self, example):
        return 
    
    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    train_sep = "\n\n"
    
    def load_dataset(self,split):
        d = load_dataset('glue', 'sst2',split=split)
        samples = [self.build_sample(example) for example in d]
        if "train" in split:
            self.samples["train"] = samples
        elif "valid" in split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples
    
    # for generative tasks, candidates are []
    
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()
        
    
class COPADataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def load_dataset(self, split):
        examples=load_dataset(path,split=split)
        samples = [self.build_sample(example) for example in examples]
        if "train" in load_split:
            self.samples["train"] = samples
        elif "valid" in load_split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples

    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BOOLQDataset(Dataset):
   
    def load_dataset(self, split):
        d = load_dataset("boolq")
        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MULTIRCDataset(Dataset):
    
    def load_dataset(self, split):
        d = load_dataset("super_glue", "multirc")

        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    
    def load_dataset(self,split):
        d = load_dataset("super_glue", "cb")
        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    def load_dataset(self, split):
        d = load_dataset("super_glue", "wic")
        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    
    def load_dataset(self, split):
        d = load_dataset("super_glue", "wsc.fixed")
        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class RECORDDataset(Dataset):
    
    def load_dataset(self, split):
        d = load_dataset("super_glue", "record")
        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    
    def load_dataset(self, split):
        d = load_dataset("super_glue", "rte")
        if "train" in split:
            train_samples = [self.build_sample(example) for example in d["train"]]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example) for example in d["validation"]]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: SGLUERTETemplate}[template_version]()

 
class SQUADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def load_dataset(self,split):
        dataset = load_dataset("squad")
        if "train" in split:
            train_samples = [self.build_sample(example, idx) for idx, example in enumerate(dataset["train"])]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(dataset["validation"])]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def load_dataset(self,split):
        dataset = load_dataset("drop")
        if "train" in split:
            train_samples = [self.build_sample(example, idx) for idx, example in enumerate(dataset["train"])]
            self.samples["train"] = train_samples
        elif "valid" in split:
            valid_samples =  [self.build_sample(example, idx) for idx, example in enumerate(dataset["validation"])]
            self.samples["valid"] = valid_samples
    
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class COLADataset(Dataset):
    train_sep = "\n\n"
    
    def load_dataset(self,split):
        d = load_dataset('glue', 'cola',split=split)
        samples = [self.build_sample(example) for example in d]
        if "train" in split:
            self.samples["train"] = samples
        elif "valid" in split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples
    
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: COLATemplate}[template_version]()

class MNLIDataset(Dataset):
    train_sep = "\n\n"
    
    def load_dataset(self,split):
        d = load_dataset('glue', 'mnli',split=split)
        samples = [self.build_sample(example) for example in d]
        if "train" in split:
            self.samples["train"] = samples
        elif "valid" in split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples
    
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1, 2])
        
    def get_template(self, template_version=0):
        return {0: MNLITemplate}[template_version]()

class MRPCDataset(Dataset):
    train_sep = "\n\n"
    
    def load_dataset(self, split):
        d = load_dataset('glue', 'mrpc',split=split)
        samples = [self.build_sample(example) for example in d]
        if "train" in split:
            self.samples["train"] = samples
        elif "valid" in split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples
    
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: MRPCTemplate}[template_version]()

class QNLIDataset(Dataset):
    train_sep = "\n\n"
    
    def load_dataset(self, split):
        d = load_dataset('glue', 'qnli',split=split)
        samples = [self.build_sample(example) for example in d]
        if "train" in split:
            self.samples["train"] = samples
        elif "valid" in split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples
    
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: QNLITemplate}[template_version]()

class QQPDataset(Dataset):
    train_sep = "\n\n"
    def load_dataset(self, split):
        d = load_dataset('glue', 'qqp',split=split)
        samples = [self.build_sample(example) for example in d]
        if "train" in split:
            self.samples["train"] = samples
        elif "valid" in split:
            self.samples["valid"] = samples
        else:
            self.samples["test"] = samples
    
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: QQPTemplate}[template_version]()
