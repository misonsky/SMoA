import csv
import re
from functools import reduce
from glob import glob
import jsonlines
import numpy as np
import evaluate
import os
import json

from tqdm import tqdm

eval_folder_path = 'results_hime/'

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

ANSWER_TRIGGER = "The answer is"

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
    
def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer
def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        pred = preds[1]
    else:
        pred = preds[-1]
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    if len(pred) == 0:
        return INVALID_ANS
    if answer_flag:
        pred = pred[0]
    else:
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred

def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['siqa', 'arcc', 'arce', 'obqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellas':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winog':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'gsm8k':
        return clean_answer(sentence)

eval_tasks = [
'boolq',
'piqa',
'siqa',
'arcc',
'arce',
'obqa',
'hellas',
'winog',
'gsm8k'
]
all_data = {}
for eval_task in eval_tasks:
    eval_paths = glob(eval_folder_path)
    eval_path = eval_paths[0]
    jsonl_files = glob(f'results_hime/*{eval_task}*/*.jsonl')
    csv_header = ['folder_name', 'acc', 'config']
    csv_data = []
    for json_file in tqdm(jsonl_files):
        print(json_file)
        json_data = []
        lines = []
        with open(json_file, 'r') as f:
            for line in f.readlines():
                lines.append(line)
        lines = [line for line in lines if len(line.strip()) > 0]
        for line in lines:
            try:
                json_data.append(json.loads(line))
            except Exception as e:
                print(e)
        configs = json_data[:2]
        contents = [line for line in json_data if line.__class__ is dict and 'context' in line.keys()]
        answers = []
        for content in contents:
            pred = extract_answer(f'{eval_task}', content['pred'])
            gt = extract_answer(f'{eval_task}', content['gt'])
            if eval_task =="gsm8k":
                answers.append(is_correct(pred,gt))
            else:
                answers.append(pred==gt)
            
        acc = np.asarray(answers).mean()
        # folder_name = json_file.split(os.sep)[2]
        # folder_name = re.sub(r'-\d\d\d\d-\d\d-\d\d.*','',folder_name)+os.sep+json_file.split(os.sep)[-1]
        _json_paths = json_file.split(os.sep)
        _json_paths[-1] = _json_paths[-1].replace(eval_task, '')
        folder_name = os.sep.join(_json_paths)
        # folder_name = json_file.replace(eval_task, '')
        csv_row = [folder_name, acc*100.0, configs]
        if folder_name in all_data.keys():
            all_data[folder_name][eval_task]=acc*100
        else:
            all_data[folder_name] = {eval_task:acc*100}
        csv_data.append(csv_row)
    with open(f'{eval_path}/results_{eval_task}.csv', 'w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)

type_name = [n for n in eval_folder_path.split(os.sep) if len(n)>0][-1]
with open(f'{eval_folder_path}/{type_name}_results_all.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['model',*eval_tasks, 'avg'])
    for folder_name in all_data.keys():
        row = [folder_name]
        for eval_task in eval_tasks:
            row.append(all_data[folder_name].get(eval_task,-1))
        avg_val = reduce(lambda x,y: x+y, row[1:])/len(row[1:])
        writer.writerow(row+[avg_val])


