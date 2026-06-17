from dataclasses import dataclass, field
import json
from typing import Dict, Optional, List
import transformers
from torch.utils.data import Dataset
import os
import sys
import torch
from peft import AutoPeftModelForCausalLM,PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, Trainer, DataCollatorForSeq2Seq
import argparse
import re
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


finetuning_model_path = "/root/APP_LLM/text2sql/scripts/output_ds_chs_sql_epoch3"
base_model_path = "/root/APP_LLM/text2sql/models/deepseek"

def parse_option():

    parser = argparse.ArgumentParser("")

    parser.add_argument('--input_sql_dev', type=str, default="/root/APP_LLM/text2sql/data/ds_chs_sql_dev.json",
                        help='''
                            prompt template for sft model
                            ''')


    parser.add_argument('--model_result_dev', type=str,
                        default="/root/APP_LLM/text2sql/data/model_ds_chs_sql_dev.json",
                        help="sql statement result.")



    opt = parser.parse_args()

    return opt


def model_infer(input_sql_dev, model_result_dev):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        #load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    finetune_tokenizer = AutoTokenizer.from_pretrained(base_model_path,padding_side="left", trust_remote_code=True)
    #finetune_tokenizer.add_eos_token = True
    finetune_tokenizer.pad_token_id = 2

    finetune_model = PeftModel.from_pretrained(base_model, finetuning_model_path)
    finetune_model.eval()

    # 读取validation文件
    with open(input_sql_dev, encoding='utf-8') as f:
        dataset = json.load(f)

    resutls_list = []

    for row in dataset:
        if int(row["id"]) <= -1:
            continue
        question = row["conversations"][0]["value"]
        model_input = finetune_tokenizer(question, return_tensors="pt").to("cuda")
        input_len = model_input["input_ids"].shape[1]
        with torch.inference_mode():
            #response = finetune_tokenizer.decode(finetune_model.generate(**model_input, max_new_tokens=200, pad_token_id=2)[0], skip_special_tokens=True)
            outputs = finetune_model.generate(**model_input, pad_token_id=2, max_new_tokens=200, return_dict_in_generate=True, output_scores=True, num_beams=3, num_return_sequences=3, do_sample=False)
            generated_ids = outputs.sequences
            #logits = outputs.scores
            #output_candidates = []
            #max_prob_mean_conf = 0.0
            #for i in range(3):
            #    diff_mean_confs = []
            #    response = finetune_tokenizer.decode(generated_ids[i][len(model_input[0]):], skip_special_tokens=True)
            #    output_candidate = {"id": i, "response": response, "total_mean_confs": 0, "step_probs":[]}
            #    # 获取生成文本的token ID和对应的概率
            #    
            #    for ids, token_id in enumerate(generated_ids[i][input_len:]):
#
#
                    #if token_id == 2:
                    #    continue
                    #step_log_probs = torch.log_softmax(logits[ids][i].float(), dim=-1)
                    #token_prob = step_log_probs[token_id].exp().item()
                    #step_probs = {"s": ids, "t": token_id, "lp": token_prob, "a": [], "mean_confs": 0}

                    
                    #topk_log_values, _ = step_log_probs.topk(5)
                    #topk_values = topk_log_values.exp()
                    #step_probs["a"] = topk_values.tolist()
                    #step_probs["mean_confs"] = -topk_log_values.mean().item()
                    #output_candidate["step_probs"].append(step_probs)
                    #diff_mean_confs.append(step_probs["mean_confs"])
                
                # 计算标准差
                #import statistics
                #std_confs = [float(x) for x in diff_mean_confs]
                #total_mean_confs = statistics.stdev(std_confs) if len(std_confs) > 1 else 0
                #output_candidate["total_mean_confs"] = total_mean_confs
                #output_candidates.append(output_candidate)
                #print(output_candidate["total_mean_confs"])
                #print(str(output_candidate["response"]).strip())

            #sorted_response = sorted(output_candidates, key=lambda x:x['total_mean_confs'])

        #####voting######
        max_prob_response = finetune_tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        response2 = str(max_prob_response).strip()
        if ";" in response2:
            response2 = response2.split(";")[0]
        if "```sql" in response2:
            response2 = response2.split("```sql")[1]
        response2 = re.sub(r'\s+', ' ', response2).strip()
        with open(model_result_dev, 'a', encoding='utf-8') as f:
            f.write(response2)
            f.write('\n')
            f.flush()
        #resutls_list.append(response2)
        print(row["id"])
        print(response2)
        print("######## choose max prob one ##########")
        del outputs, generated_ids, model_input 
        torch.cuda.empty_cache()

    # 将大模型的输出结果写出到文件中
    #with open(model_result_dev, 'w', encoding='utf-8') as f:
    #    for item in resutls_list:
    #        f.write(item)
    #        f.write('\n')

if __name__ == "__main__":
    opt = parse_option()
    model_infer(opt.input_sql_dev, opt.model_result_dev)
