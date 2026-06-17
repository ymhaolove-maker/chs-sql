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
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


finetuning_model_path = "/root/APP_LLM/text2sql/scripts/output_deepseek"
base_model_path = "/root/APP_LLM/text2sql/models/deepseek"

def parse_option():

    parser = argparse.ArgumentParser("")

    parser.add_argument('--input_sql_dev', type=str, default="/root/APP_LLM/text2sql/data/llama_std00_schema_linking_dev.json",
                        help='''
                            prompt template for sft model
                            ''')


    parser.add_argument('--model_result_dev', type=str,
                        default="/root/APP_LLM/text2sql/data/model_ds_chs_schema_linking_dev.json",
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
        question = row["conversations"][0]["value"]
        model_input = finetune_tokenizer(question, return_tensors="pt").to("cuda")
        with torch.no_grad():
            #response = finetune_tokenizer.decode(finetune_model.generate(**model_input, max_new_tokens=200, pad_token_id=2)[0], skip_special_tokens=True)
            outputs = finetune_model.generate(**model_input, pad_token_id=2, max_new_tokens=200, return_dict_in_generate=True, output_scores=True, num_beams=5, num_return_sequences=5)
            generated_ids = outputs.sequences
            logits = outputs.scores
            output_candidates = []
            result_set = set()
            model_result_sequence = ""
            for i in range(5):
                diff_mean_confs = []
                probs = [torch.softmax(log, dim=-1) for log in logits]
                response = finetune_tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                output_candidate = {"id": i, "response": response, "total_mean_confs": 0, "step_probs":[]}
                # 获取生成文本的token ID和对应的概率
                
                for ids, token_id in enumerate(generated_ids[i][len(model_input[0]):]):

                    if token_id == 2:
                        continue
                    token_prob = probs[ids][i][token_id].item()
                    step_probs = {"s": ids, "t": token_id, "lp": token_prob, "a": [], "mean_confs": 0}

                    # Only top 5
                    for value in probs[ids][i].topk(5): 
                        step_probs["a"].append(value)  # Use array instead of dict

                    step_probs["mean_confs"] = -sum(p for p in step_probs['a'][0]) / len(step_probs['a'][0])

                    output_candidate["step_probs"].append(step_probs)
                    #print(probs[ids][i].topk(5))  
                    #print(f"Trace ID: {i}, Token ID: {token_id}, Probability: {token_prob}")
                    diff_mean_confs.append(step_probs["mean_confs"])
                
                #total_mean_confs = sum(step_probs['mean_confs'] for step_probs in output_candidate["step_probs"]) / len(output_candidate["step_probs"])
                #total_mean_confs = sum(abs(diff_mean_confs[i] - diff_mean_confs[j]) for i in range(len(diff_mean_confs)) for j in range(i + 1, len(diff_mean_confs)))/(len(diff_mean_confs)-1)
                # 计算标准差
                import statistics
                std_confs = [float(x) for x in diff_mean_confs]
                total_mean_confs = statistics.stdev(std_confs) if len(std_confs) > 1 else 0
                output_candidate["total_mean_confs"] = total_mean_confs
                output_candidates.append(output_candidate)
                print(output_candidate["total_mean_confs"])
                print(str(output_candidate["response"].split("[/INST]\n")[1]).strip())
            sorted_response = sorted(output_candidates, key=lambda x:x['total_mean_confs'])

        
        max_prob_response = finetune_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response2 = str(max_prob_response.split("[/INST]\n")[1]).strip().replace("\n","")

        #{"singer":["name"]}   
        # try: 
        #     model_result_json2 = json.loads(response2.replace("'","\"").replace("(","").replace(")",""))
        # except json.decoder.JSONDecodeError as e:
        #     model_result_json2 = {}
        # for table_name in model_result_json2:
        #     from collections.abc import Sequence
        #     if isinstance(model_result_json2[table_name], Sequence) and len(model_result_json2[table_name]) == 0:
        #         result_set.add(table_name + ".*")
        #     elif isinstance(model_result_json2[table_name], Sequence) and len(model_result_json2[table_name]) != 0:
        #         for column_name in model_result_json2[table_name]:
        #             result_set.add(table_name + "." + column_name)

        #singer.name
        response_list2 = response2.split(",")
        for table_column_name in response_list2:
            result_set.add(table_column_name.strip())

        for output_id, output_candidate in enumerate(sorted_response):
            if output_candidate["total_mean_confs"] < 0.025:
               response = str(output_candidate["response"].split("[/INST]\n")[1]).strip().replace("\n","")
               #{"singer":["name"]}
               # try:   
               #      model_result_json = json.loads(response.replace("'","\"").replace("(","").replace(")",""))
               # except json.decoder.JSONDecodeError as e:
               #      model_result_json = {}
               # for table_name in model_result_json:
               #      if isinstance(model_result_json[table_name], Sequence) and len(model_result_json[table_name]) == 0:
               #          result_set.add(table_name + ".*")
               #      elif isinstance(model_result_json[table_name], Sequence) and len(model_result_json[table_name]) != 0:
               #          for column_name in model_result_json[table_name]:
               #              result_set.add(table_name + "." + column_name)

               #singer.name
               response_list = response.split(",")
               for table_column_name in response_list:
                    result_set.add(table_column_name.strip())

        model_result_sequence += ",".join(list(result_set))
        resutls_list.append(model_result_sequence)
        print(row["id"])
        print(model_result_sequence)


    # 将大模型的输出结果写出到文件中
    with open(model_result_dev, 'w', encoding='utf-8') as f:
        for item in resutls_list:
            f.write(item)
            f.write('\n')

if __name__ == "__main__":
    opt = parse_option()
    model_infer(opt.input_sql_dev, opt.model_result_dev)
