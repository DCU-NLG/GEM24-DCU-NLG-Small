import configparser
import argparse 
import os
from pathlib import Path
import gc

import torch
from opendelta import AutoDeltaModel
from copy import deepcopy

from utils import *
from prepare_dataset import DatasetConfig, WebNLGDataset
from finetuner import FintunerConfig, Finetuner
from peft import ModuleConfig, Modules
from datasets import Dataset
from transformers import (T5ForConditionalGeneration, T5Tokenizer, 
                            )

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Fluency Tests")
parser.add_argument("-c","--config",dest="filename", help="Pass a training config file",metavar="FILE")
parser.add_argument("--exp_name", help="Experiment name",type=str, default="peft_instruct_tests")
parser.add_argument("--model_name", help="Model name (Huggingface Hub model name)",type=str, default="t5-base")
parser.add_argument("--module", help="PEFT Technique", type=str, nargs="?")
parser.add_argument("--learning_steps", help="Training Steps", type=int, default=500)
parser.add_argument("--multi_stage_tune", help="whether to do multi-stage finetuning or not", action="store_true")
parser.add_argument("--source_finetuned_path", help="path to finetuned model to apply multi stage on",type=str, nargs="?")
parser.add_argument("--seed", help="Random seed for experiment reproduction",type=int, default=49)


args = parser.parse_args()
config.read(args.filename)

seed = args.seed

#import from utils
seed_everything(seed)

##########################################################
#                  Set up Configs                        #
##########################################################

                        #################
                        #    model/exp  #
                        #################

exp_name = args.exp_name
resume_exp = config['exp_config']['resume_exp']
exp_folder = config['exp_config']['exp_folder']
cache_dir = config['exp_config']['cache_dir']

outfolder, cache_dir = Path(exp_folder), Path(cache_dir)
outfolder.mkdir(exist_ok=resume_exp)
cache_dir.mkdir(exist_ok=resume_exp)
os.environ['WANDB_PROJECT'] = config['exp_config']['wandb_project_name']
train_data_path = config['exp_config']['train_data_path']
eval_data_path = config['exp_config']['eval_data_path']
task_prompt = config['exp_config']['task_prompt']

                        #################
                        #    training   #
                        #################
task_do_train = config.getboolean('task_config', 'task_do_train')
task_do_eval = config.getboolean('task_config', 'task_do_eval')
task_per_device_train_batch_size = 8 if (args.model_name== "google/flan-t5-xl" or args.model_name== "google-t5/t5-3b") else int(config['task_config']['task_per_device_train_batch_size'])
task_per_device_eval_batch_size = 4 if (args.model_name == "google/flan-t5-xl" or args.model_name== "google-t5/t5-3b") else int(config['task_config']['task_per_device_eval_batch_size'])
task_learning_rate = float(config['task_config']['task_learning_rate'])
task_max_steps = args.learning_steps           #int(config['task_config']['task_max_steps'])
task_warmup_steps = int(0.1 * task_max_steps)  #int(config['task_config']['task_warmup_steps'])
task_save_steps = int(0.5 * task_max_steps)    #int(config['task_config']['task_save_steps'])
task_eval_steps = int(0.5 * task_max_steps)    #int(config['task_config']['task_eval_steps'])
task_gradient_accumulation_steps = int(config['task_config']['task_gradient_accumulation_steps'])
task_gradient_checkpointing = config.getboolean('task_config', 'task_gradient_checkpointing')
task_weight_decay = float(config['task_config']['task_weight_decay'])


                        #################
                        # delta modules #
                        #################
lora_modified_modules = list(map(str, config['modules_config']['loar_modified_modules'].strip('[]').split(',')))
lora_r = int(config['modules_config']['lora_rank'])
lora_alpha = int(config['modules_config']['lora_alpha'])
lora_dropout = float(config['modules_config']['lora_dropout'])

adapter_modified_modules = list(map(str, config['modules_config']['adapter_modified_modules'].strip('[]').split(',')))
adapter_bottleneck_dim = int(config['modules_config']['adapter_bottleneck_dim'])

prefix_token_num = int(config['modules_config']['prefix_token_num'])
prefix_reparameterize = bool(config['modules_config']['prefix_reparameterize'])
prefix_embed_dim = int(config['modules_config']['prefix_embed_dim'])
prefix_mid_dim = int(config['modules_config']['prefix_mid_dim'])

##########################################################
#                  Set up Model/data                     #
##########################################################

tokenizer = T5Tokenizer.from_pretrained(args.model_name, mlm=False)
#load xl in fp16, others in fp32
model = T5ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,device_map="auto") if (args.model_name== "google/flan-t5-xl" or args.model_name== "google-t5/t5-3b") \
        else T5ForConditionalGeneration.from_pretrained(args.model_name,device_map="auto")

# functions for Multi stage finetuning: load a finetuned model or a finetuned PEFT module
def peft(module):
    model = T5ForConditionalGeneration.from_pretrained(args.model_name,device_map="auto")     
    delta_model = AutoDeltaModel.from_finetuned(f"./exp/{args.source_finetuned_path}/{args.model_name.replace('/','')}/{module}", backbone_model=model, check_hash=False)
    return model, delta_model

def fulltune(module):
    finetuned_model_name = f"./exp/{args.source_finetuned_path}/{args.model_name.replace('/','')}/{module}"

    #load xl in fp16, others in fp32
    model = T5ForConditionalGeneration.from_pretrained(finetuned_model_name, torch_dtype=torch.bfloat16,device_map="auto") if (args.model_name== "google/flan-t5-xl" or args.model_name== "google-t5/t5-3b") \
                                                        else T5ForConditionalGeneration.from_pretrained(finetuned_model_name,device_map="auto")

    return model
##########################################################
#                  set up modules                        #
##########################################################
if not args.multi_stage_tune:
    modules_config = ModuleConfig( lora_modified_modules = lora_modified_modules,
                                lora_r = lora_r,
                                lora_alpha = lora_alpha,
                                lora_dropout = lora_dropout,
                                adapter_modified_modules = adapter_modified_modules,
                                adapter_bottleneck_dim = adapter_bottleneck_dim,
                                prefix_token_num = prefix_token_num,
                                prefix_reparameterize = prefix_reparameterize,
                                prefix_embed_dim = prefix_embed_dim,
                                prefix_mid_dim = prefix_mid_dim,
                                )

    modules = Modules(model, modules_config)

##########################################################
#                  Task module Training                  #
##########################################################

dataconfig = DatasetConfig(
    mode = "training", # to use the train split, if you want to evaluate, comment 
    task_description = task_prompt,
    seed = seed,
    )

train_set = WebNLGDataset(train_data_path, tokenizer, model, dataconfig)
eval_set = WebNLGDataset(eval_data_path, tokenizer, model, dataconfig)

# if we are at multi stage finetuning, we can delete the main model instance
# As we loaded just to use it to prepare our dataset collator.
# we will load a new finetuned version of this model later, to start our multi stage tuning.
if args.multi_stage_tune:
    del model

collator = train_set.data_collator

finetuneset = train_set.preprocess()
evalset = eval_set.preprocess()

task_finetune_config = FintunerConfig(
            task_module = True,
            output_dir = outfolder / exp_name,
            do_train = task_do_train,
            do_eval = task_do_eval,
            per_device_train_batch_size = task_per_device_train_batch_size,
            per_device_eval_batch_size = task_per_device_eval_batch_size,
            learning_rate = task_learning_rate,
            max_steps = task_max_steps,
            warmup_steps = task_warmup_steps,
            save_steps = task_save_steps,
            eval_steps = task_eval_steps,
            gradient_accumulation_steps = task_gradient_accumulation_steps,
            gradient_checkpointing = task_gradient_checkpointing,
            run_name = f"{exp_name}_{args.model_name}_{args.module}",
            data_collator = collator,
            weight_decay=task_weight_decay,            
            seed = seed
            )
 


if args.module == "fulltune":
                        ###################
                        # Full Finetuning #
                        ###################
    if not args.multi_stage_tune: 
        full_finetune_model = model
        task_outfolder = outfolder / exp_name / args.model_name.replace("/","") / "fulltune"
    else:
        print("Load a finetuned model for multi stage tuning...")
        full_finetune_model = fulltune(args.module)
        task_outfolder = outfolder / exp_name / args.model_name.replace("/","") / "multi_stage_fulltune"

    finetuner = Finetuner(full_finetune_model, tokenizer, finetuneset, evalset, task_finetune_config)
    finetuner.finetune()
    full_finetune_model.save_pretrained(task_outfolder)
  
else:
                        ###################
                        # PEFT Finetuning #
                        ###################
    if not args.multi_stage_tune: 
        model_, delta_model = modules.task_module(args.module)
        task_outfolder = outfolder / exp_name / args.model_name.replace("/","") / args.module
    else:
        print("Load a finetuned PEFT for multi stage tuning...")
        model_, delta_model = peft(args.module)
        task_outfolder = outfolder / exp_name / args.model_name.replace("/","") / f"multi_stage_{args.module}"

    finetuner = Finetuner(model_, tokenizer, finetuneset, evalset, task_finetune_config)
    finetuner.finetune()
    #convert model's FP32 to calculate its MD5 hash,
    #a checking step to know if a PEFT is loaded with the same model trained on
    model_.to(torch.float32)
    delta_model.save_finetuned(task_outfolder)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
