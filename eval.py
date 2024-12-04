import configparser
import argparse 
import os
import json
import gc

import torch
from torch.utils.data import DataLoader

from utils import *
from prepare_dataset import DatasetConfig, WebNLGDataset

from opendelta import AutoDeltaModel
from transformers import (T5ForConditionalGeneration,
                        T5Tokenizer) 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Portability Tests")
parser.add_argument("-c","--config",dest="filename", help="Pass eval config file",metavar="FILE")
parser.add_argument("--exp_name", help="Experiment name",type=str, default="./exp")
parser.add_argument("--model_name", help="Model name (Huggingface Hub model name)",type=str, default="t5-base")
parser.add_argument("--module", help="PEFT Technique", type=str, nargs="?")
parser.add_argument("--learning_steps", help="Training Steps", type=int, default=0)
parser.add_argument("--multi_stage_tune", help="whether to do multi-stage finetuning or not", action="store_true")
parser.add_argument("--submission_mode", help="run inference for submission files (i.e. files without labels)", action="store_true")
parser.add_argument("--submission_file", help="path to the submission file to predict",type=str, default="./datasets/test_data/FORGe/D2T-1-CFA_forge.csv")
parser.add_argument("--decoding_strategy", help="whether to do a fancy decoding strategy or not", action="store_true")
parser.add_argument("--seed", help="Random seed for experiment reproduction",type=int, default=49)


args = parser.parse_args()
config.read(args.filename)

seed  = args.seed
#import from utils
seed_everything(seed)

task_prompt = config['exp_config']['task_prompt']
new_token_generate_len = int(config['eval_config']['new_token_generate_len'])
dataset_path = config['eval_config']['test_dataset'] if not args.submission_mode else args.submission_file
dataset_name = config['exp_config']['dataset_name']

def peft(module):
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, mlm=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    #load xl in bf16, others in fp32
    #model = T5ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16) if args.model_name== "google/flan-t5-xl" \
                                                        #else T5ForConditionalGeneration.from_pretrained(args.model_name) 
    
    if not args.multi_stage_tune:
        _ = AutoDeltaModel.from_finetuned(f"./exp/{args.exp_name}/{args.model_name.replace('/','')}/{module}", backbone_model=model, check_hash=False)
    else:
        _ = AutoDeltaModel.from_finetuned(f"./exp/{args.exp_name}/{args.model_name.replace('/','')}/multi_stage_{module}", backbone_model=model, check_hash=False)

    return model, tokenizer

def fulltune(module):
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, mlm=False)

    if not args.multi_stage_tune:
        finetuned_model_name = f"./exp/{args.exp_name}/{args.model_name.replace('/','')}/{module}"
    else:
        finetuned_model_name = f"./exp/{args.exp_name}/{args.model_name.replace('/','')}/multi_stage_{module}"

    #load xl in fp16, others in fp32
    model = T5ForConditionalGeneration.from_pretrained(finetuned_model_name, torch_dtype=torch.bfloat16) if args.model_name== "google/flan-t5-xl" \
                                                        else T5ForConditionalGeneration.from_pretrained(finetuned_model_name) 

    return model, tokenizer

def offshelf():
    '''
    load and test the model driectly off the shelf
    '''
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, mlm=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    return model, tokenizer


def get_dataloader(tokenizer, model):

    dataconfig = DatasetConfig(
    mode = "evaluating",
    task_description = task_prompt,
    seed = seed,
    )

    evalset = WebNLGDataset(dataset_path, tokenizer, model, dataconfig)

    compute_metrics = evalset.compute_metrics
    data_collator = evalset.data_collator
    evalset_processed = evalset.preprocess()
    

    eval_loader = DataLoader(evalset_processed, batch_size=16, shuffle=False, collate_fn=data_collator)
    
    return evalset_processed, eval_loader, compute_metrics



def main():
    if args.module == "fulltune":
        model, tokenizer = fulltune(args.module)
    elif args.module == "offshelf":
        model, tokenizer = offshelf()
    else:
        model, tokenizer = peft(args.module)
    
    model.to(device)
    model.eval()

    evalset_dataset, eval_dataloader, compute_metrics = get_dataloader(tokenizer, model)
    preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            if not args.decoding_strategy:
                predictions = model.generate(input_ids, 
                                attention_mask = mask, 
                                early_stopping= True,
                                max_length = 400,
                                num_beams = 4,  
                                )
        
            else:
                predictions = model.generate(input_ids,
                            attention_mask=mask,
                            temperature=0.1,  # Adjusts randomness: lower values make output more deterministic.
                            max_length=400,
                            top_k=100,  # Keeps the top k most likely next words. Limits the sample pool.
                            top_p=0.95,  # Nucleus sampling: keeps the top p cumulative probability. Helps in dynamic sampling.
                            repetition_penalty=0.8,  # Penalizes repetition to encourage diversity.
                            length_penalty=1.0,  # Adjusts length: values > 1 encourage longer outputs; values < 1 shorter.
                            num_return_sequences=1,  # Number of independently computed outputs to generate.
                            no_repeat_ngram_size=5,  # Prevents repetition of n-grams. Use 2 for bigrams.
                            eos_token_id=model.config.eos_token_id,  # End-of-sentence token ID to signal completion.
                            pad_token_id=model.config.pad_token_id,  # Padding token for sequences shorter than max_length.
                            use_cache=True  # Enables caching to speed up token generation.
                            )                                                             
            preds.extend(predictions)
        
        if not args.submission_mode:
            all_labels = evalset_dataset['output_text']
            results = compute_metrics((preds, all_labels))
            print(f"Eval Results: {results}")

        decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in preds]
        prediction_df = evalset_dataset.add_column("predictions", decoded_preds).to_pandas()
        prediction_df = prediction_df.drop(columns=['input_ids', 'attention_mask'])

        if not args.submission_mode:
            eval_results = {"model": args.model_name,
                            "dataset" : dataset_name,
                            "module": args.module,
                            "task prompt": task_prompt,
                            "learning steps": 0 if args.module=="offshelf" else args.learning_steps, 
                            "BLEU": f"{results['bleu']:0.3f}",
                            "METEOR": f"{results['meteor']:0.3f}",
                            #"EM": f"{results['exact_match']:0.3f}",
                            }
        
        if not args.submission_mode:
            # Check if the 'prediction' directory exists and create it if it doesn't.
            if not os.path.exists("./prediction"):
                os.makedirs("./prediction")

            file_path = f"./prediction/{args.exp_name}_seed_{args.seed}_predictions_metrics.json"

            with open(file_path, "a") as json_file:
                eval_result_dict = json.dumps(eval_results)
                json_file.write(eval_result_dict + '\n')

            prediction_df.to_csv(f"./prediction/{args.exp_name}_{dataset_name}_{args.model_name.replace('/','')}_{args.module}_seed_{args.seed}_predictions_texts.csv", index=False)
        else:
            if not os.path.exists(f"./prediction/submission_files/{args.model_name}"):
                os.makedirs(f"./prediction/submission_files/{args.model_name}")

            prediction_file_path = f"./prediction/submission_files/{args.model_name}/{dataset_path.split('/')[-1].rsplit('.', 1)[0]}_{args.module}_predictions.csv"
            prediction_df.to_csv(prediction_file_path, index=False)

if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()





