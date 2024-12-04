import datasets
import numpy as np
import ast
import torch
from torch.utils.data import Dataset
import random
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer
import evaluate

class DatasetConfig:
	def __init__(self,**kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)

class WebNLGDataset:
    def __init__(self, dataset_path, tokenizer, model, dataset_config):
        self.tokenizer = tokenizer

        # Load metrics
        self.bleu_metric = evaluate.load('bleu')
        self.meteor_metric = evaluate.load('meteor')
        #self.exact_match_metric = evaluate.load('exact_match')

        self.dataset = datasets.load_dataset("csv", data_files=dataset_path, keep_default_na=False, split="train")
        self.config = dataset_config

        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model, padding=True)
     
        self.dataset.shuffle(seed=dataset_config.seed)

    def preprocess(self):
        def tokenize_delta(examples):
            input_text = [f"{self.config.task_description} {text}" for text in examples['input_text']]
            # pad each data point to the max length
            # tokenizer, return an id for each token (called input_ids), with attention mask in pytorch tensor 
            tokenized_inputs = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

            if self.config.mode == 'training':
                # in case of multiRef select one randomly
                selected_outputs = [random.choice(ast.literal_eval(outputs) if isinstance(outputs, str) else outputs) for outputs in examples['output_text']]
                tokenized_targets = self.tokenizer(selected_outputs, padding=True,truncation=True, return_tensors="pt")
                # we assign input_ids of the output to be labels for our tokenized inputs
                tokenized_inputs['labels'] = tokenized_targets['input_ids']
                
            return tokenized_inputs

        self.dataset = self.dataset.map(tokenize_delta, batched=True, load_from_cache_file=False)
        if self.config.mode == 'training':
            self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        else:
            self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        return self.dataset


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Decode predictions
        decoded_preds = [self.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in predictions]

        # Decode labels
        #pad_token_id = self.tokenizer.pad_token_id
        # Replace -100 in labels with pad_token_id before decoding
        #decoded_labels = [self.tokenizer.decode([token_id if token_id != -100 else pad_token_id for token_id in label], skip_special_tokens=True, clean_up_tokenization_spaces=True) for label in labels]

        #references_for_bleu = [[label] for label in decoded_labels]  # BLEU expects a list of reference lists for each prediction
        references_for_metrics = [ast.literal_eval(label_list) for label_list in labels]
        
        # Compute metrics
        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=references_for_metrics)
        meteor_result = self.meteor_metric.compute(predictions=decoded_preds, references=references_for_metrics)
        #exact_match_result = self.exact_match_metric.compute(predictions=decoded_preds, references=references_for_metrics)

        # Prepare the final result
        result = {
            "bleu": bleu_result["bleu"],
            "meteor": meteor_result["meteor"],
            #"exact_match": exact_match_result["exact_match"],
        }
        
        return result



