# GEM24-DCU-NLG-Small

## Code Structure
```
GEM24-DCU-NLG-Small/
├── config.ini            # Configuration file containing settings for training, PEFT modules, and experiment folders
├── prepare_dataset.py    # Script for dataset preparation, including classes for preprocessing WebNLG data
├── peft.py               # Script for initialising/attaching PEFT modules (LoRA and Adapters) to pretrained models.
├── finetuner.py          # Script containing Hugging Face Trainer setup and training argument configurations
├── run_finetune.py       # Main script where all modules are imported and PEFT/full finetuning is executed
├── eval.py               # Script to evaluate trained PEFT modules/full models.
├── utils.py              # Script with helper functions, such as setting random seeds
├── README.md             # Documentation file for the project
└── environment.yml       # YAML file specifying the environment dependencies

```

## Run Code:
#### Environment:
```
conda env create environment.yml
conda activate struct2text_env
```


#### Training:
The datasets paths to train and validate the model are specified in the `config.ini` file.
```
python run_finetune.py -c config.ini --exp_name "$exp_name" --model_name "$model" --module "$module" --learning_steps "$learning_steps" --seed "$seed"
#`--exp_name` (`$exp_name`): The name of the experiment, used for organising logs and outputs, and saving finetuned models.
#`--model_name` (`$model`): The name (a HuggingFace model name) or path of the pretrained model to finetune.
# `--module` (`$module`): The specific module or component to finetune, choices are ('fulltune' 'lora' 'adapter').
# `--learning_steps` (`$learning_steps`): The number of training steps for model finetuning.
# `--seed` (`$seed`): A seed value for random number generation, ensuring reproducibility.
```
#### Evaluating:
The dataset to evaluate/test the trained model is specified in the `config.ini` file.
```
python eval.py -c config.ini --exp_name "$exp_name" --model_name "$model" --module "$module" --learning_steps "$learning_steps" --seed "$seed"
# We use `--exp_name` (`$exp_name`), `--model_name` (`$model`), `--module` (`$module`), `--learning_steps` (`$learning_steps`), and `--seed` (`$seed`)
# to resemble the path of the finetuned PEFT module/model resulted from the training above. 
# Obviously, we don't need them if we can pass the full name of the finetuned model, 
# but because we run the two scripts (`run_finetune.py` and `eval.py`) together frequently, 
# we pass these arguments to recombine the paths. 
```
#### Quick Evaluation for Submissions:
For quick evaluation on submission files, use the `--submission_mode` flag and specify the file with `--submission_file`. This loads the data path from the command line instead of the `config.ini` file, which is useful when working with multiple submission files.

```
python eval.py -c config.ini --exp_name "$model_exp_name" --model_name "$model" --module "$module" --submission_mode --submission_file "$file" --seed "$seed"
```


