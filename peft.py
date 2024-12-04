from copy import deepcopy
from opendelta import LoraModel, AdapterModel, PrefixModel, CompacterModel, SoftPromptModel, AutoDeltaModel

class ModuleConfig:
	def __init__(self,**kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)

class Modules(ModuleConfig):
    def __init__(self, model, module_config) -> None:
        self.config = module_config
        self.model = model
     
    def get_module(self, mode, model):
        config = self.config
        if mode == "lora":
            delta_model = LoraModel(backbone_model=model,
                                    lora_r=config.lora_r,
                                    lora_alpha=config.lora_alpha,
                                    lora_dropout=config.lora_dropout)
        elif mode == "adapter":
            delta_model = AdapterModel(backbone_model=model,
                                       bottleneck_dim=config.adapter_bottleneck_dim)
        elif mode == "prefix":
            delta_model = PrefixModel(backbone_model=model,
                                        prefix_token_num=config.prefix_token_num,
                                        reparameterize=config.prefix_reparameterize,
                                        embed_dim= config.prefix_embed_dim,
                                        mid_dim = config.prefix_mid_dim)
        elif mode == "compacter":
            delta_model = CompacterModel(backbone_model=model,
                                        reduction_factor=config.compacter_reduction_factor,
                                        hypercomplex_division=config.compacter_hypercomplex) 
        elif mode == "softprompt":
            delta_model = SoftPromptModel(backbone_model=model,
                                       soft_token_num=config.softprompt_token_num, init_range=config.softprompt_init_range)
        else:
              print("Please choose from the following: {loar, adapter, prefix, compacter, softprompts}")
        return delta_model
    
    def task_module(self, mode):
        model = self.model
        task_delta = self.get_module(mode, model)
        # freeze the model and only train the PEFT module
        task_delta.freeze_module()
        task_delta.log()
        return [model, task_delta]