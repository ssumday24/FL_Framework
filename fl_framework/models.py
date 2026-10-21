"""Lazy Transformers/PEFT backend shared by generator and evaluator."""
import gc
from pathlib import Path
from .prompts import JUDGE_SYSTEM


class TextModel:
    def __init__(self, config, seed=42, project_root=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        set_seed(seed)
        self.config = config
        self.torch = torch
        self.model = None
        self.tokenizer = None
        def location(value):
            if value and project_root:
                candidate = Path(project_root) / value
                if candidate.exists():
                    return str(candidate.resolve())
            return value
        device = config.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        if device == 'cpu' and config.load_in_4bit:
            raise ValueError('This backend supports 4-bit loading only on CUDA')
        dtype_name = config.dtype
        if dtype_name == 'auto':
            dtype_name = 'float16' if device == 'cuda' else 'float32'
        if device == 'cuda' and dtype_name == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support bfloat16')
        dtype = getattr(torch, dtype_name)
        model_id = location(config.model_id)
        adapter_id = location(config.adapter_id)
        if adapter_id:
            from peft import PeftConfig
            adapter_options = {'revision': config.adapter_revision}
            if config.adapter_subfolder:
                adapter_options['subfolder'] = config.adapter_subfolder
            adapter_config = PeftConfig.from_pretrained(adapter_id, **adapter_options)
            base = location(adapter_config.base_model_name_or_path)
            if base.rstrip('/\\') != model_id.rstrip('/\\'):
                raise ValueError(f'Adapter base {base!r} differs from configured model {model_id!r}')
        tokenizer_id = location(config.tokenizer_id) or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, revision=config.tokenizer_revision or config.revision,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = {'revision': config.revision, 'torch_dtype': dtype,
                   'device_map': {'': 0} if device == 'cuda' else 'cpu',
                   'attn_implementation': 'sdpa',
                   'low_cpu_mem_usage': True,
                   'trust_remote_code': False}
        if config.load_in_4bit:
            options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **options)
        if adapter_id:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_id, **adapter_options)
        self.model.eval()
        self.model.config.use_cache = True
        self.metadata = {'model_id': model_id, 'requested_revision': config.revision,
                         'resolved_revision': getattr(self.model.config, '_commit_hash', None),
                         'adapter_id': adapter_id, 'adapter_revision': config.adapter_revision,
                         'adapter_subfolder': config.adapter_subfolder,
                         'tokenizer_id': tokenizer_id, 'device': device, 'dtype': dtype_name,
                         'load_in_4bit': config.load_in_4bit,
                         'double_quant': config.load_in_4bit,
                         'attention': 'sdpa'}

    def generate(self, text):
        config = self.config
        if config.prompt_format == 'prometheus':
            # Prometheus 2's documented Mistral single-turn system/user format.
            prompt = f'[INST] {JUDGE_SYSTEM}\n{text} [/INST]'
            inputs = self.tokenizer(prompt, add_special_tokens=True, return_tensors='pt', truncation=False)
        elif config.prompt_format == 'alpaca':
            # Match OpenFedLLM/utils/template.py up to the response boundary.
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{text} \n\n### Response: '
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=False)
        else:
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt', truncation=False)
        length = inputs['input_ids'].shape[1]
        context = getattr(self.model.config, 'max_position_embeddings', None)
        if length > config.max_input_tokens or (context and length + config.max_new_tokens > context):
            raise ValueError(f'Prompt has {length} tokens and exceeds input/context budget; no silent truncation')
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        options = {'max_new_tokens': config.max_new_tokens, 'do_sample': config.temperature > 0,
                   'pad_token_id': self.tokenizer.pad_token_id}
        if config.temperature > 0:
            options['temperature'] = config.temperature
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **options)
        return self.tokenizer.decode(generated[0, length:], skip_special_tokens=True).strip()

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
