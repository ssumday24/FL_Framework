"""Converted from ColabNotebooks/PEFT 어댑터 모델 추론 시연 및 테스트.ipynb.

Migration baseline: original experiment choices and cell order are preserved.
Review README.md before running; upload scripts write to Hugging Face.
"""
import os
try:
    from ._runtime import PROJECT_ROOT, run_command
except ImportError:
    from _runtime import PROJECT_ROOT, run_command


def main():
    import os
    os.chdir(PROJECT_ROOT)
    # Notebook cell 0
    run_command('nvidia-smi', locals())

    # Notebook cell 1
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

    # Notebook cell 2
    read_token = os.environ['HF_TOKEN']

    # Notebook cell 3


    # Notebook cell 4
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    # Notebook cell 5
    model_1 = 'prayslaks/python_code_instructions_18k_alpaca_20000_fedavg_c4s2_i10_b4a1_l512_r32a64_20241030071530'
    model_2 = 'prayslaks/alpaca-cleaned_20000_fedavg_c4s2_i10_b4a1_l512_r32a64_20241111092921'
    model_3 = 'prayslaks/flask_evaluation_2000_fedavg_c5s4_i10_b4a1_l2048_r16a32_20241120145940'
    model_4 = 'prayslaks/wikimedia_wikipedia_1M_1000000_fedavg_c4s2_i10_b4a1_l2048_r16a32_20241201054722'
    adapter_id = model_4

    # Notebook cell 6
    config = PeftConfig.from_pretrained(adapter_id, subfolder='checkpoint-10')
    print(config.base_model_name_or_path)

    # Notebook cell 7
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

    # Notebook cell 8
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, subfolder='checkpoint-10')

    # Notebook cell 9
    model = PeftModel.from_pretrained(model, adapter_id, subfolder='checkpoint-10')

    # Notebook cell 10
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Notebook cell 12
    input_text = ['Explain about Anarchism.', 'Explain about Albedo', 'Explain about ASCII art', 'Explain about Apple II?', 'Explain about Blade Runner']
    for text in input_text:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=200)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text + '\n\n')


if __name__ == "__main__":
    main()
