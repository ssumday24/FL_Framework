"""Converted from ColabNotebooks/OpenFedLLM을 이용한 학습 과정.ipynb.

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
    run_command('python --version', locals())

    # Notebook cell 1
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

    # Notebook cell 2


    # Notebook cell 3
    os.chdir(PROJECT_ROOT.as_posix() + '/OpenFedLLM')

    # Notebook cell 4


    # Notebook cell 5


    # Notebook cell 6
    read_token = os.environ['HF_TOKEN']
    write_token = os.environ['HF_TOKEN']
    import huggingface_hub

    # Notebook cell 7
    run_command('bash setup.sh', locals())

    # Notebook cell 8


    # Notebook cell 9
    run_command('bash training_scripts/run_sft_mistral_flask.sh', locals())


if __name__ == "__main__":
    main()
