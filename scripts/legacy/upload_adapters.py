"""Converted from ColabNotebooks/OpenFedLLM 학습 완료된 어댑터 모델 한꺼번에 업로드.ipynb.

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


    # Notebook cell 1
    write_token = os.environ['HF_TOKEN']

    # Notebook cell 2
    base = 'mistral'
    id = 'flask_evaluation_2000_fedavg_c5s3_i10_b4a1_l2048_r16a32_20241129133706'

    # Notebook cell 3
    from huggingface_hub import hf_api

    # Notebook cell 4
    hf_api.create_repo(repo_id=f'prayslaks/{base}_{id}', token=write_token)

    # Notebook cell 5
    hf_api.upload_folder(folder_path=f'{PROJECT_ROOT.as_posix()}/OpenFedLLM/output/{base}_{id}', repo_id=f'prayslaks/{base}_{id}', token=write_token)


if __name__ == "__main__":
    main()
