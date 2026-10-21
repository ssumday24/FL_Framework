"""Converted from ColabNotebooks/FLASK 추론용 데이터셋을 허깅페이스에 업로드.ipynb.

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
    os.chdir(PROJECT_ROOT.as_posix() + '/FLASK/input_data')

    # Notebook cell 2
    import json
    file_path = PROJECT_ROOT.as_posix() + '/FLASK/input_data/flask_evaluation_raw.jsonl'
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    # Notebook cell 3
    print(data[1634])

    # Notebook cell 4


    # Notebook cell 5
    from datasets import Dataset
    dataset = Dataset.from_dict({'question_id': [item['question_id'] for item in data], 'text': [item['text'] for item in data], 'task': [item['task'] for item in data], 'answer': [item['answer'] for item in data]})

    # Notebook cell 6
    from huggingface_hub import HfApi, Repository
    write_token = os.environ['HF_TOKEN']
    api = HfApi()
    repo_id = 'prayslaks/flask_evaluation'
    dataset.push_to_hub(repo_id, token=write_token)


if __name__ == "__main__":
    main()
