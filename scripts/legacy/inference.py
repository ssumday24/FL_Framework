"""Converted from ColabNotebooks/FLASK 추론 단계.ipynb.

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
    # Notebook cell 1
    run_command('nvidia-smi', locals())

    # Notebook cell 2
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Notebook cell 3


    # Notebook cell 4


    # Notebook cell 6
    read_token = os.environ['HF_TOKEN']

    # Notebook cell 8


    # Notebook cell 9
    import jsonlines

    def create_empty_jsonl_file(file_path):
        data = []
        if os.path.exists(file_path):
            print(f"Error: file '{file_path}' exists.")
        else:
            with jsonlines.open(file_path, mode='w') as writer:
                writer.write_all(data)
            print(f'Empty JSONL file created at: {file_path}')

    def create_empty_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f'Empty Directory created at: {directory_path}')
        else:
            print(f"Error: file '{directory_path}' exists.")

    # Notebook cell 11
    model_path = 'prayslaks/mistral_flask_evaluation_2000_fedavg_c5s3_i10_b4a1_l2048_r16a32_20241129133706'
    model_1 = 'Mistral_FlaskDataset_c5s3'
    model_2 = 'Prometheus_FlaskDataset_c5s3'
    save_model_name = model_1
    iteration = 1
    question_num = 1000
    inference_code = PROJECT_ROOT.as_posix() + '/FLASK/model_output/inference_peft.py'
    peft_subfolder = f'checkpoint-{iteration}'
    question_file = f'{PROJECT_ROOT.as_posix()}/FLASK/input_data/flask_evaluation_q{question_num}.jsonl'
    answer_directory = f'{PROJECT_ROOT.as_posix()}/flask_inference_outputs/{save_model_name}'
    answer_file = f'{PROJECT_ROOT.as_posix()}/flask_inference_outputs/{save_model_name}/q{question_num}_i{iteration}.jsonl'

    # Notebook cell 12
    create_empty_directory(answer_directory)
    create_empty_jsonl_file(answer_file)

    # Notebook cell 13
    run_command('python $inference_code --model-path $model_path --model-id alpaca --question-file $question_file --answer-file $answer_file --num-gpus 1 --peft-subfolder $peft_subfolder --use-peft True', locals())

    # Notebook cell 15
    model_path = 'mistralai/Mistral-7B-Instruct-v0.3'
    model_1 = 'Mistral_FlaskDataset'
    model_2 = 'Prometheus_FlaskDataset'
    save_model_name = model_1
    question_num = 1000
    inference_code = PROJECT_ROOT.as_posix() + '/FLASK/model_output/inference.py'
    iteration = 0
    question_file = f'{PROJECT_ROOT.as_posix()}/FLASK/input_data/flask_evaluation_q{question_num}.jsonl'
    answer_file = f'{PROJECT_ROOT.as_posix()}/flask_inference_outputs/{save_model_name}/q{question_num}_i{iteration}.jsonl'

    # Notebook cell 16
    create_empty_jsonl_file(answer_file)

    # Notebook cell 17
    run_command('python $inference_code --model-path $model_path --model-id alpaca --question-file $question_file --answer-file $answer_file --num-gpus 1', locals())


if __name__ == "__main__":
    main()
