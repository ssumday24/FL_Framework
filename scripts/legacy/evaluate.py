"""Converted from ColabNotebooks/FLASK 평가 단계.ipynb.

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


    # Notebook cell 2
    os.chdir(PROJECT_ROOT.as_posix() + '/FLASK/gpt_review')

    # Notebook cell 3


    # Notebook cell 4


    # Notebook cell 5


    # Notebook cell 6
    import jsonlines
    import os

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

    # Notebook cell 9
    evaluation_code = PROJECT_ROOT.as_posix() + '/FLASK/gpt_review/gpt4_eval.py'
    model_1 = 'Mistral_FlaskDataset_c5s3'
    model_2 = 'Prometheus_FlaskDataset_c5s3'
    model_name = model_1
    iteration = 10
    question_num = 1000
    question_file = f'{PROJECT_ROOT.as_posix()}/FLASK/evaluation_set/flask_evaluation_q{question_num}.jsonl'
    answer_file = f'{PROJECT_ROOT.as_posix()}/flask_inference_outputs/{model_name}/q{question_num}_i{iteration}.jsonl'
    output_directory = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}'
    output_review_file = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}/review.jsonl'
    output_error_file = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}/error.jsonl'

    # Notebook cell 10
    create_empty_directory(output_directory)
    create_empty_jsonl_file(output_review_file)
    create_empty_jsonl_file(output_error_file)

    # Notebook cell 11
    run_command('python $evaluation_code --question-file $question_file --answer-file $answer_file --output-review-file $output_review_file --output-error-file $output_error_file --max-tokens 1024', locals())

    # Notebook cell 13
    evaluation_code = PROJECT_ROOT.as_posix() + '/FLASK/gpt_review/gpt4_eval.py'
    model_1 = 'Mistral_FlaskDataset'
    model_2 = 'Prometheus_FlaskDataset'
    model_name = model_1
    iteration = 10
    question_num = 1000
    question_file = f'{PROJECT_ROOT.as_posix()}/FLASK/evaluation_set/flask_evaluation_q{question_num}.jsonl'
    answer_file = f'{PROJECT_ROOT.as_posix()}/flask_inference_outputs/{model_name}/q{question_num}_i{iteration}.jsonl'
    output_directory = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}'
    output_review_file = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}/review.jsonl'
    output_error_file = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}/error.jsonl'

    # Notebook cell 14
    create_empty_directory(output_directory)
    create_empty_jsonl_file(output_review_file)
    create_empty_jsonl_file(output_error_file)

    # Notebook cell 15
    run_command('python $evaluation_code --question-file $question_file --answer-file $answer_file --output-review-file $output_review_file --output-error-file $output_error_file --max-tokens 1024', locals())


if __name__ == "__main__":
    main()
