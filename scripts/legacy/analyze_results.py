"""Converted from ColabNotebooks/FLASK 평가 결과 CSV 출력.ipynb.

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
    os.chdir(PROJECT_ROOT.as_posix() + '/FLASK/gpt_review')

    # Notebook cell 3
    model_1 = 'Prometheus_FlaskDataset_c5s3'
    model_2 = 'Prometheus_FlaskDataset'
    model_3 = 'Mistral_FlaskDataset_c5s3'
    model_4 = 'Mistral_FlaskDataset'
    model_name = model_1
    question_num = 1000
    iteration = 10
    review_file = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}/review.jsonl'
    output_path = f'{PROJECT_ROOT.as_posix()}/flask_evaluation_outputs/{model_name}/q{question_num}_i{iteration}/'

    # Notebook cell 4
    run_command('python aggregate_skill.py --file $review_file --output $output_path', locals())

    # Notebook cell 5
    run_command('python aggregate_domain.py --file $review_file --output $output_path', locals())

    # Notebook cell 7
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Notebook cell 8
    def draw_radar_chart(file_path):
        data = pd.read_csv(file_path)
        categories = ['correctness', 'factuality', 'commonsense', 'comprehension', 'insightfulness', 'completeness', 'readability', 'conciseness']
        filtered_data = data[data['skill'].isin(categories)]
        values = filtered_data['avg'].values
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        name = file_path.split('.csv')[0].split('/')[-1]
        ax.plot(angles, values, color='b', linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, values, color='b', alpha=0.25)
        ax.set_yticks(np.arange(1, 6, 1))
        ax.set_yticklabels(range(1, 6), fontsize=16, weight='bold')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{category}' for category in categories], fontsize=14, weight='bold', color='blue')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Performance BenchMark by Skills', size=20, weight='bold', position=(0.5, 1.1))
        plt.show()

    # Notebook cell 9
    def draw_bar_chart(file_path):
        data = pd.read_csv(file_path)
        categories = ['Humanities', 'Language', 'Culture', 'Technology']
        filtered_data = data[data['domain'].isin(categories)]
        values = filtered_data['avg'].values
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='#FF6F61', linewidth=2, linestyle='solid', label='Prometheus')
        ax.fill(angles, values, color='#FF6F61', alpha=0.25)
        ax.set_yticks(np.arange(1, 6, 1))
        ax.set_yticklabels(range(1, 6), fontsize=12, weight='bold')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{category}' for category in categories], fontsize=14, weight='bold', color='darkblue')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Performance Benchmark by Domain - Prometheus', size=20, weight='bold', position=(0.5, 1.1))
        plt.show()

    # Notebook cell 11
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset/q1000_i0/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset/q1000_i0/content_domain.csv')

    # Notebook cell 12
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i1/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i1/content_domain.csv')

    # Notebook cell 13
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i3/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i3/content_domain.csv')

    # Notebook cell 14
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i5/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i5/content_domain.csv')

    # Notebook cell 15
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i7/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i7/content_domain.csv')

    # Notebook cell 16
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i10/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Prometheus_FlaskDataset_c5s3/q1000_i10/content_domain.csv')

    # Notebook cell 18
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset/q1000_i0/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset/q1000_i0/content_domain.csv')

    # Notebook cell 19
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i1/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i1/content_domain.csv')

    # Notebook cell 20
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i3/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i3/content_domain.csv')

    # Notebook cell 21
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i5/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i5/content_domain.csv')

    # Notebook cell 22
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i7/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i7/content_domain.csv')

    # Notebook cell 23
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i10/review_skill.csv')
    draw_bar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/Mistral_FlaskDataset_c5s3/q1000_i10/content_domain.csv')

    # Notebook cell 25
    draw_radar_chart(PROJECT_ROOT.as_posix() + '/flask_evaluation_outputs/gpt4_review_skill.csv')


if __name__ == "__main__":
    main()
