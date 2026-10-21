"""Run from the repository root: python -m fl_framework --help."""
import argparse
import json

from .analysis import summarize
from .config import load_config
from .environment import check_environment
from .pipeline import generate_answers, prepare, score_answers


def main():
    parser = argparse.ArgumentParser(description='Mistral → FLASK rubrics → Prometheus → score report')
    parser.add_argument('stage', choices=['doctor', 'validate', 'generate', 'score', 'summarize', 'all'])
    parser.add_argument('--config')
    args = parser.parse_args()
    if args.stage == 'doctor':
        result = check_environment()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['ready_for_model_smoke_test'] else 1
    if not args.config:
        parser.error('--config is required for this stage')
    config = load_config(args.config)
    questions, rubrics = prepare(config)
    if args.stage == 'validate':
        print(json.dumps({'run_id': config.run_id, 'questions': len(questions),
                          'judge_calls': sum(len(q['skill']) for q in questions),
                          'rubric_count': len(rubrics), 'output': str(config.run_dir),
                          'generator': config.generator.model_id, 'judge': config.judge.model_id},
                         ensure_ascii=False, indent=2))
        return
    if args.stage in {'generate', 'all'}:
        generate_answers(config)
    if args.stage in {'score', 'all'}:
        score_answers(config)
    if args.stage in {'summarize', 'all'}:
        result = summarize(config)
        print(json.dumps({k: v for k, v in result.items() if k not in {'skills', 'domains'}}, ensure_ascii=False, indent=2))
    print(f'Outputs: {config.run_dir}')


if __name__ == '__main__':
    raise SystemExit(main())
