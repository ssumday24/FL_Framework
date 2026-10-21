"""Independent generation/scoring stages with immutable run provenance."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from .data import index_records, load_questions, load_rubrics, read_jsonl, validate_dataset
from .models import TextModel
from .prompts import PROMPT_VERSION, build_judge_prompt, parse_score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def new_jsonl(path):
    if path.exists():
        raise FileExistsError(f'{path} already exists; use a new run_id')
    temporary = path.with_suffix('.jsonl.partial')
    # Refuse concurrent/interrupted stage reuse rather than silently overwrite it.
    with temporary.open('x', encoding='utf-8') as stream:
        yield lambda row: (stream.write(json.dumps(row, ensure_ascii=False) + '\n'), stream.flush())
    temporary.replace(path)


def prepare(config):
    questions = load_questions(config.questions, config.limit)
    rubrics = load_rubrics(config.rubrics)
    validate_dataset(questions, rubrics)
    return questions, rubrics


def manifest_for(config):
    source_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    identity = {'config': config.as_dict(), 'questions_sha256': digest(config.questions),
                'rubrics_sha256': digest(config.rubrics), 'prompt_version': PROMPT_VERSION,
                'source_sha256': source_hashes}
    config.run_dir.mkdir(parents=True, exist_ok=True)
    path = config.run_dir / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            raise ValueError('Run configuration, inputs, or code changed; choose a new run_id')
        return manifest
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {'identity': identity, 'created_at': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(), 'packages': packages, 'stages': {}}
    write_json(path, manifest)
    return manifest


def save_stage(config, manifest, stage, output, metadata):
    manifest['stages'][stage] = {'completed_at': datetime.now(timezone.utc).isoformat(),
                                 'output_sha256': digest(output), 'model': metadata}
    write_json(config.run_dir / 'manifest.json', manifest)


def validate_artifact(config, manifest, stage, filename):
    path = config.run_dir / filename
    saved = manifest['stages'].get(stage)
    if not saved or not path.exists() or saved['output_sha256'] != digest(path):
        raise ValueError(f'{filename} is missing, incomplete, or changed after {stage}')
    return path


def matching_records(path, questions, run_id):
    records = index_records(read_jsonl(path))
    if records.keys() != {q['question_id'] for q in questions}:
        raise ValueError('Output question IDs differ from the configured evaluation set')
    if any(row.get('run_id') != run_id for row in records.values()):
        raise ValueError('Output belongs to another run')
    return records


def generate_answers(config, model_factory=TextModel):
    questions, _ = prepare(config)
    manifest = manifest_for(config)
    output = config.run_dir / 'answers.jsonl'
    with new_jsonl(output) as write:
        model = model_factory(config.generator, config.seed, config.project_root)
        try:
            for number, question in enumerate(questions, 1):
                text = model.generate(question['instruction'])
                if not text.strip():
                    raise ValueError(f'{question["question_id"]}: empty generated answer')
                write({'run_id': config.run_id, 'question_id': question['question_id'], 'text': text})
                print(f'Generate {number}/{len(questions)}', flush=True)
            metadata = model.metadata
        finally:
            model.close()
    save_stage(config, manifest, 'generate', output, metadata)


def score_answers(config, model_factory=TextModel):
    questions, rubrics = prepare(config)
    manifest = manifest_for(config)
    answers_path = validate_artifact(config, manifest, 'generate', 'answers.jsonl')
    answers = matching_records(answers_path, questions, config.run_id)
    output = config.run_dir / 'reviews.jsonl'
    with new_jsonl(output) as write:
        model = model_factory(config.judge, config.seed, config.project_root)
        try:
            for number, question in enumerate(questions, 1):
                scores, raw, errors = {}, {}, {}
                for skill in question['skill']:
                    prompt = build_judge_prompt(question, answers[question['question_id']]['text'], rubrics[skill])
                    # Runtime/model failures stop the stage; malformed grading output is recorded.
                    raw[skill] = model.generate(prompt)
                    try:
                        scores[skill] = parse_score(raw[skill])
                    except ValueError as error:
                        scores[skill] = None
                        errors[skill] = str(error)
                write({'run_id': config.run_id, 'question_id': question['question_id'],
                       'score': scores, 'raw': raw, 'errors': errors})
                print(f'Score {number}/{len(questions)}', flush=True)
            metadata = model.metadata
        finally:
            model.close()
    save_stage(config, manifest, 'score', output, metadata)
