"""Normalize FLASK evaluation records and validate their identities."""
import json


def read_jsonl(path):
    with open(path, encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{number}: invalid JSON') from error


def index_records(rows):
    result = {}
    for row in rows:
        key = str(row['question_id'])
        if key in result:
            raise ValueError(f'Duplicate question_id: {key}')
        result[key] = row
    return result


def load_questions(path, limit=None):
    records = []
    for row in read_jsonl(path):
        identifier = row.get('question_id', row.get('idx'))
        if identifier is None:
            raise ValueError('A question needs question_id or idx')
        instruction = row.get('instruction', row.get('text'))
        skills = row.get('skill')
        domains = row.get('domain', [])
        reference = row.get('answer')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f'{identifier}: missing instruction')
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f'{identifier}: Prometheus evaluation needs a reference answer')
        if not isinstance(skills, list) or not skills or not all(isinstance(s, str) and s.strip() for s in skills):
            raise ValueError(f'{identifier}: missing FLASK skill annotations; use evaluation_set JSONL')
        if len(skills) != len(set(skills)):
            raise ValueError(f'{identifier}: duplicate skills')
        if not isinstance(domains, list) or not all(isinstance(d, str) and d.strip() for d in domains):
            raise ValueError(f'{identifier}: domain must be a list of names')
        records.append({'question_id': str(identifier), 'instruction': instruction,
                        'reference_answer': reference, 'skill': skills,
                        'domain': list(dict.fromkeys(domains))})
    index_records(records)  # Validate before limiting; don't hide duplicate IDs.
    if not records:
        raise ValueError('No evaluation questions found')
    return records[:limit] if limit else records


def load_rubrics(path):
    with open(path, encoding='utf-8-sig') as stream:
        rows = json.load(stream)
    result = {}
    for row in rows:
        name = row['Skill']
        if name in result:
            raise ValueError(f'Duplicate rubric: {name}')
        for score in range(1, 6):
            if not row['Scoring'].get(str(score)):
                raise ValueError(f'{name}: incomplete 1–5 rubric')
        result[name] = row
    return result


def validate_dataset(questions, rubrics):
    unknown = {skill for q in questions for skill in q['skill']} - rubrics.keys()
    if unknown:
        raise ValueError(f'Unknown FLASK skills: {sorted(unknown)}')
