"""Named-axis aggregation and a standalone HTML score report."""
import csv
from html import escape
from statistics import mean

from .pipeline import (manifest_for, matching_records, prepare,
                       validate_artifact, write_json)


def summarize(config):
    questions, rubrics = prepare(config)
    manifest = manifest_for(config)
    path = validate_artifact(config, manifest, 'score', 'reviews.jsonl')
    reviews = matching_records(path, questions, config.run_id)
    skills = {name: [] for name in rubrics}
    domains = {}
    failed = {name: 0 for name in rubrics}
    domain_failed = {}
    for question in questions:
        scores = reviews[question['question_id']]['score']
        if scores.keys() != set(question['skill']):
            raise ValueError(f'{question["question_id"]}: score keys differ from assigned skills')
        for domain in question['domain']:
            domains.setdefault(domain, [])
            domain_failed.setdefault(domain, 0)
        for skill, score in scores.items():
            if score is None:
                failed[skill] += 1
                for domain in question['domain']:
                    domain_failed[domain] += 1
                continue
            if type(score) is not int or not 1 <= score <= 5:
                raise ValueError('Scores must be integers 1–5 or null')
            skills[skill].append(score)
            for domain in question['domain']:
                domains[domain].append(score)
    def rows(groups, failures):
        return [{'name': name, 'count': len(values), 'failed': failures[name],
                 'avg': mean(values) if values else None} for name, values in groups.items()]
    skill_rows = rows(skills, failed)
    domain_rows = rows(domains, domain_failed)
    all_scores = [score for values in skills.values() for score in values]
    summary = {
        'run_id': config.run_id, 'question_count': len(questions),
        'valid_score_count': len(all_scores), 'failed_score_count': sum(failed.values()),
        'micro_avg': mean(all_scores) if all_scores else None,
        'macro_skill_avg': mean([r['avg'] for r in skill_rows if r['avg'] is not None]) if all_scores else None,
        'judge': config.judge.model_id, 'generator': config.generator.model_id,
        'skills': skill_rows, 'domains': domain_rows,
    }
    for filename, values in (('skills.csv', skill_rows), ('domains.csv', domain_rows)):
        with (config.run_dir / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['name', 'count', 'failed', 'avg'])
            writer.writeheader()
            writer.writerows(values)
    write_json(config.run_dir / 'summary.json', summary)
    write_report(config.run_dir / 'report.html', summary)
    return summary


def write_report(path, summary):
    def table(records):
        lines = []
        for row in records:
            avg = row['avg']
            label = f'{avg:.3f}' if avg is not None else 'N/A'
            width = avg / 5 * 100 if avg is not None else 0
            lines.append(f'<tr><th scope="row">{escape(row["name"])}</th>'
                         f'<td><div class="track"><div class="bar" style="width:{width:.2f}%"></div></div></td>'
                         f'<td>{label}</td><td>{row["count"]}</td><td>{row["failed"]}</td></tr>')
        return '<table><thead><tr><th>항목</th><th>점수 막대 (0–5)</th><th>평균</th><th>유효 채점 수</th><th>실패</th></tr></thead><tbody>' + ''.join(lines) + '</tbody></table>'
    micro = summary['micro_avg']
    average = f'{micro:.3f}' if micro is not None else 'N/A'
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(summary['run_id'])} 평가 결과</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#182333}}h1{{font-size:28px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.track{{width:260px;max-width:30vw;background:#edf0f5;height:16px}}.bar{{height:100%;background:#3679c8}}p{{max-width:85ch}}</style>
<h1>{escape(summary['run_id'])}</h1>
<p>답변 모델: {escape(summary['generator'])}<br>평가 모델: {escape(summary['judge'])}</p>
<p>문항 {summary['question_count']}개 · 유효 채점 {summary['valid_score_count']}개 · 채점 실패 {summary['failed_score_count']}개 · 전체 채점 평균 {average}/5</p>
<p>FLASK 기준을 적용한 Prometheus 평가입니다. 원래 FLASK GPT 평가와 평가자가 다릅니다.
각 문항에 지정된 스킬만 채점합니다. 실패 점수는 평균에서 제외하며 0점으로 처리하지 않습니다.
도메인 평균은 해당 도메인 문항의 유효 스킬 점수를 모두 합산한 평균입니다. 복수 도메인 문항은 각 도메인에 포함됩니다.</p>
<h2>스킬별 결과</h2>{table(summary['skills'])}
<h2>도메인별 결과</h2>{table(summary['domains'])}
<p>막대의 범위는 0–5이며 유효 점수는 1–5입니다. 점수는 정확도 백분율이 아닙니다.</p></html>'''
    path.write_text(page, encoding='utf-8')
