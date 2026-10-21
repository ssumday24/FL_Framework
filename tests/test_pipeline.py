"""Contract tests run without downloading models or requiring CUDA."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from fl_framework.analysis import summarize
from fl_framework.config import ExperimentConfig, ModelConfig
from fl_framework.data import load_questions
from fl_framework.pipeline import generate_answers, prepare, score_answers
from fl_framework.prompts import build_judge_prompt, parse_score


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = self.root / 'questions.jsonl'
        self.questions.write_text('\n'.join(json.dumps(q) for q in [
            {'idx': 2, 'instruction': 'Question B', 'answer': 'Reference B',
             'skill': ['Readability', 'Factuality'], 'domain': ['Language']},
            {'idx': 1, 'instruction': 'Question A', 'answer': 'Reference A',
             'skill': ['Factuality'], 'domain': ['Technology', 'Language']},
        ]), encoding='utf-8')
        self.rubrics = self.root / 'rubrics.json'
        self.rubrics.write_text(json.dumps([
            {'Skill': skill, 'Criteria': f'Only assess {skill}',
             'Scoring': {str(n): f'Level {n}' for n in range(1, 6)}}
            for skill in ['Factuality', 'Readability', 'Completeness']
        ]), encoding='utf-8')
        self.config = ExperimentConfig('test', self.root, self.questions, self.rubrics,
                                       self.root / 'runs', ModelConfig('generator'),
                                       ModelConfig('judge', prompt_format='prometheus'))
        self.events = []
        events = self.events
        class FakeModel:
            def __init__(self, config, seed, project_root):
                self.config = config
                self.metadata = {'model_id': config.model_id, 'test_backend': True}
                events.append(('load', config.model_id))
            def generate(self, text):
                events.append(('prompt', self.config.model_id, text))
                if self.config.model_id == 'generator':
                    return f'Candidate for {text}'
                if '[Readability:' in text:
                    return 'Good writing. [RESULT] 5'
                if 'Question A' in text:
                    return 'Unparseable grading response'
                return 'Mostly accurate. [RESULT] 3'
            def close(self):
                events.append(('close', self.config.model_id))
        self.factory = FakeModel

    def run_models(self):
        generate_answers(self.config, self.factory)
        score_answers(self.config, self.factory)

    def test_end_to_end_contract_and_failed_scores(self):
        self.run_models()
        result = summarize(self.config)
        self.assertEqual(result['valid_score_count'], 2)
        self.assertEqual(result['failed_score_count'], 1)
        self.assertEqual(result['micro_avg'], 4)
        self.assertEqual([r['name'] for r in result['skills']], ['Factuality', 'Readability', 'Completeness'])
        self.assertEqual(result['skills'][0], {'name': 'Factuality', 'count': 1, 'failed': 1, 'avg': 3})
        self.assertIsNone(result['skills'][2]['avg'])
        self.assertEqual(result['domains'][1]['failed'], 1)
        report = (self.config.run_dir / 'report.html').read_text(encoding='utf-8')
        self.assertIn('N/A', report)
        self.assertIn('Factuality', report)
        self.assertTrue((self.config.run_dir / 'skills.csv').exists())
        self.assertLess(self.events.index(('close', 'generator')), self.events.index(('load', 'judge')))
        generation_prompts = [e[2] for e in self.events if e[:2] == ('prompt', 'generator')]
        self.assertEqual(generation_prompts, ['Question B', 'Question A'])
        judge_prompts = [e[2] for e in self.events if e[:2] == ('prompt', 'judge')]
        self.assertEqual(len(judge_prompts), 3)
        self.assertIn('Reference B', judge_prompts[0])
        self.assertIn('Candidate for Question B', judge_prompts[0])
        self.assertIn('Score 5: Level 5', judge_prompts[0])

    def test_existing_answers_not_overwritten(self):
        generate_answers(self.config, self.factory)
        with self.assertRaises(FileExistsError):
            generate_answers(self.config, self.factory)

    def test_changed_answer_file_rejected_before_judge_load(self):
        generate_answers(self.config, self.factory)
        with (self.config.run_dir / 'answers.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            score_answers(self.config, self.factory)
        self.assertNotIn(('load', 'judge'), self.events)

    def test_changed_config_rejected(self):
        generate_answers(self.config, self.factory)
        with self.assertRaisesRegex(ValueError, 'changed'):
            score_answers(replace(self.config, limit=1), self.factory)

    def test_duplicate_question_ids_rejected_even_beyond_limit(self):
        text = self.questions.read_text(encoding='utf-8')
        self.questions.write_text(text + '\n' + text.splitlines()[0], encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_questions(self.questions, limit=1)

    def test_unknown_rubric_rejected(self):
        self.questions.write_text(self.questions.read_text(encoding='utf-8').replace('Factuality', 'Unknown'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unknown FLASK skills'):
            prepare(self.config)

    def test_strict_score_parsing(self):
        self.assertEqual(parse_score('Criteria mention 1 and 5. [RESULT] 4\n'), 4)
        for value in ('Score 4', '[RESULT] 0', '[RESULT] 6', '[RESULT] 4.5',
                      '[RESULT] 4 extra', '[RESULT] 2 [RESULT] 4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_score(value)

    def test_failed_model_closes_and_preserves_partial_output(self):
        events = self.events
        class Broken(self.factory):
            def generate(self, text):
                raise RuntimeError('GPU out of memory')
        with self.assertRaisesRegex(RuntimeError, 'GPU out of memory'):
            generate_answers(self.config, Broken)
        self.assertIn(('close', 'generator'), events)
        self.assertFalse((self.config.run_dir / 'answers.jsonl').exists())
        self.assertTrue((self.config.run_dir / 'answers.jsonl.partial').exists())


if __name__ == '__main__':
    unittest.main()
