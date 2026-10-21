"""Shared configuration; importing this module never loads a model."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str = 'main'
    adapter_id: str | None = None
    adapter_revision: str = 'main'
    adapter_subfolder: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    prompt_format: str = 'chat'
    device: str = 'auto'
    dtype: str = 'auto'
    load_in_4bit: bool = False
    max_input_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.0

    def __post_init__(self):
        if not self.model_id.strip():
            raise ValueError('model_id must not be empty')
        if self.prompt_format not in {'chat', 'alpaca', 'prometheus'}:
            raise ValueError('prompt_format must be chat, alpaca, or prometheus')
        if self.device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('device must be auto, cpu, or cuda')
        if self.dtype not in {'auto', 'float32', 'float16', 'bfloat16'}:
            raise ValueError('Unsupported dtype')
        if min(self.max_input_tokens, self.max_new_tokens) < 1 or self.temperature < 0:
            raise ValueError('Token limits must be positive and temperature nonnegative')
        if self.adapter_subfolder and not self.adapter_id:
            raise ValueError('adapter_subfolder requires adapter_id')


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    project_root: Path
    questions: Path
    rubrics: Path
    output_root: Path
    generator: ModelConfig
    judge: ModelConfig
    limit: int | None = None
    seed: int = 42

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', self.run_id):
            raise ValueError('run_id must be a simple directory name')
        if self.limit is not None and self.limit < 1:
            raise ValueError('limit must be positive or null')
        if self.judge.prompt_format != 'prometheus':
            raise ValueError('The judge must use the prometheus prompt format')
        if self.judge.adapter_id:
            raise ValueError('Keep the judge fixed: judge adapters are not supported')

    @property
    def run_dir(self):
        return self.output_root / self.run_id

    def as_dict(self):
        result = asdict(self)
        for key in ('project_root', 'questions', 'rubrics', 'output_root'):
            result[key] = str(result[key])
        return result


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    root = (path.parent / raw.pop('project_root', '..')).resolve()
    def resolve(value):
        return (root / value).resolve()
    return ExperimentConfig(
        project_root=root,
        questions=resolve(raw.pop('questions')),
        rubrics=resolve(raw.pop('rubrics')),
        output_root=resolve(raw.pop('output_root', 'runs')),
        generator=ModelConfig(**raw.pop('generator')),
        judge=ModelConfig(**raw.pop('judge')),
        **raw,
    )
