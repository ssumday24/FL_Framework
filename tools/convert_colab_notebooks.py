"""Export the original notebooks without executing their cells."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'FLASK 추론 단계': 'inference',
    'FLASK 추론용 데이터셋을 허깅페이스에 업로드': 'upload_dataset',
    'FLASK 평가 결과 CSV 출력': 'analyze_results',
    'FLASK 평가 단계': 'evaluate',
    'OpenFedLLM 학습 완료된 어댑터 모델 한꺼번에 업로드': 'upload_adapters',
    'OpenFedLLM을 이용한 학습 과정': 'train',
    'PEFT 어댑터 모델 추론 시연 및 테스트': 'demo_adapter',
}
PREFIXES = ('/content/drive/MyDrive/OpenFedFLASK', '/content/drive/MyDrive/Colab Models')


class Paths(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith(PREFIXES):
            prefix = next(p for p in PREFIXES if node.value.startswith(p))
            return ast.copy_location(ast.BinOp(
                left=ast.Call(func=ast.Attribute(value=ast.Name(id='PROJECT_ROOT', ctx=ast.Load()), attr='as_posix', ctx=ast.Load()), args=[], keywords=[]),
                op=ast.Add(), right=ast.Constant(node.value[len(prefix):])), node)
        return node

    def visit_JoinedStr(self, node):
        values = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith(PREFIXES):
                prefix = next(p for p in PREFIXES if value.value.startswith(p))
                values.extend([ast.FormattedValue(value=ast.Call(func=ast.Attribute(value=ast.Name(id='PROJECT_ROOT', ctx=ast.Load()), attr='as_posix', ctx=ast.Load()), args=[], keywords=[]), conversion=-1), ast.Constant(value.value[len(prefix):])])
            else:
                values.append(self.visit(value))
        node.values = values
        return node


def convert():
    output = ROOT / 'scripts' / 'legacy'
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / 'ColabNotebooks').glob('*.ipynb')):
        notebook = json.loads(source.read_text(encoding='utf-8'))
        cells = []
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            lines = []
            for line in ''.join(cell['source']).splitlines():
                stripped = line.strip()
                if stripped.startswith(('from google.colab import', 'drive.mount(')):
                    lines.append('# Colab setup removed: use local paths and environment variables.')
                elif stripped.startswith('!pip '):
                    lines.append('# Environment setup (install separately): ' + stripped[1:])
                elif stripped.startswith('!huggingface-cli login'):
                    lines.append('# Hugging Face clients use HF_TOKEN from the environment.')
                elif stripped.startswith(('cd ', '%cd ')):
                    path = stripped.split(' ', 1)[1]
                    lines.append(f'os.chdir({path!r})')
                elif stripped.startswith('!'):
                    lines.append(f'run_command({stripped[1:]!r}, locals())')
                else:
                    line = re.sub(r"userdata.get\(['\"]huggingface_(?:read|write)['\"]\)", 'os.environ["HF_TOKEN"]', line)
                    lines.append(line)
            tree = Paths().visit(ast.parse('\n'.join(lines)))
            ast.fix_missing_locations(tree)
            cells.append(f'    # Notebook cell {index}\n' + '\n'.join('    ' + line if line else '' for line in ast.unparse(tree).splitlines()))
        header = f'''"""Converted from ColabNotebooks/{source.name}.

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
'''
        text = header + '\n\n'.join(cells) + '\n\n\nif __name__ == "__main__":\n    main()\n'
        ast.parse(text)
        (output / (NAMES[source.stem] + '.py')).write_text(text, encoding='utf-8')
        print(f'{source.name} -> {NAMES[source.stem]}.py')


if __name__ == '__main__':
    convert()
