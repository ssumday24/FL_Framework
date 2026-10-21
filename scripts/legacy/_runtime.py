"""Small compatibility helpers for the first notebook migration."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(os.environ.get('FL_FRAMEWORK_ROOT', Path(__file__).resolve().parents[2])).resolve()


def run_command(command, namespace):
    # Tokenize before interpolation so paths containing spaces stay one argument.
    arguments = shlex.split(command)
    def expand(match):
        name = match.group(1)
        if name not in namespace:
            raise ValueError(f'Unknown notebook command variable: {name}')
        return str(namespace[name])
    arguments = [re.sub(r'\$(\w+)', expand, part) for part in arguments]
    if arguments[0] == 'python':
        arguments[0] = sys.executable
    subprocess.run(arguments, check=True)
