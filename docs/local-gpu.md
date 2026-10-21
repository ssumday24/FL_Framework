# RTX 4060 Ti 8GB에서 로컬 평가 실행

현재 PC에서 RTX 4060 Ti, VRAM 8188MiB를 확인했습니다. 7B 모델을 FP16으로 전부 GPU에 올리는 것은 어렵지만, 4비트 양자화와 모델별 순차 실행이면 추론·평가를 시도할 수 있습니다. 실제 성공 여부와 속도는 모델을 로드한 뒤 확인해야 합니다. 연합 미세조정 학습 가능 여부는 별도 검증 대상입니다.

## 적용한 설정

- Mistral 답변 생성 후 GPU 메모리를 해제하고 Prometheus를 로드합니다.
- 두 모델 모두 CUDA, NF4 4비트, 이중 양자화, FP16 계산을 사용합니다.
- SDPA attention을 사용하며 배치는 문항 1개입니다.
- 생성 입력 한도 2048토큰, 답변 한도 256토큰으로 시작합니다.
- 평가 입력 한도 4096토큰, 평가 출력 한도 512토큰으로 시작합니다.
- 입력이 한도를 넘으면 잘라서 채점하지 않고 오류를 냅니다. 긴 답변이 필요한 문항은 별도 실험에서 토큰 한도를 조정해야 합니다.
- CPU/디스크에 일부 모델이 자동 배치되어 느려지는 것을 피하려고 전체 모델을 첫 CUDA GPU에 명시적으로 배치합니다. 메모리가 모자라면 OOM이 발생합니다.

7B 가중치를 4비트로만 계산하면 약 3.5GB(십진 단위)지만, 실제로는 비양자화 파라미터, 양자화 정보, KV 캐시와 연산 메모리가 추가됩니다. 따라서 3.5GB만 확보하면 된다는 뜻은 아닙니다. 다른 GPU 프로그램을 종료한 상태에서 3문항부터 검증하세요.

새 파이프라인은 OpenAI API를 호출하지 않으며 `openai_info`를 읽지 않습니다. 최초 Hugging Face 모델 다운로드에는 인터넷과 저장 공간이 필요합니다. 모델 접근 권한이 필요하면 HF_TOKEN을 사용합니다. 캐시 다운로드 완료 후 실제 모델 생성·채점은 이 PC에서 수행합니다. 기존 GPT 평가 스크립트는 비교용으로 남겨두었습니다.

## Python 환경

현재 기본 Python은 3.13입니다. 기존에 고정한 추론 의존성은 Python 3.10–3.12 환경을 사용해야 합니다. Python 3.12 설치 후 프로젝트 전용 가상환경을 만들고 CUDA용 PyTorch부터 설치합니다. 아래 명령은 Python 3.12 설치가 선행 조건입니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt

# 모델 다운로드 없이 환경 확인
.\.venv\Scripts\python.exe -m fl_framework doctor
.\.venv\Scripts\python.exe -m fl_framework validate --config configs/mistral_prometheus.json

# 실제 모델 다운로드 및 3문항 생성·평가
.\.venv\Scripts\python.exe -m fl_framework all --config configs/mistral_prometheus.json
```

환경 진단은 패키지·CUDA 상태를 확인합니다. 정상 진단이 모델 실행 성공까지 보장하지는 않습니다. 이 작업에서는 라이브러리 설치, 모델 다운로드 및 실제 7B 모델 실행을 수행하지 않았습니다.

참고: [PyTorch CUDA 빌드 설치](https://pytorch.org/get-started/previous-versions/), [bitsandbytes 설치](https://huggingface.co/docs/bitsandbytes/installation).
