# Mistral 답변과 Prometheus 평가 파이프라인

```text
별도 학습 데이터 → OpenFedLLM 연합 미세조정 → 라운드별 LoRA 어댑터
                                               ↓ 선택적으로 적용
평가 문항 → Mistral 답변 생성 → answers.jsonl
                                  ↓
          FLASK 스킬 기준 + 참고 답안 + Prometheus 2
                                  ↓
                    reviews.jsonl → CSV / HTML 보고서
```

이번 모듈화 범위는 답변 생성, 평가, 집계입니다. 연합학습은 기존 OpenFedLLM을 사용하며 새 파이프라인에 아직 통합하지 않았습니다. 원본 노트북과 `scripts/legacy`는 이관 근거로 보존합니다.

## 역할과 평가 방식

- Mistral은 답변 생성 모델입니다. `generator.adapter_id`가 null이면 추가 학습 전 모델을 사용합니다. 어댑터 ID와 서브폴더를 설정하면 해당 라운드 어댑터를 적용합니다.
- FLASK는 평가 문항, 스킬별 기준, 1–5점 설명을 제공합니다. 평가 모델이 아닙니다.
- Prometheus는 고정된 평가 모델입니다. 각 문항의 지정된 스킬마다 한 번씩 절대 평가합니다. 문항에 스킬이 3개면 모델 호출도 3번입니다.
- 본 구현은 Prometheus 모델 카드의 입력 요소와 Mistral 대화 형식을 따르되 자체 프롬프트 문구를 사용합니다. `fl_framework/prompts.py`에서 버전을 관리합니다. 원래 FLASK의 GPT 다중 스킬 평가 프롬프트와 다릅니다.
- 참고 답안은 평가자에게만 전달합니다. Mistral의 생성 입력에는 질문만 전달합니다. 평가자는 FLASK 참고 답안을 고득점 기준으로 사용하므로 참고 답안 품질도 결과에 영향을 줍니다.
- FLASK의 기존 GPT 평가 결과와 새 Prometheus 점수를 같은 실험 조건으로 취급하지 마세요. 새 점수의 신뢰도는 추후 일부 문항에 대한 사람/GPT 교차 평가로 확인해야 합니다.

자료: [Prometheus 2 모델 카드](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0), [FLASK](https://github.com/kaistAI/FLASK), [Transformers 대화 템플릿](https://huggingface.co/docs/transformers/v4.46.2/en/chat_templating).

## 모듈 구성

| 모듈 | 책임 |
| --- | --- |
| config.py | 공통 모델 및 실험 설정, 프로젝트 기준 경로 |
| data.py | FLASK 문항/기준 읽기, ID·필수 필드 검증 |
| models.py | Transformers 로딩, PEFT 적용, 프롬프트 형식과 생성 |
| prompts.py | FLASK 기준을 Prometheus 입력으로 변환, 점수 파싱 |
| pipeline.py | 답변 생성과 채점, 결과 파일 및 실행 이력 |
| analysis.py | 스킬/도메인 집계, HTML 막대 그래프 |

## 환경과 실행

Python 3.10–3.12를 사용합니다. 기존 학습용 환경과 분리한 가상환경에 `requirements-inference.txt`를 설치하세요. 이 파일의 버전 조합은 GPU 통합 실행을 아직 검증하지 않았습니다. PyTorch는 사용할 CUDA에 맞는 빌드를 먼저 설치하세요. Hugging Face 인증이 필요한 모델은 `HF_TOKEN` 환경변수 또는 기존 로그인 정보를 사용합니다. 현재 PC 환경과 설치 순서는 [로컬 GPU 안내](local-gpu.md)에 정리했습니다.

프로젝트 루트에서 실행합니다.

```powershell
# 모델 다운로드 없이 데이터/설정/호출 수 확인
python -m fl_framework validate --config configs/mistral_prometheus.json

# GPU 실행: 생성 모델을 해제한 후 평가 모델을 로드
python -m fl_framework all --config configs/mistral_prometheus.json

# 각 단계를 독립 실행할 수도 있음
python -m fl_framework generate --config configs/mistral_prometheus.json
python -m fl_framework score --config configs/mistral_prometheus.json
python -m fl_framework summarize --config configs/mistral_prometheus.json

# GPU 없는 계약 검증
python -m unittest discover -s tests -v
```

`all`과 개별 실행 명령은 대안입니다. `all` 완료 후 같은 run_id로 generate/score를 다시 실행하면 기존 결과를 덮어쓰지 않고 오류를 냅니다. summarize는 검증된 리뷰에서 CSV와 HTML을 재생성할 수 있습니다.

예제는 3문항만 실행하며 확인된 로컬 GPU(RTX 4060 Ti 8GB)에 맞춰 양쪽 모델에 4비트 로딩을 설정했습니다. 실제 메모리 사용은 입력 길이와 실행 환경에 따라 달라지며 8GB 실행 성공을 아직 검증하지 않았습니다. 전체 평가에는 `limit`을 null로 변경하고 새 run_id를 지정하세요. CPU에서 실행하려면 `load_in_4bit`을 false로 바꿔야 하며, 7B 모델은 RAM 사용량이 크고 매우 느릴 수 있습니다. `device`를 cuda로 지정하면 GPU가 없을 때 즉시 오류를 냅니다. 양자화 여부도 비교 조건으로 기록해야 합니다.

`revision`과 `adapter_revision`의 main은 변할 수 있습니다. 재현 실험에는 커밋 SHA를 지정하세요. 베이스 모델의 확인 가능한 resolved revision 및 실제 장치·정밀도를 manifest에 기록합니다. 로컬 모델/어댑터 디렉터리의 파일 내용과 원격 어댑터의 실제 resolved SHA까지 고정·해시하는 기능은 아직 없습니다.

## 연합학습 어댑터 평가

예제 설정을 복사해 새 run_id를 주고 generator의 다음 값을 바꿉니다.

```json
{
  "adapter_id": "본인의/어댑터저장소",
  "adapter_revision": "main",
  "adapter_subfolder": "checkpoint-10",
  "prompt_format": "alpaca"
}
```

`model_id`는 어댑터의 base_model_name_or_path와 일치해야 합니다. 토크나이저는 기본적으로 베이스 모델에서 로드합니다. 학습 중 토크나이저를 변경했다면 `tokenizer_id`와 `tokenizer_revision`으로 해당 버전을 지정하세요. 로컬 경로는 프로젝트 루트를 기준으로 해석합니다.

기존 학습 스크립트는 alpaca 템플릿을 사용합니다. 과거 어댑터를 평가할 때는 학습 당시 형식을 확인해야 합니다. chat과 alpaca 형식의 점수를 직접 비교하면 프롬프트 효과가 섞일 수 있으므로, 베이스/중앙/연합 모델 비교에는 동일한 생성 입력 형식을 적용하세요. judge 설정은 동일하게 유지합니다.

## 결과와 재실행

`runs/<run_id>/`에 다음을 저장합니다.

- `manifest.json`: 설정, 데이터·기준·파이프라인 코드 해시, 패키지 버전, 단계별 모델 메타데이터와 결과 해시
- `answers.jsonl`: 질문 ID와 생성 답변
- `reviews.jsonl`: 질문·스킬별 점수, 채점 원문, 파싱 실패 이유
- `skills.csv`, `domains.csv`, `summary.json`, `report.html`: 평균, 유효 채점 수, 실패 수, 시각화

문항은 ID로 연결합니다. 점수는 마지막 `[RESULT] 1`부터 `[RESULT] 5` 형식만 인정합니다. 잘못된 응답은 null과 오류로 보존하고 평균에서 제외합니다. 전체 평균은 모든 유효 문항-스킬 점수의 평균(micro)이며, 별도로 스킬 평균들의 평균(macro)을 저장합니다. 도메인 평균은 해당 도메인의 유효 스킬 점수 평균입니다. 모든 점수가 실패하면 N/A로 표시합니다.

생성/채점 도중 메모리 오류나 토큰 한도 초과가 발생하면 해당 단계는 중단됩니다. `.partial` 파일이 남고 완료 결과로 취급하지 않습니다. 자동 문항별 재개·실패 점수 재채점은 아직 구현하지 않았습니다. 원인을 해결한 뒤 새 run_id로 다시 실행하거나, 동일 설정에서 해당 partial 파일을 검토·삭제한 후 그 단계를 처음부터 다시 실행하세요. 완료된 generate 결과가 있으면 score만 실행해 Mistral 생성을 반복하지 않을 수 있습니다.

학습·평가 데이터 중복 검사는 이번 범위에 포함하지 않았습니다. FLASK 평가 문항을 학습에 사용한 과거 어댑터의 점수는 독립적인 일반화 성능 증거로 해석할 수 없습니다.
