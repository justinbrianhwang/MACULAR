# 추가 실험 지시서 (공저자용)

리뷰어 대응용 추가 실험 5종. 코드·설정은 전부 커밋돼 있고, 아래 명령을
**순서대로** 실행하면 됩니다. 결과 JSON은 `results/`에 자동 저장됩니다
(`results/`는 gitignore — 끝나면 JSON 파일들을 zip이 아닌 개별 파일로 전달).

## 0. 환경

```bash
git pull
conda activate macular          # peft 0.20, transformers 5.14, torch 2.x
```

GPU 메모리 32GB 기준. 각 실행은 백그라운드로 걸어도 되고, 순차 실행 총
약 14시간입니다.

## 1. 스모크 테스트 (필수, ~3분)

새 플래그 2개(deterministic, eval_matched_half)가 이 환경에서 도는지 먼저
확인:

```bash
python -m pytest tests/test_ocr_adapt.py -q          # 6 passed 확인
python -c "import torch; torch.use_deterministic_algorithms(True, warn_only=True); print('det ok')"
```

## 2. 실험 (우선순위 순)

### 2-1. 결정론 통제 (~5h) — 가장 중요
동일 seed 발산(0.099 vs 0.630)의 원인 규명. seed 0을 **두 번** 돌립니다.

```bash
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_det.yaml --out results
```

판독: `results/ocr_adapt_xfund_det.json`의 per_seed에서 첫 두 항목(둘 다
seed 0)의 `cer_after`가
- **완전 동일** → 커널 비결정성이 원인으로 확정 (논문 §4.3 hypothesis → confirmed로 승격 가능)
- **여전히 다름** → 우리가 통제 못한 상태가 원인 (논문 문구 유지)

주의: `warn_only=True`라 일부 op는 여전히 비결정적일 수 있음. 실행 로그에
deterministic 경고가 뜨면 어떤 op인지 캡처해서 같이 전달.

### 2-2. 전이 비교의 평가셋 통일 (~2.5h)
syn→real 전이를 in-domain과 **같은 평가 절반**에서 재측정 (리뷰 지적:
기존 전이 수치는 다른 평가셋이라 크기 비교 불가).

```bash
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_syn2real_matched.yaml --out results
```

판독: baseline이 in-domain 표(ja 0.846, es 0.924)와 ±0.005 안에서 일치해야
정상. 그 후 after CER를 in-domain 풀과 직접 비교.

### 2-3. rsLoRA r8 표본 보강 (~2.5h)
3 seed → 7 run으로 늘려 LoRA r16 풀과 n-매칭.

```bash
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_rslora_r8_more.yaml --out results
```

판독: 기존 `results/ocr_adapt_xfund_rslora_r8.json`(seeds 0,1,2)과 합쳐
7 run. 발산(예: ja CER > 0.3) 등장 여부가 핵심 — 7 run에서도 없으면
권고가 강해지고, 나오면 rsLoRA도 같은 꼬리를 가진다고 논문 수정.

### 2-4. 변형별 lr 공정성 (~5h, 2개)
"공유 lr이 변형에 불리했나"에 대한 답.

```bash
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_pissa_r8_lr3e5.yaml --out results
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_rslora_r16_lr3e5.yaml --out results
```

판독:
- PiSSA가 lr 3e-5에서 LoRA r8 수준으로 회복하면 → "PiSSA 열세는 lr 문제"로
  논문 수정. 여전히 나쁘면 현행 유지.
- rsLoRA r16이 lr 3e-5에서 정상화되면 → "스케일×lr 곱이 원인"으로 더
  정밀한 결론. 여전히 붕괴하면 현행 유지.

## 3. 결과 전달

각 JSON에서 언어별 `cer_after`/`exact_match`(per seed)만 있으면 되고,
파일 전체를 주는 게 가장 확실합니다. 판독 요약은 제가(Claude) 하니
숫자 해석이 애매하면 그대로 전달만 해주세요.

## 참고: 실행이 죽거나 이상할 때
- OOM: 다른 GPU 프로세스 확인 (`nvidia-smi`), 배치는 1이라 OOM이면 대부분
  다른 프로세스 탓.
- 중간에 끊긴 실행은 그냥 처음부터 재실행 (checkpoint 없음, 실행당 ~1.5h).
- 결과 파일은 실행이 끝나야 써짐 — 파일이 없으면 아직 안 끝난 것.
