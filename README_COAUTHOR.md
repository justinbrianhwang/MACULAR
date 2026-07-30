# MACULAR — 실험 하네스

> 이 문서는 원래 공저자 핸드오프용으로 작성됐고, 현재는 단독 작업 기준입니다.
> (zip 패키징은 더 이상 사용하지 않고 저장소 파일로만 관리합니다.)

이 패키지는 **바로 돌아가는 실험 하네스**입니다. GPU 없이 CPU만으로
합성 데이터 생성 → shortcut 감사 → OCR baseline → 지표까지 end-to-end로
돌아가 **결과 JSON을 뱉습니다.** MACULAR 모델 코어(GPU 필요)는 같은 러너에
꽂히는 스캐폴드로 들어 있습니다.

> **역할 분담 전제:** 공저자 컴퓨터가 더 강력하므로 **GPU 실험(모델 학습)은
> 공저자 쪽에서** 돌리는 것을 가정합니다. 지금 당장 결과가 나오는 CPU 실험을
> 먼저 여러 조건으로 돌려 보내 주시면, 그 결과로 다음 단계를 정합니다.

---

## 0. 30초 요약

```bash
conda env create -f environment.yml
conda activate macular
pip install -e .

macular probe                                   # 환경 리포트 (GPU/tesseract 감지)
macular run data_gen        --config configs/coauthor.yaml
macular run shortcut_audit  --config configs/coauthor.yaml
macular run data_stats      --config configs/coauthor.yaml
```

결과는 `results/*.json`에 쌓입니다. **이 폴더를 zip으로 보내 주시면 됩니다.**

---

## 1. 설치 (conda)

```bash
# 저장소 압축을 푼 폴더에서
conda env create -f environment.yml
conda activate macular
pip install -e .
```

설치 확인:

```bash
macular probe
pytest -q          # 8개 테스트가 통과해야 정상
```

`macular probe`가 GPU, tesseract 설치 여부를 알려주고 `results/env_report.json`에
저장합니다. **이 파일도 함께 보내 주세요** — 어떤 하드웨어에서 돈 결과인지
기록으로 필요합니다.

---

## 2. 지금 바로 돌아가는 실험 (CPU, 결과를 보내주실 것)

| 실험 | 명령 | 산출물 | 의미 |
|---|---|---|---|
| 합성 데이터 생성 | `macular run data_gen --config configs/coauthor.yaml` | `data/meddoc/{train,val,test}.jsonl` + `images/` | 3개 언어 · PII 정답·bbox·FHIR path 포함. train=A, val=B, test=C **분리 생성기** |
| shortcut 감사 | `macular run shortcut_audit --config configs/coauthor.yaml` | `results/shortcut_audit.json` | **좌표만으로 PII를 맞히는지** 검사 (제안서 §14.4, §25 게이트 5) |
| 데이터 통계 | `macular run data_stats --config configs/coauthor.yaml` | `results/data_stats.json` | 문서 수·PII 비율·언어 분포 |
| OCR baseline | `macular run ocr_baseline --config configs/coauthor.yaml` | `results/ocr_baseline.json` | 엔진 CER/WER (tesseract/easyocr. 없으면 자동 skip) |
| **실제 스캔 문서(FUNSD)** | `macular run fetch_funsd --config configs/funsd.yaml` | `data/funsd/` | 공개 실제 영어 스캔 폼 다운로드+변환 (§10.1) |
| **실제 스캔 문서(XFUND)** | `macular run fetch_xfund --config configs/xfund.yaml` | `data/xfund/` | 공개 실제 일본어·스페인어 스캔 폼 |

> **합성 데이터가 현실화되었습니다(#1).** 이제 실제 의료 용어(LOINC 검사·RxNorm
> 약물)와 현실적 값 분포를 쓰고, **검사결과지 + 처방전** 두 문서 유형을 섞어
> 생성합니다. FHIR path에 실제 LOINC/RxNorm 코드가 들어갑니다.

### 꼭 해주셨으면 하는 것: shortcut 감사 A/B 비교

이게 연구의 첫 판정 게이트입니다. **두 설정으로 각각 돌려서 F1을 비교**해 주세요.

```bash
# 기본 레이아웃 (PII 위치가 고정) — F1이 높게 나옵니다 = shortcut 있음
macular run data_gen       --config configs/coauthor.yaml
macular run shortcut_audit --config configs/coauthor.yaml

# counterfactual 레이아웃 (PII 위치를 섞음) — F1이 떨어져야 정상 = 방어 작동
macular run data_gen       --config configs/counterfactual.yaml --out results/cf
macular run shortcut_audit --config configs/counterfactual.yaml --out results/cf
```

참고로 이쪽(개발 환경)에서 나온 값은 다음과 같습니다. (반사실 레이아웃을
강화해서, PII·decoy·임상값을 페이지 전체에 뒤섞고 block_type도 무작위화했습니다
— 이제 좌표만으로는 PII를 base rate 이상으로 못 맞힙니다.)

| 설정 | 좌표-only F1 | precision | shortcut_detected |
|---|---:|---:|---|
| 기본 레이아웃 | ~1.00 | ~1.00 | **true** (위치가 PII를 그대로 알려줌) |
| counterfactual | ~0.43 | ~0.28 (≈base rate) | **false** (위치 신호 소멸) |

**두 결과 JSON을 모두 보내 주세요.** 공저자 컴퓨터에서도 비슷하게 나오는지,
그리고 `n_per_split`을 크게 키웠을 때도 유지되는지가 관심사입니다. 만약
counterfactual에서도 F1이 높게(>0.75) 유지되면 생성기에 다른 누출이 있다는
뜻이라 제가 생성기를 고칩니다.

#### 결과 읽는 법 (중요)

- **shortcut 감사와 OCR baseline은 완전히 독립된 실험입니다.** 좌표-only 감사는
  일부러 텍스트도 픽셀도 쓰지 않고 **박스 좌표 + block type만** 봅니다. 그래서
  OCR이 잘 되든 안 되든 감사 결과는 바뀌지 않습니다. "OCR이 나빠서 위치로
  맞춘다"는 인과는 성립하지 않습니다.
- 위치로 PII를 맞히는 건 **버그가 아니라 감사가 데이터 누출을 적발한 것**입니다.
- counterfactual에서 `f1`은 떨어지는데 `recall`은 1.0으로 남을 수 있습니다.
  이건 모든 PII가 아직 `value` block type을 공유하기 때문입니다(위치는 섞였지만
  block type이 남은 신호). precision이 떨어지면서 F1이 내려간 것이고, 이게
  제가 데이터에서 다음으로 손볼 지점입니다.
- **이 감사는 합성 데이터셋에 대한 진단이지 MACULAR 모델 판정이 아닙니다.**
  모델 코어(`macular/models`)는 아직 미구현이라 여기서 돌아가지 않습니다.

### OCR baseline을 돌리려면 (선택)

**중요 — OCR baseline은 region 단위·언어별로 측정합니다.** 페이지 전체를 한
언어로 읽지 않습니다. 각 문서를 그 문서의 언어팩으로(en→eng, ko→kor, ja→jpn)
후보 영역별로 OCR해서 CER/WER를 **언어별로 따로** 보고합니다. 언어들을 하나의
평균으로 섞지 않습니다.

**OCR 엔진은 교체할 수 있습니다.** Tesseract는 CJK에서 사실상 못 씁니다
(그게 제안서가 PaddleOCR-VL을 쓰는 이유). `configs/coauthor.yaml`의
`ocr_engine`으로 고릅니다.

| 엔진 (`ocr_engine`) | 설치 | CJK | 비고 |
|---|---|---|---|
| `tesseract` | `pip install -e ".[ocr]"` + 바이너리·언어팩 | 약함(하한선) | 영어만 쓸만함 |
| `easyocr` | `pip install -e ".[easyocr]"` | 좋음 | GPU 자동, 최초 실행 시 모델 다운로드 |
| **`paddleocr`** | `pip install -e ".[paddleocr]"` + paddlepaddle | **매우 좋음** | **PP-OCR. CJK 강함. GPU. 권장** |
| `paddleocr_vl` | `.[model]` + 모델 다운로드 | 최상(제안서 목표) | **실험적**, 설치 무거움, 미검증 |

```bash
# 권장: 같은 세트에서 여러 엔진을 비교해 주세요.
pip install -e ".[easyocr]"
pip install -e ".[paddleocr]"
# PP-OCR은 CUDA에 맞는 paddlepaddle이 필요: https://www.paddlepaddle.org.cn/install
#   예) python -m pip install paddlepaddle-gpu
# tesseract도 보려면:
conda install -c conda-forge tesseract   # + kor/jpn 언어팩
pip install -e ".[ocr]"

macular probe    # 결과의 ocr_engines 딕셔너리에서 어떤 엔진이 available 인지 확인

# 엔진은 config의 ocr_engine 으로 고릅니다. paddle 전용 config도 있습니다:
macular run ocr_baseline --config configs/paddle.yaml     # ocr_engine: paddleocr
macular run ocr_baseline --config configs/coauthor.yaml   # ocr_engine 값에 따름
```

> **CJK를 제대로 재려면 `paddleocr`(PP-OCR)을 쓰세요.** Tesseract는 하한선일
> 뿐입니다. `paddleocr_vl`(PaddleOCR-VL-1.6)은 제안서의 최종 목표 엔진이라
> 배선은 해뒀지만 설치가 무겁고 제가 검증하지 못했습니다 — 안 되면 자동
> skip되며, 되게 하려면 모델 카드대로 세팅이 필요합니다. **엔진별 결과를 모두
> 보내 주시면** 실제 스캔(FUNSD/XFUND)과 합성 양쪽에서 CJK 차이를 비교합니다.

`macular probe`로 `installed_languages`(tesseract 언어팩)를 확인하세요. 엔진이
처리 못 하는 언어는 `skipped_languages`로 기록되고 잘못된 엔진으로 읽지
않습니다. 결과 JSON의 `engine` 필드로 어떤 엔진 결과인지 구분됩니다.
**tesseract와 easyocr 결과를 둘 다 보내 주시면** CJK에서 얼마나 차이 나는지
바로 비교할 수 있습니다.

결과 JSON은 `per_language`(언어별 cer/wer), `macro`(언어 평균),
`skipped_languages`, `installed_languages`를 담습니다. tesseract가 없으면
실험은 깨지지 않고 `skipped`로 남습니다.

**CJK의 WER은 null로 나옵니다.** 한국어·일본어는 단어 사이 공백이 없어서
공백 기반 WER이 의미가 없습니다(그래서 예전에 163% 같은 값이 나왔던 것).
CJK는 **CER만** 보세요.

#### 폰트 경고를 꼭 확인하세요 (한글·일본어 CER이 높게 나왔다면)

한글/일본어 CER이 비정상적으로 높으면(예: 80%) 거의 항상 **렌더링 폰트 문제**
입니다. 한 폰트가 모든 문자를 담지 못해서(예: 일본어 폰트는 한글이 없음)
한글이 tofu(□□□)로 렌더되면 Tesseract가 읽을 게 없습니다.

이제 언어별로 폰트를 고르고 **글자가 실제로 렌더되는지 런타임에 검사**합니다.
커버하는 폰트가 없으면 데이터 생성 결과(`results/data_gen.json`)에
`font_warnings`로 표시됩니다. **거기에 `ko`나 `ja`가 있으면** CJK 폰트를 까세요:

```bash
# Linux
conda install -c conda-forge fonts-noto-cjk    # 또는: sudo apt install fonts-noto-cjk
# Windows: 보통 malgun.ttf(한글)·YuGothic(일본어)이 기본 설치되어 있음
```

폰트를 깐 뒤 **data_gen부터 다시** 돌리고(이미지 재렌더) ocr_baseline을 재실행
하세요. `font_warnings`가 비어 있고 나서의 CER만 신뢰하세요.

> 이 baseline은 약한 외부 OCR(Tesseract)의 **하한선**입니다. 폰트를 고쳐도
> Tesseract는 노이즈 있는 CJK 의료 스캔에서 원래 약합니다 — CER이 좀 높게
> 남아도 정상이며, 이게 바로 실제 시스템이 Tesseract가 아니라 PaddleOCR-VL /
> VLM 백본을 쓰는 이유입니다(제안서의 핵심 동기). shortcut 감사·통계는 이
> 문제와 무관합니다.

### 규모를 키우려면

`configs/coauthor.yaml`의 `n_per_split`을 올리면 됩니다(기본 500 → split당 500,
총 1500문서). 공저자 컴퓨터가 더 강하니 수천 단위로 올리셔도 됩니다.
이미지 렌더링이 느리면 `render_images: false`로 두면 라벨만 빠르게 생성됩니다
(shortcut 감사·통계는 이미지가 필요 없습니다).

---

## 3. 실제 스캔 문서로 OCR 비교 (#2, 핵심 요청)

제 합성 렌더는 폰트 아티팩트가 있었습니다. **진짜 스캔 이미지**에서 OCR을
공정 비교하려면 공개 데이터셋 FUNSD(영어)·XFUND(일본어·스페인어)를 씁니다.
사람이 단 GT 텍스트 + bbox가 있어서 CER/WER를 바로 계산할 수 있습니다.

```bash
# 영어 실제 스캔 폼
macular run fetch_funsd  --config configs/funsd.yaml    # 다운로드+변환 (인터넷 필요)
macular run ocr_baseline --config configs/funsd.yaml    # easyocr 기본
macular run data_stats   --config configs/funsd.yaml

# 일본어·스페인어 실제 스캔 폼
macular run fetch_xfund  --config configs/xfund.yaml
macular run ocr_baseline --config configs/xfund.yaml
```

엔진 비교를 위해 각 config의 `ocr_engine`을 `tesseract`↔`easyocr`로 바꿔가며
돌려서 **결과를 둘 다 보내 주세요.** 실제 스캔 문서에서 두 엔진이 언어별로
얼마나 차이 나는지가 이번 핵심 관심사입니다.

> **데이터 거버넌스:** FUNSD·XFUND는 **공개 비-PHI 연구 데이터셋**입니다(연구용
> 라이선스, 각 `manifest`에 출처 기록). **실제 환자 의료 문서는 이 패키지 범위가
> 아니며**, IRB/DUA/기관 내부 절차로만 다룹니다(제안서 §14.6). 실제 PHI 문서를
> 개인 PC나 외부로 옮기지 마세요.

---

## 3. 모델 코어 (구현됨, CPU에서 학습 검증)

MACULAR의 **미분 가능한 핵심**(백본 무관 부분)은 이제 real torch 모듈로
구현됐고 CPU에서 실제로 학습됩니다:

- CDR projector, **differentiable redaction gate**(11.5), relation graph(11.7),
  EMA raw teacher + consistency(11.6), 결합 손실(11.12)
- 핵심 성질 검증: **임상 손실이 gate를 통해 PII head로 역전파**됩니다
  (`tests/test_model_core.py`), 합성 신호에서 loss가 4.7→0.4로 하강

```bash
pip install -e ".[model]"     # torch
pytest -q tests/test_model_core.py     # 6개 통과
python -c "from macular.models import fit_synthetic; m,h=fit_synthetic(); print(h[0],'->',h[-1])"
```

### OCR → 모델 코어 연결: 실제 데이터로 학습 (`train_core`)

OCR 도구가 뽑은 **region 텍스트를 피처로** 모델 코어를 **우리 실제 데이터**로
학습합니다(백본 무관 CPU 버전). `feature_source`로 텍스트 출처를 고릅니다:

- `gt` (기본): 주석/정답 텍스트 (상한선)
- `ocr`: 엔진으로 crop을 재인식 → **OCR 오류가 피처로 전파**(현실적)

```bash
macular run data_gen   --config configs/coauthor.yaml
macular run train_core --config configs/coauthor.yaml     # results/train_core.json
```

산출물: `loss_start/end`, `train_pii_f1`, **`val_pii_f1`**(val=family C = PII-value
-held-out, 즉 학습에 없던 생성기 → 일반화 측정).

> **주의 — 기본 레이아웃의 F1 0.94는 실력이 아닙니다.** 기본 레이아웃에서는 PII가
> value 열에 고정돼 있어, 모델이 **텍스트를 안 읽고 좌표만으로** 답을 맞힙니다.
> 실제 성능을 보려면 **counterfactual 레이아웃**에서 재세요(아래).

### OCR 오류 전파 측정 (`ocr_propagation`)

"OCR 오류가 하류로 전파된다"는 제안서 §2의 전제를 실측합니다. 제어된 CER로
텍스트를 손상시켜 downstream PII 성능 곡선을 뽑습니다(엔진 없이 CPU 재현 가능).

```bash
# 반드시 counterfactual 데이터로! (기본 레이아웃은 곡선이 평평하게 나옴)
macular run data_gen        --config configs/counterfactual.yaml
macular run ocr_propagation --config configs/counterfactual.yaml
```

개발 환경 실측 결과 (en, 120문서, val=family C):

| CER | 기본 레이아웃 F1 | **counterfactual F1** |
|---:|---:|---:|
| 0.00 | 0.941 | **0.648** |
| 0.10 | 0.941 | 0.585 |
| 0.20 | 0.941 | 0.592 |
| 0.30 | 0.941 | 0.578 |
| 0.50 | 0.941 | **0.530** |
| **낙폭** | **0.000 (평평!)** | **−0.117** |

두 가지가 동시에 확인됩니다:
1. **기본 레이아웃은 곡선이 완전히 평평** = 모델이 텍스트를 아예 안 씀(위치 shortcut).
   좌표-only 감사(F1 1.0)와 정확히 일치하는 현상입니다.
2. **shortcut을 없애면 OCR 품질이 실제로 하류에 전파**됩니다(F1 0.648→0.530).
   이게 제안서 §2가 말하는 오류 전파를 우리 파이프라인에서 실측한 값입니다.

**아직 GPU가 필요한(미구현) 부분**: 실제 VLM 백본 어댑터(Qwen/Ministral/Llama의
single-forward ROI 인코딩 + region recognition, 11.2/RQ6), privacy adversary
(11.9), FHIR compiler(11.11), regional re-reading(11.10). 지금은 `MockBackboneAdapter`
가 그 자리를 채워 CPU 학습을 가능하게 합니다. 실제 백본을 끼우는 게 다음 GPU 작업입니다.

### 실제 VLM 백본 (구현·검증 완료, GPU)

여러 VLM의 vision tower에서 **페이지당 단일 forward + ROI pooling**으로 영역별
시각 특징을 뽑습니다(제안서 §11.2). 백본은 config의 `backbones`로 교체합니다.

```bash
macular run backbone_gate --config configs/backbone_gate.yaml
```

이 실험이 **RQ6 / §25 조기 판정 게이트 #3**입니다: A1(공통 파서 텍스트 특징) vs
A2(백본 시각 특징)를 동일 조건에서 학습해 Δ를 잽니다. Δ≈0이면 백본이 무의미하다는
뜻이므로 **그 결과도 그대로 보고**합니다.

**측정은 F1이 아니라 AP(average precision)로 합니다.** 두 조건이 서로 다른
작동점에 수렴하기 때문입니다(VLM 특징은 고recall 쪽으로 치우침). 실제로 Qwen-8B는
ΔF1=−0.067(악화)인데 ΔAP=+0.152(개선)로 정반대 결론이 나옵니다. 3 seed 반복.

### MACULAR ablation과 프라이버시 측정 (§11.9, §17.1)

```bash
macular run ablation --config configs/ablation.yaml      # 컴포넌트 ablation
```

**프라이버시는 held-out attacker로만 판정합니다.** in-training adversary(GRL)는
학습 신호일 뿐 증거가 아니므로(제안서 §11.9), 다른 아키텍처의 공격자를 **동결된
표현에 사후 학습**시켜 **미학습 영역**에서 평가합니다.

**누출은 "유형 baseline 대비"로 재야 합니다 (중요).** 원문 텍스트 복원 점수는
거의 전부 PII 유형으로 설명됩니다(전화번호는 다 숫자, 이름은 다 글자) — 그리고
유형은 gate가 의도적으로 보존합니다. 실제로 유형 평균만으로 cosine 0.737이
나왔고, 이는 측정된 "누출"(safe 0.733 / raw 0.731)보다 **높습니다.** 그래서
`identity_leakage = attacker cosine − type baseline`을 씁니다.

#### frozen 백본에서의 결과 (Ministral, counterfactual, 3 seed)

| 변형 | 신원 누출 | 임상 F1 |
|---|---:|---:|
| full | +0.0103 | 0.779 |
| hard_mask | +0.0044 | 0.731 |
| no_consistency | +0.0095 | **0.953** |
| no_adversary | +0.0109 | 0.782 |
| no_gate | +0.0042 | 0.933 |
| *(raw, 보호 없음)* | *+0.0091* | — |

**해석:** frozen 백본에서는 gate/consistency/adversary가 **측정 가능한 프라이버시
이득 없이 효용을 크게 희생**합니다. 단 신원 누출 자체가 전부 미미하고(~0.01),
보호 없는 raw도 +0.009뿐이라 **애초에 새어나갈 신원 정보가 거의 없습니다.**

### LoRA 백본 학습 (§11.13 Stage 1)

위 음수 결과의 원인 후보: frozen 백본은 **재가중만 가능하고 표현을 재구성할 수
없어서** gate가 내용을 제거할 여지가 없습니다. 그래서 vision tower를 LoRA로
학습 가능하게 만들었습니다.

```python
VLMBackboneConfig(family="paddleocr_vl", lora=True)   # 385만 학습 파라미터
```

frozen 경로는 특징을 캐시해 재사용하지만, LoRA 경로는 **문서당·스텝당 vision
forward**가 필요해 훨씬 비쌉니다(`max_docs`/`epochs`를 작게).

#### LoRA ablation 결과: 측정 불가 (3 seed × 3 변형)

| 변형 | 누출 감소 (mean ± std) | 임상 F1 | 붕괴 |
|---|---:|---:|---|
| full | −0.0020 ± 0.0066 | 0.61 ± 0.54 | seed 0 |
| no_gate | −0.0458 ± 0.0457 | 0.64 ± 0.56 | seed 1 |
| no_adversary | −0.0261 ± 0.0617 | 0.52 ± 0.32 | — |

**모든 행에서 std가 mean보다 크고, 9회 중 2회가 완전히 붕괴**했습니다(AP 0.15 =
base rate 이하, 임상 F1 정확히 0.0 — 같은 설정의 다른 seed는 0.99 도달).
따라서 **이 실험에서는 gate에 대해서도 adversary에 대해서도 아무 결론을 낼 수
없습니다.** 단일 seed 수치를 인용하지 마세요.

원인 두 가지를 모두 고쳤습니다.
- **학습 불안정** — 24문서 단일 배치 스텝에서 간헐적으로 거대한 gradient가 한 번의
  업데이트로 LoRA 어댑터를 파괴했습니다. `clip_grad_norm_(max_norm=1.0)` 추가,
  그리고 붕괴한 run은 평균에 섞이지 않도록 `collapsed` 플래그로 표시합니다.
- **지표 잡음** — 아래 참조.

### 왜 gate를 버리고 closed-form erasure로 갔는가

지금까지의 실패는 두 층위입니다.

**메커니즘.** "adversary로 표현에서 속성을 지운다"는 접근은 학습 중 adversary만
속이고 정보를 남긴다는 것이 알려진 실패 모드입니다(Elazar & Goldberg, EMNLP 2018;
Gonen & Goldberg, NAACL 2019). 우리 결과는 그 signature와 일치합니다. 후속 계보는
**closed-form linear concept erasure**입니다: INLP(ACL 2020) → RLACE(ICML 2022) →
**LEACE**(NeurIPS 2023). LEACE는 공분산 한 번 추정 + affine 변환이라 학습 루프가
없고, **모든 선형 분류기가 개념을 복원할 수 없다**는 증명을 줍니다.

**측정.** cosine 유사도는 연속값이라 학습 잡음을 그대로 탑니다. 동일 설정 재실행에서
부호가 뒤집혔고, 3-seed에서 std가 mean을 넘었습니다. 이산 지표는 그러지 않습니다.

```bash
macular run erasure_comparison --config configs/erasure_comparison.yaml
```

| 메커니즘 | 내용 |
|---|---|
| `none` | 보호 없음 — 누출 상한 |
| `hard_mask` | 임계값 마스킹 — **바닥선.** 이걸 못 이기면 복잡도를 쓸 이유가 없음 |
| `gate` | MACULAR의 미분 가능 gate |
| `leace` | pooled region feature에 closed-form erasure |

공격 3종을 같은 표현에 붙입니다: **fresh 선형 probe**, **fresh 비선형 probe(MLP)**,
**inversion 공격(exact-match / CER)**.

> **반드시 지킬 것: 비선형 probe를 빼지 마세요.** LEACE는 정의상 *선형* readout만
> 막습니다. 선형 probe만 떨어지고 비선형 probe가 그대로면 그건 프라이버시 결과가
> 아니라 **정리(theorem)를 다시 읽은 것**이고, 개념이 비선형 구조로 옮겨간
> 것뿐입니다. `tests/test_privacy_erasure.py`가 이 성질을 회귀 테스트로 고정해
> 둡니다(비선형으로 인코딩된 개념은 LEACE 이후에도 복원됨을 확인).

#### 결과 (frozen PaddleOCR-VL, counterfactual, 3 seed, n_pii=863)

| 메커니즘 | 임상 F1 | 선형 | 비선형 | EM | CER | ctx선형 | ctx비선형 | ctxEM | ctxCER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.976 | 0.967 | 0.982 | 0.094 | 0.389 | 0.958 | 0.975 | 0.046 | 0.454 |
| **hard_mask** | 0.947 | 0.853 | 0.966 | **0.014** | **0.510** | **0.849** | **0.843** | **0.002** | **1.080** |
| gate | 0.917 | 0.850 | 0.976 | 0.025 | 0.490 | 0.919 | 0.937 | 0.007 | 0.705 |
| leace | 0.942 | 0.916 | 0.942 | 0.085 | 0.456 | 0.871 | 0.917 | 0.049 | 0.549 |

majority 0.847. seed 간 std 0.002~0.023 — **모든 차이보다 작습니다.** 이산 지표가
재현성 문제를 해결했습니다.

세 줄 요약:
1. **자명한 hard mask가 모든 프라이버시 축에서 이깁니다.** graph 이후 두 probe 모두
   majority(0.847)에 도달하고(0.849/0.843) 원문 복원이 사실상 0입니다(EM 0.002).
   효용 비용은 0.947 vs 0.976으로 적당합니다.
2. **미분 가능 gate는 완전히 지배당합니다** — 효용도 낮고(0.917) 누출도 전부 큽니다.
3. **LEACE의 보장은 전이되지 않습니다.** 잔차 공분산이 fit에서 1.5e-07인데 val에서
   0.771입니다. train/val이 PII 생성기 family를 분리해 놓았기 때문이고, 이건 곧
   실제 배포 조건(처음 보는 이름 분포)입니다.

3번은 통제 실험으로 확정했습니다(`scripts/leace_transfer_control.py`): val에 fit해서
val을 probe하면 선형 정확도가 **정확히 majority(0.847)**로 떨어집니다. 구현은
맞고 전이가 안 되는 겁니다. 저비용 수정(이진 개념·shrinkage·rank 절단)과 다중
family 학습(A+D+E, 17,176 영역) 모두 실패했습니다 — 0.932 → 0.925.

#### graph 이후를 반드시 함께 재세요

`z_safe`만 재면 hard mask가 부분적으로 보이고(비선형 0.966), graph 이후에는
완전합니다(0.843). 실제로 하류에 넘어가는 건 후자입니다. **메커니즘 출력만 보고
낸 프라이버시 수치는 아무도 배포하지 않는 표현을 설명하는 것입니다.**

### 러너의 스캐폴드 실험 (아직 GPU 필요)

```bash
macular run train_macular    # 실제 백본 학습 루프 — NotImplementedError로 안내
macular run ablation
macular run backbone_swap
```

GPU 준비 확인:

```bash
pip install -e ".[model]"    # torch, transformers, accelerate
macular probe                # cuda_available: true 와 GPU 목록이 나와야 함
```

---

## 4. 결과를 보내주실 때

다음을 zip으로 묶어 주세요.

- `results/` 폴더 전체 (`env_report.json` 포함)
- counterfactual까지 돌리셨다면 `results/cf/`도
- (선택) `data/meddoc/`는 용량이 크면 안 보내셔도 됩니다 — seed가 고정이라
  제 쪽에서 동일하게 재생성됩니다. 라벨만(`*.jsonl`) 필요하면 그것만.

seed가 `configs/*.yaml`에 고정되어 있어 **같은 config면 누구 컴퓨터에서 돌려도
동일한 데이터**가 재현됩니다.

---

## 5. 구조

```
macular/
├── schema.py              공유 데이터 계약 (Document, Candidate, BBox)
├── data/
│   ├── pii_generators.py  분리 생성기 A/B/C (제안서 §14.4)
│   └── generate.py        합성 문서 + 렌더링
├── baselines/
│   ├── coordinate_only.py shortcut 감사 (돌아감)
│   └── ocr_tesseract.py   OCR baseline (선택)
├── evaluation/
│   └── metrics.py         PII P/R/F1/F2, DZLR, CER/WER (돌아감)
├── models/                MACULAR 코어 스캐폴드 (구현 필요, GPU)
└── runner.py              실험 러너 + CLI
configs/                   coauthor.yaml, counterfactual.yaml
tests/                     pytest 스모크 테스트
```

## 6. 문제가 생기면

- `macular: command not found` → `pip install -e .`를 conda 환경 안에서 다시
- 한국어/일본어 글자가 이미지에서 깨짐 → 렌더링 폰트 문제일 뿐, **라벨·bbox는
  정확**하므로 shortcut 감사·지표에는 영향 없음. OCR baseline만 영향받습니다.
- 그 외에는 `results/env_report.json`과 에러 메시지를 그대로 보내 주세요.
