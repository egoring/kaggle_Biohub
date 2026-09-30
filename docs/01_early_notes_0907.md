# Biohub 세포 추적 대회 — 프로젝트 요약 (2026-09-07 기준)

## 1. 대회 개요

Kaggle **Biohub - Cell Tracking During Development** (호스트: Chan Zuckerberg Biohub). 제브라피시 배아의 3D+시간 형광 현미경 영상에서 세포를 검출하고, 시간축으로 연결하고, 세포 분열을 찾아 계보(lineage) 그래프를 만드는 과제.

- 마감: 참가·팀 병합 **9월 22일**, 최종 제출 **9월 29일** (UTC 23:59)
- 상금: 총 $60,000 (1위 $18k, 2위 $12k, 3위 $8k, 4~5위 $6k, 6~7위 $5k)
- 형식: 노트북 제출(Code Competition), GPU 12시간 이내, 인터넷 차단, 하루 5회 제출
- 데이터: 87.6 GB, zarr v3 볼륨 (T,Z,Y,X)≈(100,64,256,256), voxel 크기 z 1.625 / y·x 0.40625 µm. 정답은 .geff 그래프(희소 라벨, 클립당 GT 세포 ~50개). train 199클립, 히든 테스트도 약 200클립. **train과 test는 배아가 겹치지 않음**.
- 평가: `score = adjusted_edge_jaccard + 0.1 × division_jaccard`. 예측 노드를 GT 노드와 7 µm 이내로 매칭하고, **GT에 매칭된 노드에서만 엣지 FP를 계산**. 노드를 추정 세포 수보다 많이 예측하면 `0.1 × 초과비율`만큼 페널티.
- 공개 리더보드 상위권 0.95~0.97.

## 2. 실행 구조 (Kaggle)

Kaggle 기본 이미지에 zarr가 없고 인터넷이 차단되므로 노트북 2개로 나눔.

| | 노트북 ① 준비 | 노트북 ② 제출 |
|---|---|---|
| 파일 | `biohub_prep_bundle.ipynb` | `biohub_v7_adaptive_radius.ipynb` (최신) |
| Internet | ON | OFF |
| GPU | 없음 | T4 x2 |
| Input | (선택) 대회 데이터 | 대회 데이터 + ①의 Output (Your Work 탭) |
| 역할 | wheel 44개 + Trackastra `ctc` 사전학습 모델 다운로드 → Output | 추론 → submission.csv |

①은 한 번만 실행(완료됨, `weense/biohub-prep-bundle`). ②는 `/kaggle/input` 아래 어디에 있든 `model.pt`와 `trackastra-*.whl`을 자동으로 찾음.

②의 첫 셀 설정값:

```python
VALIDATE_ON_TRAIN = False   # train 클립 간단 검증 (최종 실행 시 False)
MAX_TEST_SAMPLES  = None    # None = 전체, 1~2 = 빠른 디버그
AUTO_CALIBRATE    = True    # 검출 임계값 자동 캘리브레이션 (train 3클립, ~4분)
RUN_CV            = False   # True면 CV_CONFIGS 설정들을 train 5클립에서 공식 규칙으로 채점 (~25분)
FINAL_CONFIG      = "adapt_a26"   # 테스트 추론에 쓸 설정 이름
```

- **CV 실행**: `RUN_CV=True, MAX_TEST_SAMPLES=1` → 제출 횟수 소모 없이 설정 비교
- **최종 실행**: `RUN_CV=False, MAX_TEST_SAMPLES=None` → 약 4~5시간 → Submit
- 셀 수정 후에는 반드시 **Save & Run All**로 커밋해야 반영됨. 파이프라인에 난수가 없어 같은 설정이면 점수는 항상 동일.

## 3. 파이프라인 (v7)

1. 경로 자동 탐색 (87 GB 폴더를 재귀 탐색하지 않도록 프루닝)
2. 오프라인 pip 설치 (`--no-deps`, Kaggle의 torch/numpy는 건드리지 않음)
3. **임계값 캘리브레이션**: train 3클립에서 `thresh_k`를 0.5~3.0으로 훑고, geff 메타의 `estimated_number_of_nodes`에 검출 수가 가장 가까운 값을 선택 (실데이터에서 0.5 선택됨)
4. **클립별 자동 반경**: 5개 프레임을 미리 검출해 프레임당 세포 수 N을 재고 `r = 26.5 × N^(-1/3)` (3.0~6.0 µm)로 반경 결정. 근거: 초기 배아는 부피가 거의 일정해 세포가 많을수록 작음
5. 프레임별 3D 검출: 비등방 DoG(Difference of Gaussians) + 국소 최댓값 (GPU는 torch, CPU는 scipy)
6. 검출점 주위 구형 pseudo-mask → **Trackastra**(사전학습 CTC 3D 트랜스포머) greedy 연결
7. **분열 프루닝**: 두 번째 자식 엣지는 weight ≥ 0.5, 부모–자식 ≤ 8 µm, 전체 노드의 0.6% 예산 안일 때만 유지 (greedy가 분열을 수천 개 남발하던 문제 해결)
8. submission.csv 작성 + 형식 self-check
9. (옵션) 공식 규칙을 재구현한 채점기 `score_official`로 train 클립 CV

## 4. 제출 이력과 CV 결과

| 버전 | 핵심 변경 | 로컬 CV | 공개 LB |
|---|---|---|---|
| v2 | DoG r3.5 + Trackastra greedy | – | 0.600 |
| v4 | 임계값 캘리브레이션(k=0.5) + 분열 프루닝 | 0.558 | 0.623 |
| v6 r4.0 | 검출 반경 4.0 | 0.641 | **0.674** |

로컬 CV는 LB보다 0.03~0.06 낮게 나오지만 순서는 일치 → CV로 설정을 골라도 됨.

**CV 1라운드** (r3.5 기준): base .558 / 분열 끔 .557 / r3.0 .469 / r4.0 .641 / 단순 최근접 연결 .551
→ Trackastra가 단순 연결보다 확실히 낫고, 반경을 키우는 방향이 맞음.

**CV 2라운드**:

| 설정 | micro | 44b6_0b24845f | 44b6_97725144 | 6bba_0e7c0d07 | 6bba_5f89039d | 6bba_aeee7805 |
|---|---|---|---|---|---|---|
| r4.0 | 0.641 | 0.367 | 0.676 | 0.543 | 0.546 | 0.807 |
| r4.5 | 0.761 | 0.313 | 0.620 | 0.585 | 0.750 | 0.899 |
| r5.0 | 0.804 | 0.263 | 0.544 | 0.518 | 0.860 | 0.952 |
| r3.5+병합5µm | 0.570 | 0.445 | 0.694 | 0.528 | 0.422 | 0.771 |
| r4.0+병합6µm | 0.680 | 0.421 | 0.674 | 0.553 | 0.592 | 0.847 |
| r4.0 연결거리128 | 0.641 | r4.0과 완전 동일 → 연결 거리는 무관 | | | | |

핵심 발견: **배아마다 세포 크기가 다르다.** 밀집 배아 44b6(프레임당 ~400개)은 반경 3.5가 최적이고 5.0에서는 recall이 0.51까지 추락. 희소 배아 6bba(~50개)는 반경 5.0에서 0.86~0.95. 중간 밀도(~210개)는 4.5. 히든 테스트는 다른 배아이므로 고정 반경은 위험 → v7의 클립별 자동 반경으로 해결(N≈400→3.5, 210→4.5, 50→6.0으로 각 클립 최적과 일치).

**CV 3라운드** (v7, 실행 예정): r5.0 / r6.0 / adapt_a26 / adapt_a29 / adapt_a26_merge.

## 5. 파일 목록 (C:\kaggle_bio\notebook)

- `biohub_prep_bundle.ipynb` — 노트북 ①
- `biohub_v7_adaptive_radius.ipynb` — 노트북 ② 최신 (이후 버전은 `biohub_vN_*.ipynb`로 새 파일)
- `biohub_trackastra_inference.ipynb` — ② 구버전 누적 수정본 (v7과 동일 내용)
- `pipeline.py` — ②의 3번 셀과 동일한 코드 (로컬 편집·테스트용)
- `make_synthetic.py` — 대회와 같은 zarr v3/geff 레이아웃의 합성 데이터 생성기 (로컬 e2e 검증용)
- `README_biohub_notebook.md` — 영문 사용법
- `upload_biohub_trackastra/` — (미사용) 원본 아티팩트를 데이터셋으로 올리려던 폴더

## 6. 시간·자원

- GPU 기준 클립당 약 60~140초(검출 15초 + Trackastra 15~120초, 노드 수에 비례) → 200클립 약 4~5시간
- Kaggle 무료 GPU 주당 30시간 → 최종 실행 1~2회 + CV 몇 회 가능
- 하루 제출 5회. Save & Run All은 제출 횟수를 소모하지 않음

## 7. 다음 단계

1. v7 CV 3라운드 → adaptive가 모든 클립에서 최적 근처(micro 0.80+)인지 확인 → FINAL_CONFIG 확정 → 최종 실행 → 제출 (LB 0.8 초반 기대)
2. 그 다음 상한은 DoG 검출기 자체. 0.85 이상은 **3D U-Net 검출기 학습**(Internet ON 노트북에서 train .geff로 학습 → Output을 ②에 첨부)이 필요
3. 연결 오류가 많은 클립(recall 0.9인데 FP 많음)에는 Trackastra ILP 모드(`trackastra_mode="ilp"`, motile/ilpy 설치됨) 시험 — 느리므로 시간 예산 확인 필요
