# Biohub - Cell Tracking During Development (Kaggle) — 작업 노트

## 대회 요약 (2026-09-06 확인)
- 마감: 참가/팀병합 9/22, 최종제출 9/29 (UTC 23:59). 상금 총 $60k (1위 $18k).
- 과제: 제브라피시 배아 3D+t 형광 영상에서 세포 검출→시간 연결→분열 식별. Code competition, 12h, 인터넷 차단.
- 데이터: zarr v3, (T,Z,Y,X)≈(100,64,256,256) uint16, 배열 경로 `0/`. voxel z 1.625 / y,x 0.40625 µm. zarr attrs에 `image_statistics.quantiles` {"0.001","0.999"}. GT는 .geff, **sparse 라벨**: 클립당 GT 트랙 1~12개. geff attrs에 `estimated_number_of_nodes`. 87.6 GB, train 199클립, 히든 테스트 ~200클립, train/test 배아 disjoint (44b6=밀집, 6bba=희소).
- 평가(metrics.md): FP 엣지는 GT 매칭 노드에서만 계산; adjusted J = J·(1−0.1·(T_pred−T_true)/T_true); score = adj_edge_J + 0.1·div_J.
- 공개 LB 상위 0.95~0.97.

## Kaggle 실행 구조
- Kaggle 이미지: Python 3.12.13, numpy 2.0.2, torch 2.10 (GPU T4 x2, 15 GB/GPU, 4 CPU). **기본 polars 1.35(구버전) → 번들 wheel로 먼저 업그레이드 필요**(tracksdata가 pl.Float16 요구). GPU 할당량 30h/주(주간 초기화), TPU는 별도 할당량이지만 코드가 CUDA 전용이라 미사용. 할당량 소진 시 세션 강제 종료 → Output 유실.
- **prep-bundle v2** (`biohub_prep_bundle_v2.ipynb`, Internet ON): wheel + Trackastra ctc + 공식 저장소 clone → `repo/`. 완료 (`weense/biohub-prep-bundle-v2`, commit 075fc5f).
- **v13 `biohub_v13_train_cellmot`** (GPU 12h): 공식 baseline 학습. Input = 대회 데이터 + prep-v2 (+ 이전 v13 Output → `last.pth` 자동 resume). Output `weights/unet_transformer/split_0/{edge_predictor_best.pth, edge_predictor_last.pth, last.pth, config.json, history.json}`. 단축판 `_short3h.ipynb`(TIME_BUDGET 2.75h)도 있음.
- **v14 `biohub_v14_cellmot_infer`** (GPU): Input = 대회 데이터 + prep-v2 + v13 Output → 12 hold-out CV(score_official, CV_CONFIGS: greedy_t95 / ilp_t95 / ilp_t90 / ilp_t99) → FINAL_CONFIG로 제출. CPU 변형 `biohub_v14_cellmot_cv_cpu.ipynb`(MAX_TEST_SAMPLES=0, TTA off) 실행 완료.
- **v15 `biohub_v15_cellmot_cv_tune`** (CPU 가능, 제출 없음; `build_cv_tune_notebook.py`): Input = 대회 + prep-v2 + v13 Output (+ 이전 v15 Output이면 `raw_cache/` 재사용). 1단계 네트워크(DET_CONFIGS: t95_p3/t99_p3/t95_p5/t99_p5 = det_threshold × pool_kernel_um)를 클립당 1회 돌려 `raw_cache/<det>/<clip>.npz`(TAG = 가중치 md5|tta)로 저장 → 2단계 POST_CONFIGS(ILP appearance/disappearance/division, edge_threshold 0.5, greedy) × MIN_TRACK_LENS [1,3,5,10] 를 초 단위로 스코어. Output `cv_rows.csv`, `cv_summary.csv`, **`best_config.json`**. CPU 예상: det 설정당 ~78분(TTA off) → 4개 ≈ 5.5h + ILP.
- **v16 `biohub_v16_cellmot_submit`** (GPU 제출용; `build_submit_cellmot_notebook.py` = v14 + pool_kernel_um/min_track_len 노브): v15 Output의 `best_config.json`이 붙어 있으면 자동으로 CV 설정 `v15_best` 추가 + FINAL_CONFIG로 사용(USE_BEST_CONFIG=True). CV_CONFIGS = ilp_t95(v14 기준) / ilp_t99_p5_L5 / v15_best.
- 공통 셀 `nb_common.py`(LOCATE: MODEL_DIRS/PKG_DIR/CACHE_DIRS/REPO_DIRS/CKPT_DIRS; PIP: 버전 비교 업그레이드 → 누락 설치 → fallback). 빌더: `build_notebook.py`, `build_finetune_notebook.py`, `build_prep_notebook.py [파일명]`, `build_train_notebook.py`, `build_infer_cellmot_notebook.py`.

## 공식 baseline 학습 (cellmot_train.py / cellmot_infer.py)
- 저장소 코드 그대로 import, 래퍼가 추가: 전체 resume(model+optimizer+epoch), 시간 예산 중단, 매 epoch last/best 저장, cosine lr(1e-4→1e-5, EPOCHS_TOTAL 기준), val = 12 hold-out(동일 규칙), quantile fallback, tqdm 끔, **프레임당 검출 top-K 상한(MAX_DET_PER_FRAME=1200, `detect_and_match` 래핑; 초반 미숙 검출기의 수천 개 피크가 어텐션 n² 메모리 폭발 → 6 GiB OOM 방지)**. BATCH_SIZE 16→8 (T4 OOM).
- 모델: TemporalUNet3D(32,64,128; (1,4,4) 서브샘플 → 64³) + SimpleNodeTransformer, 2.08M 파라미터. 학습 윈도우 17,714/epoch(batch 8 → 2,215 iter), val 993 윈도우.
- **Kaggle 실측 (2026-09-09, v13 1차 완료, 10.04h)**: epoch당 **~2.5h**(train 8850s + val 85s; backward 72%). 4 epoch 완료. val: ep1 acc .9996 recall .9172 → ep4 acc .9995 **recall .9465** (score acc×recall .9461, best=ep4). det-cap hits 318→174→53→71. 60 epoch 목표는 비현실적(15세션) → 다음 주 30h로 v14(3h) + v13 이어서 ~8h(3 epoch) 정도. 속도 개선 후보: AMP(fp16 autocast, train_epoch 복제 필요), 트랜스포머 grad checkpointing 해제, cosine 스케줄 EPOCHS_TOTAL을 현실적 값(예 10)으로 축소.
- 추론(cellmot_infer): 2단계 분리. `predict_raw` (repo `predict_video` → coords int16 (N,4), edges float32 (M,4)=[src,tgt,prob,dist]; `save_raw/load_raw` npz 캐시) → `postprocess` (edge_threshold 재필터 → tracksdata ILP(solution attr) 또는 greedy → `prune_short_tracks`(union-find 연결요소 < min_track_len 제거, id 1..N 재부여)). `predict_clip` = 둘 합침. `make_cfg(..., pool_kernel_um, min_track_len)`. 단위테스트 `test_postprocess.py`(합성 그래프: 12/4/2노드 트랙+고립 5 → ILP 16노드/14엣지, L5 → 12/11, L13 → 0; edge_threshold 0.5 필터 확인) 통과. ILP 특성: 트랙 유지 조건 Σprob > appearance+disappearance → dis 1.4에서 2노드 트랙은 항상 탈락. 알려진 비효율: 엣지 후보 파이썬 루프 O(n²).

## 제출 이력 / CV
| 버전 | 설정 | CV(12클립) | LB |
|---|---|---|---|
| v8 | DoG adapt_a26_merge + Trackastra | 0.793 | 0.766 |
| v11/v12 | + Trackastra 파인튜닝 | 0.8175 | **0.798** |
| v13 | 공식 baseline 자체 학습 4 epoch | val recall .9465 (지표 아님) | – |
| v14-cpu | v13(4 epoch) 가중치, ilp_t95, TTA off, CPU | 0.8196 (78분) | – |
| v15 | CPU 후처리 스윕(TTA off): best t99/ilp_d14/L10 | 0.8406 | – |
| v16 | GPU+TTA: ilp_t95 .8371 / **t99 L5 .8549** / t99 L10(v15_best) .8508 → 제출물은 v15_best | 0.8508 | 제출 대기 |
| v17 | 제출 전용, RUN_CV=False, FINAL t99/L5, ILP 병렬 워커 3개 | (0.8549) | 실행 대기 |
- 12클립 CV ≈ LB + 0.02~0.03. 공개 기록: 50 epoch 모델 clean LB ≈0.913, 후처리 튜닝 0.94.

## v14-cpu CV 진단 (2026-09-09)
- 12클립 micro adjJ 0.8196, divJ 0. node_recall 0.92~1.00(검출 병목 해소). **n_pred/est 1.15~2.26배** → 노드 수 패널티: 패널티 없으면 0.8586 (로그 수치로 재계산해 0.8196 정확히 재현). 최악: 44b6_0b24845f ×1.82(J .818→.751), 6bba_b204cac7 ×2.26(.942→.823). 밀집 44b6 클립은 엣지 FP/FN 자체가 큼(44b6_87bba6c4 J .475, 44b6_e29f0176 .522) → 학습 epoch 추가 필요. 원인 추정: 추론 pool_kernel 3 µm(학습 5 µm)로 핵 하나에 피크 2개 + 임계 0.95의 거짓 검출 트랙. 클립당 CPU 시간 250~580s(첫 클립 1040s) → **CPU로 히든 ~200클립 제출은 불가**(≈22h > 12h).

## v15/v16 결과 (2026-09-12)
- v15(CPU 10.8h, t99_p5 9클립에서 예산 종료): 상위 t99_p3/ilp_d14/L10 .8406, L5 .8389, ilp_d14_a05/L5 .8384, ilp_d20_a10 .8376. t95 기준 L1 .8196 → L10 .8328. greedy는 .64~.66(ILP 필수). edge_threshold 0.5는 손해(.8223). **pool_kernel 3 vs 5 µm는 완전 동일**(다운샘플 후 voxel 1.625 µm → 둘 다 커널 3) → p5는 의미 없음.
- v16(GPU T4, TTA on): TTA만으로 ilp_t95 .8196→.8371. t99/L5 **.8549**(최고), t99/L10 .8508(TTA로 검출이 깨끗해져 L10은 과잉 가지치기: node_recall .79~.96으로 하락). 제출물(FINAL=v15_best=L10) 생성됨, format OK. 클립당 시간: net 55~75s + ILP post 9~276s(밀집 클립) → 4클립 14.4분 = **~216 s/clip → 히든 ~200클립이면 12h 초과 위험**(v16은 CV 67분까지 포함). 
- v17(`build_submit_fast_notebook.py`): RUN_CV=False, FINAL t99/L5, `CI.init_worker/post_task`로 ILP를 spawn ProcessPool(3)에서 병렬(GPU는 다음 클립 진행), 예산 초과 시 placeholder. 합성 데이터 e2e + 워커 결과==인프로세스 결과 검증(test_post_pool.py). 로컬 러너는 spawn 재임포트 때문에 `if __name__=="__main__"` 가드 필요.

## 공개 노트북 계열 분석 (2026-09-08)
- 공개 0.94 = 공식 baseline 50 epoch(pilkwang 가중치) + 후처리 knob. 사용자는 직접 학습 선택. `improvedmetrichacklastcall`은 metric hack → 사용 비권장.

## 다음 단계
1. v16 submission.csv 제출(LB 확인) + **v17 실행 → 제출**(Input: 대회 + prep-v2 + v13 Output; GPU ~4h 예상).
2. 남은 GPU(~24h)로 v13 이어서 학습(EPOCHS_TOTAL 축소, AMP 검토) → 새 가중치로 TTA 포함 CV(v15의 GPU/TTA 버전) → v17 재실행.
3. 분열(divJ) 0 → division 후처리는 아직 미착수.
