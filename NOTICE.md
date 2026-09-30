# Attribution / Notice

## 라이선스 구조

- **우리가 작성한 코드·문서**: Apache License 2.0 (`LICENSE`).
- **Kaggle 공개 노트북에서 셀 단위로 보존한 부분**: 원저작자가 Kaggle 기본 라이선스인 Apache License 2.0으로 공개. 저작자 표기를 유지하며 같은 라이선스로 재배포합니다.
- **주최측 baseline(royerlab) 및 Trackastra 코드 조각**: BSD 3-Clause. 아래 원문 고지를 유지합니다.

어느 노트북에 어떤 출처의 코드가 들어 있는지는 아래 표를 보세요. 우리 수정은 노트북 안에서 `# OURS` 주석 또는 `VARIANT` 셀로 표시돼 있습니다.

| 노트북 | 포함된 제3자 코드 |
|---|---|
| `01_trackastra_dog/*` | Trackastra API 호출 (BSD-3). 파이프라인 자체는 우리 코드 |
| `02_own_baseline_training/*` | royerlab baseline을 import·패치 (BSD-3). 래퍼는 우리 코드 |
| `03_public_pipeline/v18~v20` | 우리 파이프라인 + pilkwang 공개 가중치 로더 |
| `03_public_pipeline/v21~v24`, `04_*`, `05_*` | **Kaggle 공개 노트북 셀 원문**(Apache-2.0) + royerlab 예측 스크립트 패치 문자열(BSD-3) + 우리 `VARIANT`/게이트 셀 |
| `04_division_classifier/v28c/e/f` | 우리 코드만 |

## 원저작자

- **주최측 baseline** — royerlab/kaggle-cell-tracking-competition (Thibaut Goldsborough): TemporalUNet3D + SimpleNodeTransformer, `predict_unet_transformer.py`, tracksdata ILP 연결. BSD 3-Clause.
- **Trackastra** — weigertlab/trackastra (Benjamin Gallusser, Martin Weigert): 초반 단계 링커, CTC 사전학습 모델. BSD 3-Clause.
- **공개 가중치·support pack** — Kaggle user pilkwang: `biohub-tracking-support-pack-50ep-v1`, `biohub-temporal-unet3d-seed314159-v1`, `biohub-deepcenter-unet3d-center-prior-v1`. (이 저장소에는 가중치가 포함되지 않음 — Kaggle에서 직접 첨부)
- **공개 노트북 계열** (Apache-2.0, Kaggle): analyticaobscura → nusrati → pilkwang (0.942 파이프라인), "harmonic-fusion" (0.947), "sota-0.948 density-adaptive", "x138" / "harmonic-fusion-v3_2" / "0.953-original" (0.953), 데이터셋 `biohub-v1284-head-s075`.
- **3위 솔루션 write-up** — yu4u. `WRITEUP.md` §9의 비교 대상(코드 미포함).

## 대회 데이터

대회 데이터(`.zarr` 볼륨, `.geff` 라벨, 제출 파일)는 대회 규칙(2.4b, 비참가자에게 재배포 금지)에 따라 이 저장소에 **포함하지 않습니다**. 데이터는 Kaggle 대회 페이지에서 받으세요: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data

## BSD 3-Clause 고지 (royerlab baseline)

Copyright (c) 2026, Thibaut Goldsborough. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## BSD 3-Clause 고지 (Trackastra)

Copyright (c) 2024, Benjamin Gallusser, Martin Weigert. All rights reserved.

(위와 동일한 BSD 3-Clause 조건 및 면책 조항이 적용됩니다.)
