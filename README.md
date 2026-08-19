# OpenCV 초 단위 객체 추적기

입력 영상에서 객체를 자동 검출하고 각 객체에 `ID`를 붙여 추적합니다. 결과는
ID와 경계 상자가 표시된 MP4, 그리고 각 ID의 상태를 일정한 시간 간격으로 기록한
CSV 두 파일로 저장됩니다.

## 설치

```bash
cd opencv_object_tracker
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 실행

정확도가 높은 YOLO26s + BoT-SORT + ReID 사람 추적이 기본값입니다. 모델과 트래커
설정이 프로젝트에 포함되어 있어 입력과 출력 경로만 지정하면 됩니다.

```bash
python3 object_tracker.py input.mp4 \
  --csv tracking.csv \
  --output tracked_output.mp4
```

기본 설정은 검출 문턱 `0.02`, 추론 크기 `1280`, 예측 박스 `0.2초`, 내부 ID 기억
`2초`입니다. 한쪽 길이가 3000px 이상인 고해상도 영상은 추론 크기를 `1920`으로
자동 상향합니다. 다른 모델이나 트래커 YAML을 사용할 때만 `--model` 또는
`--tracker-config`을 지정하면 됩니다.

추가 모델 없이, 고정 카메라에서 움직이는 일반 객체를 추적할 때:

```bash
python3 object_tracker.py input.mp4 \
  --detector motion \
  --csv tracking.csv \
  --output tracked_output.mp4
```

OpenCV 내장 HOG로 사람을 추적할 때(별도 모델 파일 불필요, 정확도 낮음):

```bash
python3 object_tracker.py input.mp4 \
  --detector person \
  --csv people.csv \
  --output people_tracked.mp4
```

처리 화면을 보려면 `--show`를 추가합니다. 화면에서 `q`를 누르면 조기 종료됩니다.

```bash
python3 object_tracker.py input.mp4 --show
```

## 5개 영역 GNN 예측

`gnn_zone_predictor.py`는 사람 대신 **영역 하나를 그래프 노드 하나**로 만듭니다.
서로 이동할 수 있는 인접 영역을 엣지로 연결하고, 각 노드의 현재/이전 인원수,
인원 변화, 유입·유출, 새 진입·이탈, 평균 속도와 검출 신뢰도를 사용해 다음 1초의
영역별 전체 인원수를 예측합니다.

기본 쇼핑 영역은 `configs/shopping_zones.json`에 다음 5개로 설정되어 있습니다.

| ID | 영역 | 화면 범위 |
|---|---|---|
| 0 | `top_entrance` | 상단 입구/통로 |
| 1 | `left_aisle` | 왼쪽 통로 |
| 2 | `center_store` | 중앙 매장 |
| 3 | `right_aisle` | 오른쪽 통로 |
| 4 | `bottom_exit` | 하단 출구/대기 공간 |

영역 그래프는 상단 영역과 중간 3개 영역, 중간 영역과 하단 영역을 연결하고 중간
영역끼리도 연결한 5노드·8엣지 구조입니다. 노드와 엣지는 같은 JSON의 `zones`와
`graph_edges`에서 수정할 수 있습니다.

쇼핑 추적 CSV로 GNN을 학습하고 예측 결과와 영역 미리보기를 만드는 명령어:

```bash
python3 gnn_zone_predictor.py results/shopping_improved.csv \
  --video "/원본/영상/경로/shopping.mp4" \
  --model-output models/shopping_zone_gnn.pt \
  --predictions results/shopping_region_gnn_predictions.csv \
  --preview-output results/shopping_region_gnn_zones_preview.jpg
```

저장한 모델로 재학습 없이 다시 예측할 때:

```bash
python3 gnn_zone_predictor.py results/shopping_improved.csv \
  --load-model models/shopping_zone_gnn.pt \
  --epochs 0 \
  --predictions results/shopping_region_gnn_predictions.csv
```

생성 파일의 의미:

- `shopping_region_gnn_predictions.csv`: 영역별 현재/이전 인원, 유입·유출,
  다음 인원 예측, 실제 다음 인원과 오차
- `shopping_zone_gnn.pt`: 재사용 가능한 PyTorch GNN 체크포인트
- `shopping_region_gnn_zones_preview.jpg`: 5개 지역 노드와 현재/예상 인원을 표시한 이미지

`predicted_next_count`는 GNN이 예측한 다음 시각의 영역별 전체 인원입니다. 현재 쇼핑
예제는 약 12초뿐이므로 실행 구조 확인용이며, 실제 배치에서는 최소 수분 이상의 추적
CSV로 다시 학습하는 것이 좋습니다.

## Town Centre 정답·추적 입력 비교

`towncentre_experiment.py`는 `archive/TownCentre-groundtruth.top`을 정제해 1초
단위 정답 CSV로 바꾸고, 깨끗한 정답 입력과 YOLO+BoT-SORT 입력을 각각 사용한
GNN을 같은 미래 정답으로 학습·평가합니다. 시간 순서대로 앞 60%를 학습, 다음
20%를 검증, 마지막 20%를 테스트에 사용합니다.

먼저 `.top`에 정답이 존재하는 약 123초를 5fps로 준비하고 추적합니다.

```bash
ffmpeg -y -i ../archive/TownCentreXVID.mp4 -t 123.2 -vf fps=5 -an \
  -c:v libx264 -preset veryfast -crf 18 \
  results/towncentre_experiment/towncentre_annotated_5fps.mp4

python3 object_tracker.py \
  results/towncentre_experiment/towncentre_annotated_5fps.mp4 \
  --csv results/towncentre_experiment/yolo_tracking_1s.csv \
  --output results/towncentre_experiment/yolo_tracked_5fps.mp4 \
  --model models/yolo26s.pt \
  --tracker-config configs/botsort_persistent.yaml \
  --sample-interval 1 --imgsz 960 --device cpu --min-hits 1
```

그다음 두 입력을 동일한 `.top` 미래 정답과 비교합니다.

```bash
python3 towncentre_experiment.py \
  --tracker-csv results/towncentre_experiment/yolo_tracking_1s.csv \
  --epochs 500
```

수치 결과는 `results/towncentre_experiment/REPORT.md`, 테스트 그래프는
`test_predictions.png`, 60초 검출 비교는 `detection_comparison_60s.jpg`에
저장됩니다. 현재 로컬 `.top`의 마지막 행은 불완전하므로 변환기가 해당 행을
보고하고 제외합니다.

## Town Centre 전체 5분·3개 ROI 시공간 GNN

`towncentre_three_roi_gnn.py`는 `.top`을 사용하지 않고 5분 전체의 추적 CSV로
1초 뒤 추적 인원수를 학습합니다. 화면의 검정 경계선을 따라 왼쪽 인도, 중앙
통행로, 오른쪽 인도 3개 노드를 만들고 경계 박스 아래 중앙점(발 위치)으로 사람을
배정합니다. ROI 경계와 `왼쪽↔중앙↔오른쪽` 그래프는
`configs/towncentre_three_rois.json`에서 수정할 수 있습니다.

전체 영상을 5fps로 준비하고 연속 추적합니다.

```bash
ffmpeg -y -i ../archive/TownCentreXVID.mp4 -vf fps=5 -an \
  -c:v libx264 -preset veryfast -crf 18 \
  results/towncentre_three_roi/towncentre_full_5fps.mp4

python3 object_tracker.py \
  results/towncentre_three_roi/towncentre_full_5fps.mp4 \
  --csv results/towncentre_three_roi/full_tracking_1s.csv \
  --output results/towncentre_three_roi/full_tracked_5fps.mp4 \
  --model models/yolo26s.pt \
  --tracker-config configs/botsort_persistent.yaml \
  --sample-interval 1 --imgsz 960 --device cpu --min-hits 1
```

최근 4·8·12초 입력창과 모델 크기 후보를 검증한 뒤 최종 모델을 학습합니다.

```bash
python3 towncentre_three_roi_gnn.py --epochs 600 --patience 80
```

생성 결과:

- `three_roi_dataset.csv`: 300초×3노드로 집계한 GNN 데이터셋
- `three_roi_stgnn.pt`: ROI 설정과 전처리 정보를 포함한 학습 모델
- `predictions.csv`, `metrics.csv`: 시점·영역별 예측과 평가 지표
- `three_roi_preview_248s.jpg`: ROI 및 발 위치 배정 미리보기
- `full_roi_counts.png`, `test_predictions.png`: 전체 인원과 테스트 예측 그래프
- `REPORT.md`: 발표에 사용할 수 있는 최종 결과 요약

현재 실행에서는 최근 8초를 사용하는 hidden 32 모델이 선택됐고, 마지막 60초의
영역별 1초 뒤 추적 인원 예측 MAE는 1.010명, 반올림 완전일치율은 30.6%, ±1명
이내 정확도는 76.1%였습니다. 이는 사람 검출의 수작업 정답 정확도가 아니라 추적기가
생성한 시계열에 대한 미래 예측 정확도입니다.

## 검수 완료된 6개 ROI로 재학습

`configs/towncentre_six_rois.json`은 검수한 세 직선을 사용해 기존 좌·중·우
영역을 각각 원거리와 근거리로 나눕니다. 노드 연결은 다음과 같습니다.

```text
Z0 far_left ─── Z1 far_center ─── Z2 far_right
     │                │                 │
Z3 near_left ── Z4 near_center ── Z5 near_right
```

학습 전에 ROI만 확인하려면 다음 명령을 사용합니다. 이 명령은 데이터셋을 만들거나
모델을 학습하지 않습니다.

```bash
python3 towncentre_six_roi_gnn.py --roi-review-only --preview-second 248
```

ROI 확인이 끝난 뒤 기존 5분 추적 CSV로 데이터셋 생성, 모델 선택, 학습 및 평가를
한 번에 다시 실행합니다.

```bash
python3 towncentre_six_roi_gnn.py --epochs 600 --patience 80
```

생성 결과는 `results/towncentre_six_roi/`에 저장됩니다.

- `six_roi_review.jpg`: 추적·학습 없이 확인하는 ROI 그림
- `six_roi_preview_248s.jpg`: 추적 객체의 발 위치와 ROI 배정 결과
- `six_roi_dataset.csv`: 300초×6노드, 총 1,800행 학습 데이터
- `six_roi_stgnn.pt`: ROI 설정과 전처리를 포함한 최종 모델
- `predictions.csv`, `metrics.csv`: 시점·영역별 예측 및 평가 지표
- `test_predictions.png`, `full_roi_counts.png`: 테스트 및 전체 시계열 그래프
- `REPORT.md`: 최종 결과 요약

검수 완료 ROI의 현재 실행에서는 최근 8초, hidden 32 모델이 선택됐습니다. 마지막
60초×6노드 테스트 MAE는 0.674명, 반올림 완전일치율은 53.1%, ±1명 이내 정확도는
88.9%입니다. 현재값 유지 기준 MAE 0.750보다는 10.2% 개선됐지만, 이동 외삽 기준
MAE 0.606보다는 높습니다. 이는 사람 검출 정확도가 아니라 추적 시계열의 1초 뒤
영역별 인원 예측 성능입니다.

### 0.2초 추적·이동 보정·흐름 보존 GNN

더 촘촘한 이동 정보를 사용하려면 0.2초 추적 CSV를 만든 뒤 새 파이프라인을
실행합니다. `full_tracking_02s.csv`가 이미 있으면 두 번째 명령만 다시 실행하면
데이터셋 생성, 롤링 검증, 학습, 마지막 60초 평가가 모두 재실행됩니다.

```bash
python3 object_tracker.py \
  results/towncentre_three_roi/towncentre_full_5fps.mp4 \
  --csv results/towncentre_six_roi_02s/full_tracking_02s.csv \
  --output results/towncentre_six_roi_02s/full_tracked_5fps.mp4 \
  --model models/yolo26s.pt --tracker-config configs/botsort_persistent.yaml \
  --sample-interval 0.2 --imgsz 960 --device cpu --min-hits 1

python3 towncentre_flow_gnn.py
```

새 모델은 최근 8초의 0.2초 간격 자료로 1초 뒤를 예측합니다. 픽셀 이동 외삽과
카메라 보정 지면 이동, ROI 간 흐름을 함께 사용하며, 세 롤링 검증 구간에서 모두
손해가 없는 ROI에만 GNN 보정을 적용합니다. 정확히 240초 이전 자료를 학습에 쓰고
마지막 60초는 최종 평가에만 사용합니다.

현재 결과에서 선택된 설정은 hidden 32, learning rate 0.003, dropout 0.2,
19 epochs입니다. Z0만 GNN 보정을 사용하고 Z1~Z5는 픽셀 이동 외삽을 유지했습니다.
정수 초 기준 MAE는 0.597명, 완전일치율은 53.6%, ±1명 정확도는 89.2%입니다.
기존 1초 GNN의 MAE 0.674보다 11.3%, 픽셀 외삽 MAE 0.606보다 1.3% 낮습니다.

결과는 `results/towncentre_six_roi_02s/`에 저장됩니다.

- `flow_dataset_02s.csv`: 1,500시점×6노드, 총 9,000행 데이터셋
- `flow_stgnn.pt`: 모델, ROI, 보정값, ROI별 보정 계수를 포함한 체크포인트
- `flow_predictions.csv`, `flow_metrics.csv`: 전체 예측과 마지막 60초 지표
- `rolling_cv_trials.csv`: 네 후보의 세 구간 롤링 검증 결과
- `flow_test_predictions.png`: 마지막 60초의 ROI별 예측 그래프
- `FLOW_REPORT.md`: 기존 모델과 새 모델의 비교 보고서

## CSV 열

기본적으로 영상 시각 0초, 1초, 2초처럼 1초마다 현재 보이는 각 ID를 한 행씩
기록합니다.

| 열 | 의미 |
|---|---|
| `sample_second` | 예정된 샘플 시각(초) |
| `video_time_s` | 실제 사용한 프레임의 영상 시각(초) |
| `frame_index` | 프레임 번호 |
| `track_id` | 객체에 할당된 고유 ID |
| `label`, `confidence` | 검출 종류와 신뢰도 |
| `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` | 경계 상자(px) |
| `center_x`, `center_y` | 객체 중심 좌표(px) |
| `area_px` | 경계 상자 면적(px²) |
| `displacement_px` | 직전 CSV 기록 이후 이동 거리(px) |
| `speed_px_s` | 위 이동 거리를 시간으로 나눈 속도(px/s) |
| `track_age_s` | 해당 ID를 추적한 시간(초) |
| `tracking_state` | 실제 검출은 `detected`, 누락 보간은 `predicted` |
| `missed_frames` | 마지막 실제 검출 이후 지난 프레임 수 |

엑셀에서 한글이 깨지지 않도록 CSV는 UTF-8 BOM 형식으로 저장됩니다.

## 자주 조정하는 옵션

```text
--sample-interval 1.0     CSV 기록 간격(초)
--min-area 900            작은 노이즈를 제외할 최소 면적
--max-distance 100        같은 ID로 연결할 최대 이동 거리(px/frame)
--max-lost-seconds 2.0    화면에서 사라진 ID를 내부적으로 기억하는 시간
--prediction-seconds 0.2  누락 중 예측 박스를 화면·CSV에 표시하는 시간
--prediction-min-hits 6   예측 박스 허용 전 필요한 실제 검출 횟수
--prediction-min-confidence 0.20  예측 박스 허용 최소 최고 신뢰도
--relink-seconds 0.75     새 외부 ID를 기존 ID로 재연결하는 시간
--bbox-smoothing 0.78     검출 박스 떨림 완화 비율
--min-hits 3              객체로 확정하기 전 필요한 검출 횟수
--classes 0               YOLO 클래스(0=사람, all=전체 객체)
--imgsz 1280              YOLO 추론 해상도
--no-auto-resolution      고해상도 영상의 1920 자동 상향을 끔
--device cpu              YOLO 실행 장치(cpu, mps, 0 등)
```

작은 객체가 누락되면 `--min-area`를 낮추고, 노이즈가 많이 잡히면 높입니다. 빠르게
움직이는 객체의 ID가 자주 바뀌면 `--max-distance`를 높입니다.

## 방식과 제한

- `motion`은 카메라가 고정된 영상에 적합합니다. 카메라 자체가 움직이거나 객체가
  오랫동안 완전히 정지하면 배경 차분 특성상 검출 품질이 낮아질 수 있습니다.
- `person`은 OpenCV 기본 보행자 검출기라 설치는 간단하지만, 복잡한 장면에서는
  YOLO 같은 별도 딥러닝 검출기보다 정확도가 낮을 수 있습니다.
- `yolo`는 카메라 이동과 복잡한 배경에 가장 적합합니다. YOLO가 객체를 검출하고
  BoT-SORT가 ID를 유지합니다. 기본 `--classes 0`은 사람만 추적합니다.
- 검출이 잠깐 끊기면 최근 속도를 이용한 `predicted` 박스를 기본 0.2초 표시합니다.
  ID 자체는 2초 동안 기억해 재연결합니다. 예측 박스는 실제 검출이 아니므로 CSV의
  `tracking_state`로 구분해야 합니다.
- 같은 위치에서 실제 검출 박스가 복구되면 겹치는 예측 박스는 즉시 제거합니다.
- 현재 검출 박스는 매우 강하게 겹칠 때만 중복으로 처리하여, 서로 가까운 두 사람의
  박스가 하나로 사라지지 않게 합니다.
- 속도 단위는 실제 거리 단위가 아닌 `px/s`입니다. m/s가 필요하면 영상의 픽셀과
  실제 거리 사이 보정값이 추가로 필요합니다.
