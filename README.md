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
