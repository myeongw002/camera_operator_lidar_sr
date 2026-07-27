# Camera-Guided LiDAR SR 구조 개편 작업 계획

> 대상 저장소: `myeongw002/camera_operator_lidar_sr`  
> 기준 브랜치: `main`  
> 검토 기준 커밋: `b3126e3ebf8882fd1d4534b3ca0b17e32e5ad634`  
> 목적: 기존 CNN 기반 Student–Teacher–Distillation 구조를 **Geometric Prior + LiDAR Relation MLP + Camera Relation Adapter + Transfer Adapter** 구조로 개편한다.

---

## 0. Codex 작업 원칙

이 문서는 구현 지시서다. 작업할 때 다음 원칙을 지킨다.

1. **현재 데이터 전처리·projection·실험 관리·평가 인프라는 최대한 재사용한다.**
2. 기존 모델을 한 번에 삭제하지 말고, 새 모델이 end-to-end로 동작할 때까지 legacy 구현을 보존한다.
3. 초기 버전에서는 다음을 넣지 않는다.
   - Transformer
   - attention
   - global range-image CNN
   - range residual head
   - return probability head
   - query의 provisional camera projection
   - temporal feature
4. 초기 목표는 “성능 최대화”가 아니라 다음 효과를 분리해서 검증하는 것이다.
   - `B0 >` 고정 수직 선형 보간
   - `L0 > B0` LiDAR relation 학습 효과
   - `G > L0` relative-depth guidance 효과
   - `L1-KD > L1-noKD` camera knowledge transfer 효과
   - `L1-KD > L0` 최종 LiDAR-only 개선
5. 관측된 16개 입력 row는 추론 결과에서 그대로 보존한다.
6. 모든 학습·평가는 generated 48개 row를 중심으로 수행한다.
7. LiDAR-only branch에는 절대 horizontal azimuth와 camera FOV flag를 입력하지 않는다.
8. 모델의 모든 correction은 기본 geometric interpolation을 덮어쓰지 않고 **보정량**으로 적용한다.
9. 새 코드에는 tensor shape, mask 의미, candidate 순서를 docstring으로 명시한다.
10. 각 PR은 단독으로 테스트 가능해야 하며, 기존 파이프라인을 불필요하게 동시에 고치지 않는다.

---

# 1. 최종 목표 구조

## 1.1 단계별 모델

### B0: Fixed Geometric Prior

```text
16ch LiDAR
→ target query elevation 생성
→ 같은 azimuth column의 lower/upper input beam 탐색
→ 실제 elevation 기반 수직 선형 보간
→ 64ch 출력
```

학습 파라미터가 없다.

### L0: LiDAR Relation MLP

```text
16ch LiDAR
→ query별 2×3 local LiDAR context 수집
→ 6개 candidate를 shared Point MLP로 인코딩
→ local relation aggregation
→ LiDAR correction δL 예측
→ geometric prior weight 수정
→ lower/upper anchor의 가중합
```

카메라를 사용하지 않는다.

### G: Camera-Guided Relation Model

```text
Frozen L0
+
6개 observed LiDAR candidate를 relative-depth image에 projection
+
sampled depth relation feature
→ Camera Adapter가 추가 correction δC 예측
→ prior + δL + δC
```

학습 시 카메라 FOV 내부에서만 사용한다.

### L1: Final LiDAR-Only Model

```text
Frozen L0
+
LiDAR Transfer Adapter
→ camera correction을 흉내 낸 δT 예측
→ prior + δL + δT
```

추론 시 relative depth, camera projection, Camera Adapter를 모두 제거한다.

---

## 1.2 Correction 정의

candidate 순서가 `horizontal_radius=1`일 때:

```text
slot 0: lower-left
slot 1: lower-center   ← 직접 interpolation anchor
slot 2: lower-right
slot 3: upper-left
slot 4: upper-center   ← 직접 interpolation anchor
slot 5: upper-right
```

기본 weight:

```text
w_lower_geo = 1 - t
w_upper_geo = t
```

여기서 `t`는 query elevation이 lower에서 upper로 이동한 비율이다.

LiDAR correction:

```text
lower_logit = log(w_lower_geo + eps) - δL
upper_logit = log(w_upper_geo + eps) + δL
```

Camera-guided correction:

```text
lower_logit_G = log(w_lower_geo + eps) - δL - δC
upper_logit_G = log(w_upper_geo + eps) + δL + δC
```

최종 LiDAR-only correction:

```text
lower_logit_L1 = log(w_lower_geo + eps) - δL - δT
upper_logit_L1 = log(w_upper_geo + eps) + δL + δT
```

두 logit에 softmax를 적용한다.

---

# 2. 현재 코드 분류

## 2.1 그대로 재사용

다음 파일은 원칙적으로 유지한다.

```text
scripts/prepare_range_images.py
scripts/precompute_relative_depth.py

src/camera_operator_sr/data/dataset.py
src/camera_operator_sr/data/range_image.py
src/camera_operator_sr/data/normalization.py
src/camera_operator_sr/data/masks.py
src/camera_operator_sr/data/split.py
src/camera_operator_sr/data/collate.py

src/camera_operator_sr/geometry/projection.py
src/camera_operator_sr/geometry/feature_sampling.py
src/camera_operator_sr/geometry/validation.py
src/camera_operator_sr/geometry/visibility.py

src/camera_operator_sr/inference.py

src/camera_operator_sr/training/experiment.py
src/camera_operator_sr/training/reproducibility.py
src/camera_operator_sr/training/resume.py

src/camera_operator_sr/pipeline/runner.py
src/camera_operator_sr/pipeline/stages.py
```

재사용 이유:

- 64→16 synthetic input 생성
- calibrated elevation/azimuth 저장
- relative-depth control mode
- LiDAR→camera projection
- bilinear feature sampling
- generated-row mask
- observed-row fusion
- resume/checkpoint 실험 관리
- FOV/side/rear 평가 기반

---

## 2.2 부분 수정

```text
src/camera_operator_sr/geometry/candidate_graph.py
src/camera_operator_sr/training/checkpoint.py
src/camera_operator_sr/pipeline/commands.py
src/camera_operator_sr/pipeline/config.py

scripts/train_student.py
scripts/train_teacher.py
scripts/train_distill.py
scripts/evaluate_sr.py
scripts/evaluate_teacher.py
scripts/infer.py
```

---

## 2.3 Legacy 보존 후 대체

```text
src/camera_operator_sr/models/student.py
src/camera_operator_sr/models/teacher.py
src/camera_operator_sr/models/lidar_encoder.py
src/camera_operator_sr/models/depth_encoder.py
src/camera_operator_sr/models/fusion.py
src/camera_operator_sr/models/operator_decoder.py
src/camera_operator_sr/models/query_embedding.py
src/camera_operator_sr/models/outputs.py

src/camera_operator_sr/losses/total_loss.py
src/camera_operator_sr/losses/distillation.py
src/camera_operator_sr/evaluation/operator_metrics.py
```

기존 구현은 새 구조가 정상 작동할 때까지 삭제하지 않는다.

권장 방식:

```text
src/camera_operator_sr/models/legacy/
src/camera_operator_sr/losses/legacy/
```

또는 기존 파일을 유지하고 새 파일명을 별도로 둔다.

---

# 3. 새 파일 구조

```text
src/camera_operator_sr/models/relation/
├── __init__.py
├── outputs.py
├── geometric_prior.py
├── local_context.py
├── point_encoder.py
├── relation_aggregator.py
├── lidar_relation.py
├── camera_adapter.py
├── transfer_adapter.py
├── lidar_model.py
├── guided_model.py
└── final_model.py

src/camera_operator_sr/losses/relation/
├── __init__.py
├── supervised.py
├── guidance.py
├── distillation.py
└── masks.py

src/camera_operator_sr/evaluation/relation_metrics.py
```

기존 공개 import가 깨지지 않도록 필요 시 compatibility wrapper를 둔다.

---

# 4. Tensor 계약

## 4.1 입력 batch

기존 Dataset 계약을 유지한다.

```python
batch["lidar"]["range"]       # [B, 1, 16, W]
batch["lidar"]["intensity"]   # [B, 1, 16, W]
batch["lidar"]["valid"]       # [B, 1, 16, W]
batch["lidar"]["elevation"]   # [B, 16] 또는 [16]
batch["lidar"]["azimuth"]     # [B, W] 또는 [W]

batch["target"]["range"]      # [B, 1, 64, W]
batch["target"]["valid"]      # [B, 1, 64, W]
batch["target"]["elevation"]  # [B, 64] 또는 [64]

batch["camera"]["relative_depth"]  # [B, 1, H_img, W_img]
batch["camera"]["depth_valid"]     # [B, 1, H_img, W_img]

batch["calibration"]["K"]           # [B, 3, 3]
batch["calibration"]["T_cam_lidar"] # [B, 4, 4]
```

---

## 4.2 Candidate context

```python
candidate_ranges      # [B, 64, W, 6]
candidate_intensity   # [B, 64, W, 6]
candidate_valid       # [B, 64, W, 6]
delta_elevation       # [B or 1, 64, W, 6]
delta_azimuth         # [B or 1, 64, W, 6]
```

관측 row도 candidate graph에 포함될 수 있으나 loss는 generated mask로 제한한다.

---

## 4.3 Relation output

```python
@dataclass
class RelationOutput:
    prior_weights: Tensor       # [B, 64, W, 2]
    final_weights: Tensor       # [B, 64, W, 2]
    correction: Tensor          # [B, 1, 64, W]

    anchor_ranges: Tensor       # [B, 64, W, 2], order=[lower, upper]
    anchor_valid: Tensor        # [B, 64, W, 2]
    has_anchor: Tensor          # [B, 1, 64, W]

    predicted_range: Tensor     # [B, 1, 64, W]
    relation_feature: Tensor    # [B, 64, W, D]
```

Camera-guided output에는 추가 필드를 둔다.

```python
camera_correction: Tensor       # [B, 1, 64, W]
camera_context_valid: Tensor    # [B, 1, 64, W]
guided_weights: Tensor          # [B, 64, W, 2]
guided_range: Tensor            # [B, 1, 64, W]
```

Final output에는 다음을 둔다.

```python
transfer_correction: Tensor     # [B, 1, 64, W]
```

---

# 5. PR 1 — B0 Geometric Prior 및 Candidate 계약 정리

## 5.1 `candidate_graph.py` 수정

기존 기능을 유지하면서 다음 metadata를 추가한다.

```python
@dataclass(frozen=True)
class CandidateIndex:
    row_indices: Tensor
    column_offsets: Tensor
    geometric_valid: Tensor
    delta_elevation: Tensor
    delta_azimuth: Tensor

    lower_center_slot: int
    upper_center_slot: int
    query_fraction: Tensor
```

`horizontal_radius=1` 기준:

```python
lower_center_slot = 1
upper_center_slot = 4
```

`query_fraction`의 의미를 명확히 고정한다.

```text
t = 0 → lower beam
t = 1 → upper beam
```

실제 elevation 기준으로 계산한다.

### 주의

- input elevation이 ascending/descending 모두 가능해야 한다.
- exact observed row에서는 lower==upper일 수 있다.
- generated row만 학습 대상이므로 exact observed row는 추론 fusion과 평가 flag에만 사용한다.
- horizontal wrapping 동작을 유지한다.

---

## 5.2 `geometric_prior.py`

구현 대상:

```python
@dataclass
class PriorOutput:
    weights: Tensor          # [B,H,W,2]
    anchor_ranges: Tensor    # [B,H,W,2]
    anchor_valid: Tensor     # [B,H,W,2]
    has_anchor: Tensor       # [B,1,H,W]
    predicted_range: Tensor  # [B,1,H,W]
```

invalid 규칙:

```text
lower valid, upper valid:
    elevation 기반 linear interpolation

lower valid, upper invalid:
    lower weight = 1

lower invalid, upper valid:
    upper weight = 1

둘 다 invalid:
    has_anchor = 0
    prediction = 0
```

---

## 5.3 B0 평가 지원

새 model 또는 evaluator wrapper:

```python
class GeometricBaselineModel(nn.Module):
    def forward(self, batch) -> RelationOutput:
        ...
```

`evaluate_sr.py`가 B0 checkpoint 없이도 평가 가능하도록 선택지를 추가할 수 있다.

권장 CLI:

```bash
python scripts/evaluate_sr.py \
  --model-type geometric_prior \
  --dataset-root ... \
  --split-file ... \
  --output-root ...
```

기존 checkpoint 기반 CLI를 유지하려면 B0 전용 script를 추가한다.

---

## 5.4 PR 1 테스트

추가:

```text
tests/test_geometric_prior.py
tests/test_candidate_query_fraction.py
tests/test_relation_candidate_slots.py
```

필수 케이스:

- uniform elevation
- nonuniform elevation
- ascending elevation
- descending elevation
- lower/upper 모두 valid
- lower만 valid
- upper만 valid
- 모두 invalid
- exact observed row
- column 0/W-1 wrapping
- output shape
- generated row mask와 일치

### PR 1 완료 조건

- B0가 end-to-end 평가 가능
- observed row fusion 정상
- generated 48 row만 MAE에 포함
- 기존 range projection 테스트 통과

---

# 6. PR 2 — L0 LiDAR Relation MLP

## 6.1 Local normalization

`local_context.py`에 구현한다.

각 query의 6개 candidate에 대해:

```text
log_range = log(range + eps)
median = valid candidate의 median
MAD = valid candidate의 median absolute deviation
normalized_log_range = (log_range - median) / (MAD + eps)
```

클램프 권장:

```python
normalized_log_range = normalized_log_range.clamp(-5.0, 5.0)
```

모두 invalid일 때 NaN이 발생하지 않도록 한다.

Intensity는 다음 중 하나로 정규화한다.

- 데이터 전처리에서 이미 [0,1]이면 그대로 사용
- 아니면 local median/MAD 또는 dataset-level scaling

초기 V1은 raw normalized intensity + validity로 충분하다.

---

## 6.2 Candidate feature

각 candidate token에 다음을 넣는다.

```text
1. normalized log range
2. normalized intensity
3. validity
4. delta elevation
5. sin(delta azimuth)
6. cos(delta azimuth)
7. is_upper
8. is_center
9. horizontal_offset_normalized
```

예상 `point_input_dim=9`.

절대 azimuth는 넣지 않는다.

---

## 6.3 Shared Point Encoder

`point_encoder.py`

```python
class RelationPointEncoder(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, tokens, valid):
        # tokens: [B,H,W,K,F]
        # output: [B,H,W,K,D]
        ...
```

invalid token은 embedding 후에도 mask한다.

---

## 6.4 Relation Aggregator

`relation_aggregator.py`

입력:

```python
embeddings       # [B,H,W,6,D]
candidate_valid  # [B,H,W,6]
anchor_ranges    # [B,H,W,2]
prior_weights    # [B,H,W,2]
query_fraction   # [1 or B,H,W,1]
```

출력 relation vector 구성:

```text
masked mean embedding                  D
masked max embedding                   D
lower-center embedding                 D
upper-center embedding                 D
upper-center - lower-center             D
abs(upper-center - lower-center)        D
query fraction                          1
normalized upper-lower range diff       1
valid candidate ratio                   1
prior weight entropy                    1
```

총 relation dimension을 runtime에서 계산하거나 명시한다.

masked max는 all-invalid에서 안전한 0을 반환해야 한다.

---

## 6.5 `lidar_relation.py`

```python
class LidarRelationMLP(nn.Module):
    def __init__(self, relation_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
```

출력 correction:

```python
delta_l = correction_limit * torch.tanh(raw_delta)
```

초기 `correction_limit` 권장값:

```text
3.0
```

극단적인 softmax saturation을 막기 위한 제한이다.

---

## 6.6 `lidar_model.py`

```python
class RelationLidarModel(nn.Module):
    model_type = "relation_l0"
```

forward 순서:

1. candidate index 획득
2. candidate range/intensity/valid gather
3. lower/upper center anchor 추출
4. geometric prior 계산
5. 6개 candidate feature 구성
6. shared point encoder
7. relation aggregator
8. `δL` 예측
9. prior logit 수정
10. softmax
11. lower/upper weighted sum
12. `RelationOutput` 반환

### V1에서 금지

- residual range 추가
- 6개 candidate 전체를 직접 weighted sum
- return head
- CNN
- attention

---

## 6.7 L0 loss

`losses/relation/supervised.py`

```python
def log_range_huber(
    prediction,
    target,
    target_valid,
    mask,
    delta=0.1,
):
    ...
```

권장 loss:

```text
L_range = Huber(log(pred + eps), log(gt + eps))
L_corr  = masked mean(|δL|)
L_total = L_range + λ_corr * L_corr
```

초기:

```text
λ_corr = 1e-3
```

반드시 config로 노출한다.

Loss mask:

```text
generated row
× target valid
× has_anchor
```

---

## 6.8 `train_student.py` 수정

새 CLI:

```bash
--model-type relation_mlp
--point-hidden-dim 24
--relation-hidden-dim 64
--correction-limit 3.0
--correction-reg-weight 0.001
```

기본값은 새 구조로 바꿔도 되지만, 기존 legacy 모델 비교를 위해:

```bash
--model-type legacy_operator
```

를 일정 기간 유지하는 것이 좋다.

manifest/model_config에 architecture version을 기록한다.

---

## 6.9 PR 2 테스트

```text
tests/test_relation_point_encoder.py
tests/test_relation_aggregator.py
tests/test_relation_lidar_model.py
tests/test_relation_l0_loss.py
tests/test_relation_model_invalid.py
```

필수 조건:

- output shape 정확
- invalid candidate가 pooling에 영향 없음
- 절대 azimuth input 없음
- correction=0일 때 B0와 동일
- positive correction이 upper weight 증가
- negative correction이 lower weight 증가
- correction limit 적용
- 관측 row fusion 유지
- backward finite
- all-invalid batch에서도 loss finite

### PR 2 완료 조건

- B0와 L0가 동일 evaluator에서 비교 가능
- `L0 > B0` 여부를 full/boundary/side/rear로 확인 가능
- 기존 pipeline의 student stage가 새 모델로 완료됨

---

# 7. PR 3 — Camera Relation Adapter

## 7.1 L0 freeze

Phase 3에서 L0 전체를 고정한다.

```python
for parameter in l0.parameters():
    parameter.requires_grad_(False)
l0.eval()
```

optimizer에는 Camera Adapter parameter만 넣는다.

```python
optimizer = AdamW(camera_adapter.parameters(), ...)
```

L0 BatchNorm은 없지만 eval mode를 명시한다.

---

## 7.2 Candidate 3D projection

현재 재사용:

```text
range_image_to_pointcloud
project_lidar_points
sample_image_features
build_depth_channels
```

구현 순서:

1. input 16ch range image를 3D point cloud로 변환
2. 16×W observed point를 image에 projection
3. `build_depth_channels()`의 4채널 feature를 observed point 위치에서 sampling
4. sampled observed feature map을 `[B,C,16,W]` 또는 `[B,16,W,C]`로 복원
5. 기존 candidate index를 사용해 query별 6개 depth candidate를 gather

query 자체는 projection하지 않는다.

---

## 7.3 Camera context validity

각 query마다:

```text
lower-center projection valid
upper-center projection valid
유효 depth candidate 수 >= min_valid_depth_candidates
```

초기값:

```text
min_valid_depth_candidates = 4
```

`camera_context_valid`:

```python
# [B,1,H,W]
```

GT query point projection을 model forward 입력으로 사용하지 않는다.

GT visibility는 평가/학습 mask 진단용으로만 유지할 수 있다.

---

## 7.4 Camera candidate token

각 candidate마다:

```text
sampled normalized relative depth
sampled depth gradient x
sampled depth gradient y
sampled depth validity
projection validity
LiDAR point embedding
is_upper
is_center
horizontal offset
```

Camera Adapter는 raw image 전체를 CNN으로 인코딩하지 않는다.

---

## 7.5 `camera_adapter.py`

구조:

```text
candidate-wise shared Camera Point MLP
→ masked mean/max
→ lower/upper center depth relation
→ camera correction δC
```

권장 크기:

```text
camera point dim: 24
camera relation head: 64 → 32 → 1
camera correction limit: 3.0
```

---

## 7.6 `guided_model.py`

```python
class CameraGuidedRelationModel(nn.Module):
    model_type = "relation_camera"
```

구성:

```python
self.l0
self.camera_adapter
```

출력:

- L0 output
- `δC`
- guided weights
- guided range
- camera context valid

Guided correction:

```text
prior + δL + δC
```

L0 output은 detach해서 Camera Adapter만 학습한다.

---

## 7.7 Phase 3 loss

```text
L_guided_range
+
λ_camera_corr * |δC|
```

mask:

```text
generated
× target valid
× has_anchor
× camera_context_valid
```

초기에는 boundary oversampling을 dataset sampler 또는 query sampler로 추가할 수 있으나, 먼저 일반 random frame batch로 동작을 확인한다.

---

## 7.8 `train_teacher.py` 의미 변경

기존 script 이름은 유지해도 된다.

변경 사항:

- baseline checkpoint는 `RelationLidarModel`이어야 한다.
- L0를 freeze한다.
- Camera Adapter만 optimizer에 넣는다.
- checkpoint에는 L0 source checkpoint를 기록한다.
- checkpoint payload는 Camera Adapter만 저장할지 전체 wrapper를 저장할지 한 가지로 통일한다.

권장:

```text
전체 wrapper state_dict 저장
+
source_checkpoints["l0"]
```

---

## 7.9 Depth control 실험 유지

기존 dataset mode를 그대로 사용한다.

필수 Teacher 실험:

```text
teacher_correct
teacher_none
teacher_frame_shuffled
teacher_spatial_shuffled
teacher_constant
```

판정:

```text
G(correct) > L0
G(correct) > G(shuffled/constant/none)
```

---

## 7.10 PR 3 테스트

```text
tests/test_camera_candidate_sampling.py
tests/test_camera_context_mask.py
tests/test_camera_relation_adapter.py
tests/test_guided_relation_model.py
tests/test_relation_teacher_freeze.py
```

필수 조건:

- Camera Adapter 이외 parameter gradient 없음
- query GT range를 projection에 사용하지 않음
- invalid projection이 depth token을 오염시키지 않음
- `δC=0`이면 guided output=L0
- shuffled/constant/none mode가 end-to-end 동작
- camera FOV 밖에서 camera_context_valid=0
- all-invalid depth batch에서도 loss finite

### PR 3 완료 조건

- L0와 G를 FOV 내부에서 비교 가능
- depth control 결과 생성
- Camera branch가 depth를 실제로 사용하는지 검증 가능

---

# 8. PR 4 — Transfer Adapter 및 Distillation

## 8.1 모델 구성

```python
class FinalRelationModel(nn.Module):
    model_type = "relation_l1"

    self.l0
    self.transfer_adapter
```

L0는 기본적으로 freeze한다.

Transfer Adapter 입력:

```text
L0 relation feature
L0 correction δL
L0 lower/upper weights
prior weights
normalized upper-lower range difference
weight entropy
valid candidate ratio
```

출력:

```text
δT
```

최종:

```text
prior + δL + δT
```

---

## 8.2 Transfer Adapter 구조

```text
input relation vector
→ Linear 64
→ SiLU
→ Linear 32
→ SiLU
→ Linear 1
→ bounded correction
```

Camera Adapter와 출력 공간을 동일하게 유지한다.

---

## 8.3 Frozen Teacher

Phase 4에서:

```text
L0: freeze
Camera-guided model G: freeze
Transfer Adapter: train
```

optimizer:

```python
optimizer = AdamW(transfer_adapter.parameters(), ...)
```

초기 V1에서는 L0 fine-tuning을 금지한다.

추후 ablation으로만 low-LR joint fine-tuning을 허용한다.

---

## 8.4 Advantage gating

기존 advantage 개념을 재사용한다.

```text
e_L0 = |r_L0 - r_GT|
e_G  = |r_G  - r_GT|
```

soft advantage:

```text
A = sigmoid((e_L0 - e_G - margin) / temperature)
```

최종 KD mask:

```text
generated
× target valid
× L1 has_anchor
× camera_context_valid
× camera confidence
× advantage
```

GT-visible mask가 필요한지 별도 ablation으로 둔다. 새 구조의 기본 mask는 observed candidate projection 유효성에 맞춘다.

---

## 8.5 Distillation loss

### 전체 360° supervised loss

```text
L_360 = log-range Huber(L1, GT)
```

적용:

```text
generated × target valid × has_anchor
```

### Weight KD

```text
L_weight_KD = A × KL(stopgrad(w_G) || w_L1)
```

2-anchor distribution을 사용한다.

### Correction KD

Camera Adapter의 추가 correction과 Transfer Adapter correction을 맞춘다.

```text
L_corr_KD = A × SmoothL1(δT, stopgrad(δC))
```

`δL+δC`와 `δL+δT`를 맞추지 말고 추가 correction끼리 맞춘다.

### Regularization

```text
L_transfer_reg = mean(|δT|)
```

전체:

```text
L4 =
L_360
+ λ_weight_kd * L_weight_KD
+ λ_corr_kd * L_corr_KD
+ λ_transfer_reg * L_transfer_reg
```

초기 예시:

```text
λ_weight_kd = 1.0
λ_corr_kd = 0.1
λ_transfer_reg = 1e-3
```

모두 config로 노출한다.

---

## 8.6 L1-noKD 대조군

동일한 구조와 parameter 수로 KD 없이 학습한다.

```text
L1-noKD:
L_360 + regularization

L1-KD:
L_360 + weight KD + correction KD + regularization
```

모델 생성·seed·epoch·optimizer 설정을 동일하게 맞춘다.

---

## 8.7 `train_distill.py` 수정

CLI 예:

```bash
--weight-kd-weight 1.0
--correction-kd-weight 0.1
--transfer-reg-weight 0.001
--advantage-mode soft
--range-advantage-margin 0.1
--range-advantage-temperature 0.1
--disable-kd
```

`--disable-kd`로 L1-noKD를 생성할 수 있게 한다.

통계 기록:

```text
mean camera advantage
KD active count/ratio
mean |δC|
mean |δT|
correction sign agreement
weight KL
FOV/side/rear validation MAE
```

---

## 8.8 PR 4 테스트

```text
tests/test_transfer_relation_adapter.py
tests/test_relation_distillation_loss.py
tests/test_relation_advantage_mask.py
tests/test_relation_distillation_freeze.py
tests/test_relation_distillation_end_to_end.py
```

필수 조건:

- L0와 Camera Adapter gradient 없음
- Transfer Adapter에 gradient 존재
- advantage=0이면 KD loss 기여 0
- `δT=δC`이면 correction KD≈0
- `w_L1=w_G`이면 weight KD≈0
- FOV 밖 KD mask=0
- 전체 360° supervised loss는 FOV 밖에도 적용
- `--disable-kd`가 같은 구조로 정상 학습

### PR 4 완료 조건

- L0, G, L1-noKD, L1-KD를 동일 evaluator로 비교
- side/rear non-FOV 성능 비교 가능
- KD active ratio와 correction 통계 저장

---

# 9. Checkpoint 및 모델 로딩 개편

## 9.1 Schema

```python
CHECKPOINT_SCHEMA_VERSION = 4
```

새 `model_config` 필드:

```text
model_type
architecture_version
horizontal_radius
point_input_dim
point_hidden_dim
relation_hidden_dim
correction_limit
use_intensity
min_valid_depth_candidates
```

기존 residual 관련 필드는 새 모델에서는 저장하지 않는다.

---

## 9.2 Model factory

새 파일 권장:

```text
src/camera_operator_sr/models/factory.py
```

```python
def build_model(model_config: dict) -> nn.Module:
    model_type = model_config["model_type"]

    if model_type == "legacy_operator":
        ...
    elif model_type == "relation_l0":
        ...
    elif model_type == "relation_camera":
        ...
    elif model_type == "relation_l1":
        ...
    else:
        raise ValueError(...)
```

`evaluate_sr.py`, `infer.py`에서 `LidarOperatorStudent`를 하드코딩하지 않는다.

---

## 9.3 Geometry metadata

기존 geometry 검증을 유지한다.

추가 검증:

```text
candidate slot convention
architecture version
point feature definition version
```

권장 metadata:

```python
candidate_layout = "lower[-1,0,+1],upper[-1,0,+1]"
anchor_slots = [1, 4]
```

---

# 10. 평가 코드 개편

## 10.1 유지

- global range MAE/RMSE
- beam별 metric
- distance bin별 metric
- full/camera/side/rear/transition region
- boundary/interior
- generated row mask
- count-weighted aggregation

---

## 10.2 Relation metric 추가

`evaluation/relation_metrics.py`

최소 metric:

```text
prior_mae
final_mae
mean_abs_lidar_correction
mean_abs_camera_correction
mean_abs_transfer_correction
nonzero_correction_ratio
anchor_selection_accuracy
weight_entropy
camera_transfer_sign_agreement
guided_final_weight_kl
```

### Anchor selection accuracy

GT에 더 가까운 anchor:

```text
abs(gt-lower) < abs(gt-upper) → lower
abs(gt-upper) < abs(gt-lower) → upper
```

예측 weight가 더 큰 anchor와 비교한다.

동률과 invalid anchor 처리 규칙을 명시한다.

---

## 10.3 필수 비교 출력

최종 summary에 다음 모델 결과가 한눈에 비교 가능해야 한다.

```text
B0
L0
G-correct
G-none
G-shuffled
L1-noKD
L1-KD
```

영역:

```text
full
camera_frustum
side
rear
boundary
camera_boundary
```

---

# 11. Pipeline config 개편

예시:

```yaml
model:
  architecture: relation_mlp
  architecture_version: 1
  horizontal_radius: 1
  point_hidden_dim: 24
  relation_hidden_dim: 64
  correction_limit: 3.0
  use_intensity: true

student:
  enabled: true
  experiment_name: relation_l0
  epochs: 30
  batch_size: 2
  learning_rate: 0.0003
  correction_reg_weight: 0.001

teachers:
  correct:
    enabled: true
    experiment_name: relation_g_correct
    depth_mode: correct
    epochs: 20
    batch_size: 1
    learning_rate: 0.0002
  none:
    enabled: true
    experiment_name: relation_g_none
    depth_mode: none
  frame_shuffled:
    enabled: true
    experiment_name: relation_g_frame_shuffled
    depth_mode: frame_shuffled
  constant:
    enabled: true
    experiment_name: relation_g_constant
    depth_mode: constant

distillation:
  enabled: true
  experiment_name: relation_l1_kd
  teacher: correct
  epochs: 20
  batch_size: 1
  learning_rate: 0.0001
  advantage_mode: soft
  weight_kd_weight: 1.0
  correction_kd_weight: 0.1
  transfer_reg_weight: 0.001
```

L1-noKD는 별도 experiment로 실행하거나 config에 두 개의 distillation variant를 지원한다.

---

# 12. 실험 순서

## Experiment A — B0

목표:

- geometry와 mask 정상 확인
- smooth에서는 합리적
- boundary에서는 예상대로 실패

출력:

```text
full MAE/RMSE
boundary MAE/RMSE
beam metrics
distance metrics
region metrics
```

---

## Experiment B — L0

비교:

```text
B0 vs L0
```

성공 조건:

```text
L0 full MAE < B0
L0 boundary MAE < B0
L0 smooth MAE가 크게 악화되지 않음
```

---

## Experiment C — G

비교:

```text
L0
G-correct
G-none
G-shuffled
G-constant
```

성공 조건:

```text
G-correct camera-FOV MAE < L0
G-correct boundary MAE < L0
G-correct > G-none/shuffled/constant
```

실패하면 Phase 4로 넘어가지 않는다.

---

## Experiment D — Distillability probe

Transfer Adapter가 FOV 내부에서 `δC`를 예측할 수 있는지 확인한다.

metric:

```text
correction MAE
correction sign agreement
weight KL
```

zero-correction baseline보다 좋아야 한다.

---

## Experiment E — L1

비교:

```text
L0
L1-noKD
L1-KD
G
```

핵심 성공 조건:

```text
L1-KD > L1-noKD
L1-KD > L0
side/rear에서도 L1-KD > L1-noKD
smooth 영역 악화 제한
```

---

# 13. 구현 시 주의할 오류

## 13.1 Query projection leakage

V1에서는 query GT range를 사용해 camera pixel을 만들지 않는다.

금지:

```python
query_xyz = gt_query_range * query_direction
```

Camera feature는 observed 16ch LiDAR candidate projection에서만 가져온다.

---

## 13.2 Candidate 순서 혼동

현재 candidate 순서가 lower 먼저인지 upper 먼저인지 unit test로 고정한다.

모든 코드에서:

```text
anchor order = [lower, upper]
```

를 유지한다.

---

## 13.3 Row index 대신 elevation 사용

geometric prior는 target row index 비율이 아니라 실제 elevation angle로 계산한다.

---

## 13.4 Invalid range의 log

invalid range=0에 직접 log를 적용한 뒤 feature로 쓰지 않는다.

반드시 validity mask와 clamp를 함께 사용한다.

---

## 13.5 all-invalid masked max

`-inf`가 후속 MLP에 들어가지 않도록 all-invalid query에서 0으로 대체한다.

---

## 13.6 Camera branch 추가 학습 혼입

Phase 3에서 L0 parameter가 optimizer에 포함되면 안 된다.

Phase 4에서 L0와 Camera Adapter가 optimizer에 포함되면 안 된다.

각 테스트에서 gradient 존재 여부를 검사한다.

---

## 13.7 절대 azimuth shortcut

LiDAR Relation MLP와 Transfer Adapter input에 다음을 넣지 않는다.

```text
absolute azimuth
camera frustum flag
image pixel coordinate
```

Camera Adapter 자체는 projection을 위해 image 좌표를 내부적으로 사용할 수 있지만, Transfer target 생성 후 LiDAR branch 입력에는 전달하지 않는다.

---

## 13.8 평가 leakage

관측 16개 row를 generated-row range metric에 포함하지 않는다.

---

# 14. 완료 정의

다음이 모두 충족되면 구조 개편 1차 완료로 본다.

- [ ] B0 model/evaluator 구현
- [ ] 2×3 candidate layout unit test
- [ ] Relation L0 forward/backward 구현
- [ ] L0 checkpoint train/resume/infer 지원
- [ ] Camera Adapter 구현
- [ ] L0 freeze 검증
- [ ] correct/none/shuffled/constant depth 실험 지원
- [ ] Transfer Adapter 구현
- [ ] Camera/L0 freeze 검증
- [ ] L1-noKD 및 L1-KD 지원
- [ ] advantage-gated 2-anchor weight KD
- [ ] correction KD
- [ ] FOV/side/rear/boundary 평가
- [ ] relation-specific metric CSV
- [ ] pipeline dry-run 정상
- [ ] synthetic end-to-end test 정상
- [ ] 관측 16개 row 보존
- [ ] legacy model checkpoint 로딩 정책 명시
- [ ] README 구조 설명 갱신

---

# 15. Codex가 먼저 수행할 작업

아래 순서로 진행한다.

## 첫 번째 작업

1. 현재 테스트 전체 실행
2. 현재 main 상태를 baseline으로 기록
3. `candidate_graph.py`의 candidate 순서를 unit test로 고정
4. `query_fraction`과 center anchor slot을 추가
5. `GeometricBaselineModel` 구현
6. B0 평가 script 또는 model factory 연결
7. PR 1 테스트 통과

## 두 번째 작업

1. `models/relation/` 패키지 생성
2. `RelationOutput`
3. local robust normalization
4. Point Encoder
5. Relation Aggregator
6. Lidar Relation MLP
7. RelationLidarModel
8. 새 supervised loss
9. train/evaluate/infer model factory 연결
10. PR 2 테스트 통과

Camera branch와 distillation은 B0/L0가 정상 작동한 뒤 구현한다.

# 17. 비목표

이번 1차 구조 개편에서 해결하지 않는다.

```text
실제 센서별 16ch→64ch cross-sensor generalization
query return probability
convex hull 밖 range 생성
bounded residual
semantic supervision
temporal propagation
multi-camera 360° guidance
attention/transformer
downstream SLAM 개선
```

이 항목은 V1 Relation MLP의 효과가 확인된 뒤 별도 단계로 추가한다.
