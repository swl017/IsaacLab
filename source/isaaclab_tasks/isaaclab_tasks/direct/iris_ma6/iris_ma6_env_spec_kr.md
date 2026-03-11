# iris_ma6 Environment Specification
## Observation + Interception for Facility Defense

**Version**: Draft 0.1
**Base**: iris_ma5 (V5) — IROS 2026 submission
**Scope**: 6 defenders vs N scripted attackers, facility defense with dynamic observe→intercept role transition

---

## 0) Design Philosophy

iris_ma5는 "관측 전용" 삼각측량 환경이다. iris_ma6는 이를 **관측-요격 통합 방어 시스템**으로 확장한다.

핵심 확장 원칙:

1. **V5 모듈 최대 재활용**: 삼각측량, delay system, covariance, safety 모듈은 그대로 하위 모듈로 유지
2. **역할 전환이 핵심 contribution**: uncertainty가 충분히 낮아지면 관측→요격 전환을 policy가 학습
3. **Scripted attacker + domain randomization**: 공격자는 학습 대상이 아님. 파라메트릭 행동 + 랜덤화로 다양성 확보
4. **점진적 스케일링**: 2v1 → 3v2 → 6v5 → 6v10 curriculum으로 복잡도 관리

---

## 1) Problem Formulation

### 1.1 미션 정의

- **방어 시설**: 환경 중앙에 위치한 반경 `r_facility`의 보호 구역
- **방어 드론**: `C = 6`대, 각각 gimbaled zoom camera 장착 (V5 동일)
- **공격 드론**: `T ∈ {1, 2, ..., 10}`대, scripted 행동으로 시설 접근 시도
- **승리 조건**: 모든 공격 드론을 시설 도달 전 요격
- **패배 조건**: 1대 이상의 공격 드론이 시설 반경 내 진입

### 1.2 Dec-POMDP 확장

V5의 Dec-POMDP tuple을 확장한다:

$$\langle \mathcal{S}, \{A^i\}, \{O^i\}, f, \{O^i\}, R, \gamma, \mathcal{T}, \mathcal{M} \rangle$$

추가 요소:
- $\mathcal{T} = \{t_1, ..., t_T\}$: 공격 드론 집합 (가변 크기, 요격 시 제거)
- $\mathcal{M}: \{1,...,C\} \to \{\texttt{OBSERVE}, \texttt{INTERCEPT}\}$: 에이전트 역할 매핑 (policy가 결정)

글로벌 상태:

$$s_k = \left(\{x_k^{(i)}\}_{i=1}^C,\ \{x_k^{(t)}, \text{alive}_k^{(t)}\}_{t=1}^T,\ p_{\text{facility}}\right)$$

### 1.3 에이전트 상태

**방어 드론** (V5 확장): $x^{(i)} \in \mathbb{R}^{17}$

$$x^{(i)} = [\underbrace{p, v, q, \omega}_{\text{V5: 16D}},\ \underbrace{\alpha, \beta, z}_{\text{gimbal+zoom: 이미 V5에 포함}},\ \underbrace{\texttt{role}}_{\text{NEW: 1D}}]$$

- `role ∈ {0, 1}`: 0 = OBSERVE, 1 = INTERCEPT (discrete, policy output으로 결정)

**공격 드론**: $x^{(t)} \in \mathbb{R}^{7}$

$$x^{(t)} = [p^{(t)},\ v^{(t)},\ \texttt{alive}^{(t)}]$$

- 카메라/gimbal 없음 (scripted이므로 내부 상태 불필요)
- `alive ∈ {0, 1}`: 요격되면 0으로 전환

---

## 2) Action Space

### 2.1 연속 행동 (V5 확장)

Per agent, 8차원 연속 행동:

$$a = [\underbrace{a_{vx}, a_{vy}, a_{vz}, a_{\dot\psi}}_{\text{platform (4D)}},\ \underbrace{a_{\dot\alpha}, a_{\dot\beta}, a_{\dot z}}_{\text{gimbal+zoom (3D)}},\ \underbrace{a_{\text{role}}}_{\text{NEW (1D)}}] \in [-1, 1]^8$$

### 2.2 역할 결정 메커니즘

`a_role`은 continuous output을 thresholding하여 discrete role로 변환:

$$\texttt{role}_k^{(i)} = \begin{cases} \texttt{OBSERVE} & \text{if } a_{\text{role}} < 0 \\ \texttt{INTERCEPT} & \text{if } a_{\text{role}} \geq 0 \end{cases}$$

역할 전환 시 행동 해석이 달라진다:

| 행동 차원 | OBSERVE 모드 | INTERCEPT 모드 |
|-----------|-------------|---------------|
| `a_vx, a_vy, a_vz` | 삼각측량 geometry 최적화 | 타겟 방향 추적 기동 |
| `a_ψ̇` | 관측 자세 유지 | 추적 방향 조정 |
| `a_α̇, a_β̇` | 타겟 centering (V5 동일) | 타겟 centering (동일) |
| `a_ż` | zoom 최적화 (V5 동일) | zoom out (wide FoV 선호) |

> **구현 노트**: 역할에 따라 velocity command 해석을 물리적으로 바꾸지는 않는다. 동일한 velocity controller (PointMass)가 적용되며, 역할 차이는 reward shaping을 통해 행동이 분화되도록 유도한다.

### 2.3 타겟 할당

OBSERVE 모드에서는 V5처럼 (현재는 단일 타겟) 동작하지만, 다중 타겟 환경에서는 **관측 대상 타겟 선택**이 필요하다.

**옵션 A — Implicit allocation (권장, 초기):**
- Gimbal pointing으로 자연스럽게 타겟 선택. 어떤 타겟을 향해 gimbal을 돌리느냐가 곧 할당.
- BBoxRayCaster가 모든 alive 타겟에 대해 bbox를 계산, 가장 중앙에 가까운 타겟을 현재 관측 대상으로 자동 매칭.

**옵션 B — Explicit allocation (향후 확장):**
- 추가 discrete action으로 타겟 인덱스 선택 (action space: `a_target ∈ {1, ..., T}`)

INTERCEPT 모드에서의 타겟 할당:
- 가장 가까운 alive 타겟 자동 할당 (heuristic)
- 또는 observation에서 타겟별 위협도를 포함하여 policy가 암묵적으로 선택

---

## 3) Observation Space

### 3.1 설계 원칙

V5의 고정 크기 관측 벡터를 유지하되, 다중 타겟 정보를 추가한다.
가변 타겟 수 문제는 **고정 슬롯 + zero-padding**으로 처리한다 (set-based encoding은 향후 architecture 연구로).

### 3.2 관측 벡터 구조

**Dimension formula**: `ego(28D) + allies(15D × 5) + targets(12D × T_max) + defense(4D)`

`T_max = 10`, `C = 6` 기준: 28 + 75 + 120 + 4 = **227D**

#### 3.2.1 Ego features (28D) — V5 26D + 2D 추가

| Feature | Dims | Source | Note |
|---------|------|--------|------|
| position $p^w$ | 3 | V5 | |
| yaw $\psi$ | 1 | V5 | |
| linear velocity $v^w$ | 3 | V5 | |
| yaw rate $\dot\psi$ | 1 | V5 | |
| linear acceleration | 3 | V5 | |
| gimbal pitch, yaw | 2 | V5 | |
| body angular velocity | 3 | V5 | |
| ego bbox (primary target) | 4 | V5 | 현재 관측 중인 타겟의 bbox |
| bbox valid | 1 | V5 | |
| time since detection | 1 | V5 | |
| zoom level | 1 | V5 | |
| ray direction | 3 | V5 | |
| **current role** | **1** | **NEW** | 0=observe, 1=intercept |
| **facility direction** | **1** | **NEW** | 시설 방향 각도 (ego 기준) |

**Total**: 26 + 2 = **28D**

#### 3.2.2 Per other-agent features (15D × 5) — V5 동일

V5의 other-agent observation을 `C-1 = 5`명에 대해 반복. 구조 동일:

| Feature | Dims |
|---------|------|
| position | 3 |
| linear velocity | 3 |
| angular velocity | 3 |
| bbox valid (primary target) | 1 |
| ray direction | 3 |
| data age (AoI) | 1 |
| detection age | 1 |

**Total**: 15D × 5 = **75D**

#### 3.2.3 Per-target features (12D × T_max) — NEW

각 공격 드론에 대해 (alive인 경우에만 유효, dead면 zero-padded):

| Feature | Dims | Description |
|---------|------|-------------|
| relative position | 3 | 삼각측량 추정값 또는 prediction (ego 기준 상대좌표) |
| relative velocity | 3 | 추정 속도 (유한 차분 또는 prediction) |
| localization std | 3 | $\sqrt{\text{diag}(\Sigma_X^{(t)})}$, 해당 타겟의 삼각측량 uncertainty |
| alive flag | 1 | 0 = 격추됨, 1 = 활성 |
| AoI (localization) | 1 | 해당 타겟의 마지막 유효 삼각측량 이후 경과 시간 |
| threat level | 1 | $\texttt{threat}^{(t)} = \frac{v_{\text{approach}}^{(t)}}{\|p^{(t)} - p_{\text{fac}}\|}$ (시설 접근 속도 / 거리) |

**Total**: 12D × 10 = **120D**

> **정렬 규칙**: 타겟은 threat level 내림차순으로 정렬하여 슬롯에 배치. 이는 permutation 문제를 완화하고, 가장 위험한 타겟이 항상 첫 번째 슬롯에 오도록 보장.

#### 3.2.4 Global defense features (4D) — NEW

| Feature | Dims | Description |
|---------|------|-------------|
| num alive targets | 1 | 현재 살아있는 공격 드론 수 (normalized: / T_max) |
| num intercepting allies | 1 | 현재 INTERCEPT 모드인 아군 수 (normalized: / C) |
| min target-facility dist | 1 | 가장 가까운 공격 드론의 시설까지 거리 |
| episode progress | 1 | 현재 스텝 / 최대 스텝 |

**Total**: **4D**

---

## 4) Scripted Attacker System

### 4.1 AttackerManager 모듈

V5의 `TargetMovement`를 대체하는 새 모듈: `attacker/attacker_manager.py`

```
AttackerManager
├── AttackerBehaviorCfg       # 행동 파라미터 정의
├── AttackerWaveGenerator     # 공격 wave 생성 (timing, spawn, count)
├── AttackerController        # 개별 공격 드론 velocity controller
└── InterceptResolver         # 요격 판정 로직
```

### 4.2 공격 드론 행동 모델

각 공격 드론은 finite state machine으로 동작:

```
APPROACH → [요격 드론 감지] → EVADE → [회피 성공] → APPROACH
                                    → [요격됨]   → DEAD
         → [시설 도달]      → BREACH (에피소드 실패)
```

#### 4.2.1 APPROACH: 시설 접근

기본 속도:
$$v_d^{(t)} = v_{\text{max}}^{(t)} \cdot \frac{p_{\text{fac}} - p^{(t)}}{\|p_{\text{fac}} - p^{(t)}\|}$$

Path variation (randomized):
- **Direct**: 직선 접근
- **Offset**: 중간 waypoint를 거쳐 우회 접근 (waypoint은 시설 주변 랜덤 배치)
- **Low-altitude**: 고도를 낮춰 접근 (detection 난이도 증가)

#### 4.2.2 EVADE: 회피 기동

Interceptor가 반경 `d_evade_trigger` 내로 접근하면 발동:

$$v_{\text{evade}} = v_{\text{max}}^{(t)} \cdot \hat{n}_{\perp}$$

여기서 $\hat{n}_{\perp}$는 interceptor→attacker 방향에 수직인 랜덤 방향.

Evasion 강도 파라미터:
- `evasion_agility ∈ [0, 1]`: 회피 기동의 가속도 배율
- `evasion_duration`: 회피 기동 지속 시간 (이후 APPROACH 복귀)
- `evasion_probability ∈ [0, 1]`: 회피 시도 확률 (일부 공격 드론은 회피 없이 직진)

#### 4.2.3 Behavior Profiles (Domain Randomization)

에피소드 리셋 시, 각 공격 드론에 랜덤 프로파일 할당:

| Profile | Speed | Evasion Agility | Path | 비율 |
|---------|-------|-----------------|------|------|
| **Kamikaze** | 고속 (12 m/s) | 없음 (0.0) | Direct | 20% |
| **Standard** | 중속 (7 m/s) | 중간 (0.5) | Offset | 40% |
| **Evasive** | 중속 (7 m/s) | 높음 (0.9) | Offset | 25% |
| **Stealth** | 저속 (3 m/s) | 중간 (0.5) | Low-altitude | 15% |

### 4.3 공격 Wave 구성

공격 드론은 한번에 모두 등장하지 않고, wave 단위로 spawn:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_waves` | 1–3 | 총 wave 수 |
| `drones_per_wave` | 2–5 | wave 당 드론 수 |
| `wave_interval` | 5–15 s | wave 간 시간 간격 |
| `spawn_radius` | 500–1000 m | 시설로부터 spawn 거리 |
| `spawn_arc` | 60°–360° | spawn 방위각 범위 (좁으면 집중 공격, 넓으면 분산 공격) |
| `spawn_altitude` | 20–80 m | spawn 고도 범위 |

### 4.4 요격 판정 (InterceptResolver)

요격 성공 조건:

$$\texttt{intercept}^{(i,t)} = \begin{cases} 1 & \text{if } \texttt{role}^{(i)} = \texttt{INTERCEPT} \land \|p^{(i)} - p^{(t)}\| < d_{\text{capture}} \\ 0 & \text{otherwise} \end{cases}$$

요격 시 처리:
- 공격 드론: `alive = 0`, 물리 비활성화
- 방어 드론 생존: 확률 `p_survive`로 결정

$$\texttt{alive\_defender}^{(i)} = \begin{cases} 1 & \text{with prob. } p_{\text{survive}} \\ 0 & \text{with prob. } 1 - p_{\text{survive}} \end{cases}$$

- 생존 시: 즉시 OBSERVE로 역할 복귀, 다음 타겟 할당 가능
- 손실 시: 해당 에이전트 비활성화 (action 무시, observation zero-fill)

| Parameter | Default | Range |
|-----------|---------|-------|
| `d_capture` | 5.0 m | 3–10 m |
| `p_survive` | 0.8 | 0.5–1.0 |

---

## 5) Triangulation System (V5 재활용)

### 5.1 Multi-target 삼각측량

V5의 `midpoint_method_batched`와 `triangulation_covariance_multi_camera`를 타겟별로 반복 호출:

```python
for t in range(T_max):
    if alive[t]:
        # 해당 타겟을 관측 중인 OBSERVE 모드 에이전트만 선택
        valid_cameras = [i for i in range(C) 
                        if role[i] == OBSERVE and bbox_valid[i][t]]
        
        X_tri[t], valid[t] = midpoint_method_batched(
            camera_pos[valid_cameras], ray_dirs[valid_cameras][t])
        
        Sigma_X[t] = triangulation_covariance_multi_camera(
            camera_pos[valid_cameras], ..., X_gt[t])
```

### 5.2 BBoxRayCaster 확장

V5의 BBoxRayCaster는 단일 타겟의 8 corners를 프로젝트한다. 다중 타겟 환경에서는:

- 각 카메라 × 각 alive 타겟에 대해 bbox 계산 → 텐서 차원: `[N_env, C, T_max, 4]`
- 타겟 간 occlusion: 가까운 타겟이 먼 타겟을 가리는 경우 처리 (depth ordering)
- **초기 구현에서는 occlusion 무시** (V5 default와 동일), 향후 추가

### 5.3 Detection-to-Target Association

다중 타겟이 하나의 카메라 FoV 안에 있을 때, 어떤 bbox가 어떤 타겟인지 매칭 필요:

- **시뮬레이션에서는 GT association 사용**: 각 bbox가 어떤 타겟의 projection인지 알고 있음
- 실제 배포 시에는 re-identification / tracking이 필요하지만, 학습 환경에서는 GT로 충분

---

## 6) Delay System 확장

### 6.1 V5 DelaySystemV2 재활용

에이전트 간 통신 delay는 V5 구조 그대로:
- ego path: 빠른 자기 상태
- other path: 느린 통신 경유 아군 상태

### 6.2 타겟 정보 delay

타겟 localization 결과의 공유에도 delay 적용:

| 정보 경로 | Staleness | Latency | Dropout |
|-----------|-----------|---------|---------|
| ego → ego bbox | 50 Hz (onboard detection) | 10 ms | 2% |
| triangulation result → all agents | 25 Hz (fusion rate) | 20–100 ms | 5% |
| target alive status | instant | 0 ms | 0% (critical) |

### 6.3 AoI 확장

V5의 에이전트 간 AoI에 추가하여 **타겟별 AoI**를 유지:

$$\Delta_{\text{loc}}^{(t)} = k_{\text{current}} - k_{\text{last\_valid\_tri}}^{(t)}$$

이 값이 관측 벡터의 per-target `AoI (localization)` 필드로 들어간다.

---

## 7) Reward Function

### 7.1 역할별 reward 분해

총 per-agent reward:

$$r_k^{(i)} = r_{\text{defense}} + r_{\text{role}} + r_{\text{safety}} + r_{\text{action}}$$

### 7.2 Defense reward (전체 팀 공유)

에피소드 레벨의 성과를 step 단위로 분배:

**시설 방어 보상** (모든 에이전트 공유):
$$r_{\text{defense}} = s_{\text{def}} \cdot \Delta t \cdot \sum_{t=1}^{T} \frac{\texttt{alive}^{(t)}}{T} \cdot \frac{d^{(t)}_{\text{fac}}}{d^{(t)}_{\text{fac,init}}}$$

직관: 공격 드론이 시설에서 멀리 있을수록 (또는 격추될수록) 양의 보상.

**요격 성공 보너스** (요격 수행 에이전트):
$$r_{\text{kill}} = s_{\text{kill}} \cdot \mathbb{1}[\text{intercept success at step } k]$$

**시설 침투 페널티** (전체 팀 공유):
$$r_{\text{breach}} = -s_{\text{breach}} \cdot \mathbb{1}[\text{any target reached facility}]$$

### 7.3 Role-dependent reward

#### OBSERVE 모드 ($\texttt{role}^{(i)} = 0$):

V5의 관측 보상을 그대로 사용하되, 관측 대상 타겟에 대해 계산:

$$r_{\text{obs}}^{(i)} = r_{\text{center}}^{(i)} + r_{\text{size}}^{(i)} + r_{\text{tri}}^{(i)}$$

- $r_{\text{center}}$: 타겟 bbox centering (V5 §11.2)
- $r_{\text{size}}$: bbox 크기 shaping (V5 §11.2)
- $r_{\text{tri}}$: 삼각측량 quality — **관측 중인 타겟들의 covariance에 기반** (V5 §11.3.1)

추가 관측 보상:

$$r_{\text{coverage}} = s_{\text{cov}} \cdot \Delta t \cdot \frac{\text{num targets with } \Delta_{\text{loc}} < \tau_{\text{fresh}}}{T_{\text{alive}}}$$

직관: 많은 타겟의 localization이 fresh하게 유지될수록 보상.

#### INTERCEPT 모드 ($\texttt{role}^{(i)} = 1$):

관측 보상 대신 추적/요격 보상:

**접근 보상** (타겟에 가까워질수록):
$$r_{\text{approach}} = s_{\text{app}} \cdot \Delta t \cdot \text{clip}\left(\frac{d_{k-1}^{(i,t)} - d_k^{(i,t)}}{v_{\max} \cdot \Delta t},\ -1,\ 1\right)$$

**Uncertainty-gated intercept**: 요격 전환은 uncertainty가 충분히 낮을 때만 보상:
$$r_{\text{intercept\_ready}} = s_{\text{ready}} \cdot \Delta t \cdot \mathbb{1}\left[\sqrt{\text{tr}(\Sigma_X^{(t)})} < \sigma_{\text{threshold}}\right] \cdot \mathbb{1}[\texttt{role} = \texttt{INTERCEPT}]$$

직관: 충분한 localization 없이 요격 모드로 전환하면 이 보상을 못 받음 → **observe-first 전략 유도**.

### 7.4 Safety reward (V5 확장)

V5의 inter-agent collision + TTC를 유지하되:

$$r_{\text{coll}} = s_{\text{coll}} \cdot \Delta t \cdot \phi_{\text{coll}}^{(\text{agent-agent})} \cdot \text{progress}_{\text{safety}}$$

> **Note**: Interceptor↔attacker 근접은 collision이 아니라 요격 판정 대상. Safety collision은 아군 간 충돌만 해당.

### 7.5 Action penalty (V5 동일)

$$r_{\text{action}} = r_{\text{act}} + r_{\Delta\text{act}}$$

V5 §11.1과 동일. 8차원으로 확장 (role action 포함).

### 7.6 Reward scale 가이드라인

| Term | Scale | Gating | Rationale |
|------|-------|--------|-----------|
| $r_{\text{defense}}$ | 1.0 | always | 핵심 objective |
| $r_{\text{kill}}$ | 10.0 | on intercept | sparse, 큰 보너스 |
| $r_{\text{breach}}$ | -50.0 | on breach | 에피소드 즉시 종료 |
| $r_{\text{tri}}$ | 0.5 | progress_coord | V5 수준 |
| $r_{\text{center}}$ | 0.3 | OBSERVE only | V5 수준 |
| $r_{\text{approach}}$ | 0.5 | INTERCEPT only | 접근 shaping |
| $r_{\text{intercept\_ready}}$ | 0.3 | INTERCEPT + low uncertainty | uncertainty gate |
| $r_{\text{coverage}}$ | 0.3 | OBSERVE only | 다중 타겟 커버리지 |
| $r_{\text{coll}}$ | -2.0 | progress_safety | V5 수준 |
| $r_{\text{action}}$ | -0.01 | always | V5 수준 |

---

## 8) Episode Lifecycle

### 8.1 초기화 (`_reset_idx`)

1. 시설 위치 설정: 환경 origin에 고정
2. 방어 드론 배치: 시설 주변 반경 50–100 m에 랜덤 배치 (V5 formation generator 확장)
3. 공격 wave 1 spawn: spawn_radius에서 랜덤 방위/고도로 배치
4. 모든 방어 드론 초기 역할: OBSERVE
5. 모든 타겟별 AoI: ∞ (미관측 상태)

### 8.2 에피소드 진행

```
매 control step k:
  1. _pre_physics_step:
     - curriculum progress 업데이트
     - action preprocessing (V5 동일)
     - role decision (a_role thresholding)
     - wave spawning 체크 (시간 기반)
  
  2. _apply_action:
     - 방어 드론: V5 force/torque + gimbal (동일)
     - 공격 드론: scripted velocity → acceleration controller
  
  3. _compute_intermediate_values:
     - delay system 업데이트 (V5 동일)
     - 다중 타겟 detection (BBoxRayCaster × T)
     - 다중 타겟 삼각측량 (OBSERVE agents only)
     - 요격 판정 (InterceptResolver)
     - 공격 드론 상태 업데이트 (alive, evasion FSM)
  
  4. _get_rewards: 위 §7 참조
  
  5. _get_observations: 위 §3 참조
  
  6. _check_termination:
     - breach: any target reaches facility → done, penalty
     - all killed: all targets dead → done, bonus
     - timeout: max_steps reached → done
     - all defenders lost: all defender drones destroyed → done, penalty
```

### 8.3 종료 조건

| Condition | done | Reward |
|-----------|------|--------|
| 모든 공격 드론 격추 | True | $+r_{\text{all\_clear}}$ |
| 공격 드론 1대 시설 도달 | True | $-r_{\text{breach}}$ |
| 타임아웃 (30–60 s) | True | 잔존 공격 드론 수 비례 페널티 |
| 모든 방어 드론 손실 | True | $-r_{\text{breach}}$ |

---

## 9) Curriculum

### 9.1 V5 curriculum 호환

V5의 linear progress ramp 구조를 유지:

$$\text{progress}(k; k_s, k_e) = \text{clip}\left(\frac{k - k_s}{k_e - k_s},\ 0,\ 1\right)$$

### 9.2 확장 curriculum 차원

V5의 7개 차원에 5개 추가하여 총 12개 차원:

| # | Dimension | V5 | Schedule | Description |
|---|-----------|-----|----------|-------------|
| 1 | delay/noise | ✓ | 0–100k | V5 동일 |
| 2 | tracking difficulty | ✓ | 0–60k | V5 동일 |
| 3 | triangulation reward | ✓ | 20k–100k | V5 동일 |
| 4 | safety penalties | ✓ | 40k–120k | V5 동일 |
| 5 | moving target speed | ✓ | 20k–100k | V5 동일 → attacker speed로 전환 |
| 6 | dynamics randomization | ✓ | 80k–200k | V5 동일 |
| 7 | zoom curriculum | ✓ | 0–60k | V5 동일 |
| 8 | **num attackers** | NEW | 50k–200k | 1 → T_max |
| 9 | **attacker evasion** | NEW | 100k–250k | 0 → 1 (evasion_agility scale) |
| 10 | **wave complexity** | NEW | 100k–250k | 1 wave → num_waves |
| 11 | **intercept reward enable** | NEW | 60k–150k | 요격 보상 활성화 |
| 12 | **defender loss enable** | NEW | 150k–300k | p_survive < 1 활성화 |

### 9.3 4-Phase 학습 계획

| Phase | Steps | Focus | 주요 curriculum 활성화 |
|-------|-------|-------|----------------------|
| **Phase 1** | 0–60k | 단일 타겟 관측 | V5 ego motion + tracking. 1 attacker, stationary, no evasion |
| **Phase 2** | 40k–120k | 삼각측량 + 이동 타겟 | V5 coordination. Attacker moves toward facility. OBSERVE only |
| **Phase 3** | 80k–200k | 요격 전환 학습 | Intercept reward 활성화. 1–3 attackers, mild evasion |
| **Phase 4** | 150k–350k | 풀 시나리오 | Multi-wave, 5–10 attackers, strong evasion, defender loss, full delay/noise |

---

## 10) Randomization

### 10.1 V5 randomization 유지

- 방어 드론 초기 formation (planar/grid/line)
- gimbal feasibility check
- 물리 파라미터 (mass, drag 등)

### 10.2 추가 randomization

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| num attackers per episode | Uniform(curriculum_min, curriculum_max) | 1–10 |
| attacker behavior profile | Categorical (§4.2.3) | — |
| spawn direction | Uniform | 0°–360° |
| spawn radius | Uniform | 500–1000 m |
| spawn altitude | Uniform | 20–80 m |
| wave timing | Uniform | 5–15 s interval |
| spawn arc width | Uniform | 60°–360° |
| attacker max speed | Uniform per profile | 3–12 m/s |
| evasion agility | Uniform per profile × curriculum | 0–1 |
| facility radius | Fixed (curriculum 초기) → Uniform (후기) | 10–30 m |
| `p_survive` | Fixed 1.0 (초기) → curriculum (후기) | 0.5–1.0 |
| `d_capture` | Fixed | 5 m |

---

## 11) Evaluation Metrics

### 11.1 V5 metrics 유지 (per-target 확장)

| Metric | V5 | 확장 |
|--------|-----|------|
| RMSE | ✓ | 타겟별 평균 |
| Tri Valid | ✓ | 타겟별 평균 |
| Visibility | ✓ | 타겟별 평균 |
| Convergence | ✓ | 타겟별 평균 |
| Collision (agent-agent) | ✓ | 동일 |

### 11.2 새로운 defense metrics

| # | Metric | Unit | Description |
|---|--------|------|-------------|
| 1 | **Defense Success Rate** | % | 에피소드 중 모든 공격 드론 격추 비율 |
| 2 | **Breach Rate** | % | 1대 이상 시설 도달 에피소드 비율 |
| 3 | **Kill Count** | count/ep | 에피소드 당 격추 수 |
| 4 | **Intercept RMSE** | m | 요격 시점에서의 localization 오차 |
| 5 | **Time-to-Intercept** | s | 타겟 spawn → 격추까지 소요 시간 |
| 6 | **Observe-to-Intercept Latency** | steps | 역할 전환 시점에서의 uncertainty 수준 |
| 7 | **Defender Survival Rate** | % | 에피소드 종료 시 살아남은 방어 드론 비율 |
| 8 | **Coverage Freshness** | s | 전체 alive 타겟의 평균 localization AoI |
| 9 | **Role Transition Count** | count/ep | OBSERVE↔INTERCEPT 전환 횟수 |
| 10 | **Uncertainty at Intercept** | m | $\sqrt{\text{tr}(\Sigma_X)}$ at intercept moment |

### 11.3 Hierarchical Task Success (확장)

**Level 0 (Track Maintenance)**: V5 동일 — alive 타겟에 대해 tri_valid ≥ 50%

**Level 1 (Localization Accuracy)**: V5 동일 — valid steps 중 RMSE < 2.0m ≥ 80%

**Level 2 (Defense) — NEW**:
$$\text{pass}_2 = \left(\frac{\text{killed\_targets}}{T} \geq 0.8\right) \land (\text{breach} = 0)$$

**Overall Success**:
$$\text{success} = \text{pass}_0 \land \text{pass}_1 \land \text{pass}_2$$

---

## 12) Training Configuration

### 12.1 Architecture

MAPPO-RNN 유지 (V5에서 검증됨). 주요 변경:

- **Input dim**: 227D (V5: 47D)
- **Hidden size**: 256 (V5: 64) — observation 증가에 비례 확대
- **GRU hidden**: 256 (V5: 64)
- **Action dim**: 8 (V5: 7)
- **Shared policy**: 모든 방어 에이전트가 동일 policy (V5 동일)

### 12.2 Critic (CTDE)

Centralized critic의 shared observation:
- 모든 에이전트의 ego features 연결
- 모든 타겟의 GT 위치/속도 (privileged information)
- Global defense features

Critic obs dim 추정: $\sim$ 28 × 6 + 6 × 10 + 4 = **232D**

### 12.3 학습 파라미터 가이드라인

| Parameter | Value | Note |
|-----------|-------|------|
| num_envs | 2048–4096 | V5 대비 per-env compute 증가 고려 |
| total_timesteps | 300k–500k | Phase 4까지 충분한 학습 |
| sequence_length | 32–64 | 다중 타겟 시 longer context 필요할 수 있음 |
| mini_batch_size | 조정 필요 | obs dim 증가로 GPU memory 고려 |
| episode_max_steps | 750–1500 | 30–60 s @ 25 Hz control rate |
| γ | 0.99 | V5 동일 |
| GAE λ | 0.95 | V5 동일 |

---

## 13) Implementation Roadmap

### 13.1 단계별 구현 계획

| 단계 | 목표 | 구현 내용 | 검증 기준 |
|------|------|----------|----------|
| **M1** | V5 → V6 기반 구조 | Environment class, config dataclass, attacker spawn/remove | 6 defenders + 1 attacker 에피소드 실행 |
| **M2** | Scripted attacker | AttackerManager, behavior FSM, wave generator | 다양한 attacker 프로파일로 에피소드 완주 |
| **M3** | Multi-target detection | BBoxRayCaster 다중 타겟 확장, per-target triangulation | 2+ 타겟 동시 삼각측량 성공 |
| **M4** | Role transition | Action space 8D, role reward, intercept resolver | 1v1에서 observe→intercept→kill 시퀀스 학습 |
| **M5** | Curriculum 통합 | 12-dim curriculum, 4-phase schedule | 6v5에서 defense success rate > 50% |
| **M6** | Full scale | 6v10, multi-wave, domain randomization | 6v10에서 defense success rate > 30% |

### 13.2 V5 모듈 재활용 맵

| V5 모듈 | V6에서의 역할 | 수정 정도 |
|---------|------------|----------|
| `iris_ma_env5.py` | 기반 클래스 또는 fork | **Major** — 에피소드 로직 전면 확장 |
| `point_mass.py` | 방어 드론 controller | 없음 |
| `gimbal_stabilizer.py` | gimbal 제어 | 없음 |
| `multi_agent_delay_system_v2.py` | 에이전트 간 delay | **Minor** — 타겟별 delay 채널 추가 |
| `bbox_raycaster.py` | 타겟 detection | **Medium** — 다중 타겟 루프 |
| `triang_cov_reward_torch.py` | 삼각측량 + covariance | **Minor** — 타겟별 호출 래핑 |
| `safety_manager.py` | 아군 간 충돌 | 없음 |
| `ttc_computer.py` | TTC 계산 | **Minor** — interceptor 예외 처리 |
| `target_movement.py` | **대체됨** | AttackerManager로 교체 |
| `curriculum_cfg.py` | curriculum | **Medium** — 5개 차원 추가 |
| `experiment_registry.py` | 실험 관리 | **Medium** — 새 ablation group 추가 |
| `metric_tracker.py` | 평가 | **Medium** — defense metrics 추가 |

---

## 14) Ablation Study 계획

### 14.1 실험 그룹

| Group | Focus | Variants |
|-------|-------|----------|
| **B1** | Role transition mechanism | `b1_learned_role` (본 설계), `b1_heuristic_role` (uncertainty threshold 고정), `b1_always_observe` (요격 없이 관측만) |
| **B2** | Uncertainty threshold | `b2_sigma_1m`, `b2_sigma_2m`, `b2_sigma_5m`, `b2_no_gate` (gate 없음) |
| **B3** | Attacker difficulty | `b3_easy` (slow, no evasion), `b3_medium` (mixed), `b3_hard` (fast, evasive) |
| **B4** | Agent scaling | `b4_3v2`, `b4_4v3`, `b4_6v5`, `b4_6v10` |
| **B5** | V5 ablations 재검증 | `b5_no_aoi`, `b5_no_delay`, `b5_angular_only` — V5 결론이 defense 시나리오에서도 유지되는지 |

### 14.2 핵심 연구 질문

1. **Policy가 observe→intercept 전환 시점을 uncertainty 기반으로 학습하는가?**
   → B1, B2 비교. Uncertainty at intercept metric으로 검증.

2. **관측 품질이 요격 성공률에 직접적 영향을 미치는가?**
   → B2 결과에서 uncertainty threshold vs defense success rate 상관관계.

3. **V5의 delay-aware / AoI / multi-source covariance 기여가 defense 시나리오에서 증폭되는가?**
   → B5에서 V5 ablation을 defense metric으로 재평가.

4. **수적 열세에서 자원 재활용 (p_survive) 전략이 어떻게 emerge하는가?**
   → B4 agent scaling 결과에서 sequential intercept 행동 분석.

---

## 15) 주요 구현 리스크 및 대응

| Risk | Impact | Mitigation |
|------|--------|------------|
| 다중 타겟 삼각측량 compute 비용 | per-env 속도 저하 | 타겟별 병렬 배치 처리, OBSERVE 에이전트만 계산 |
| 227D observation으로 학습 불안정 | 수렴 실패 | observation normalization, hidden size 증가, curriculum 세밀 조정 |
| Sparse intercept reward | credit assignment 어려움 | approach shaping + intercept_ready 보상으로 dense signal |
| 가변 타겟 수에 대한 generalization | overfitting to specific T | curriculum으로 T 점진 증가 + 에피소드별 랜덤화 |
| Role oscillation (빠른 전환 반복) | 불안정 행동 | role 전환에 hysteresis 또는 cooldown 적용 가능 |
| GPU memory (C=6, T=10, N=4096) | OOM | N=2048로 축소 또는 observation 압축 |

---

## Appendix A: V5 → V6 핵심 변경 요약

| Aspect | V5 | V6 |
|--------|-----|-----|
| Agents | 2–3 observers | 6 observer/interceptors |
| Targets | 1, scripted motion | 1–10, scripted attack FSM |
| Action dim | 7 (velocity + gimbal + zoom) | 8 (+role) |
| Obs dim | 47–62 | ~227 |
| Role | 고정 (OBSERVE) | 동적 (OBSERVE ↔ INTERCEPT) |
| Objective | 삼각측량 uncertainty 최소화 | 시설 방어 (관측 + 요격) |
| Target behavior | linear / circular motion | goal-directed attack + evasion |
| Episode termination | timeout only | breach / all-killed / timeout / all-defenders-lost |
| Curriculum | 7 dimensions | 12 dimensions, 4 phases |
| Key metric | RMSE, tri_valid | Defense success rate, kill count |

## Appendix B: 파일 구조 (예상)

```
iris_ma6/
├── iris_ma_env6.py                    # Main environment (fork from V5)
├── iris_ma_env6_cfg.py                # Config dataclass
├── controller/
│   ├── point_mass.py                  # V5 재활용
│   └── gimbal_stabilizer.py           # V5 재활용
├── delay_system_v2/                   # V5 재활용 + minor extension
│   ├── multi_agent_delay_system_v2.py
│   ├── delay_pipeline.py
│   └── derived_field_computers.py
├── bbox_raycaster/
│   └── bbox_raycaster.py             # V5 확장 (multi-target)
├── triangulation/
│   └── triang_cov_reward_torch.py    # V5 재활용 (per-target wrapper)
├── safety/
│   ├── safety_manager.py             # V5 재활용
│   ├── collision_detector.py         # V5 재활용
│   └── ttc_computer.py              # V5 minor extension
├── attacker/                          # NEW
│   ├── attacker_manager.py
│   ├── attacker_behavior_cfg.py
│   ├── attacker_wave_generator.py
│   ├── attacker_controller.py
│   └── intercept_resolver.py
├── curriculum/
│   └── curriculum_cfg.py             # V5 확장 (12 dims)
├── randomization/
│   ├── randomizer.py                 # V5 확장
│   └── distance_based_generator.py   # V5 재활용
└── experiments/
    ├── experiment_registry.py        # B1–B5 실험 추가
    ├── experiment_cfg.py
    ├── evaluate.py                   # defense metrics 추가
    └── metrics/
        ├── metric_tracker.py         # V5 확장
        ├── timeseries_tracker.py     # V5 확장
        └── trajectory_recorder.py    # V5 확장
```
