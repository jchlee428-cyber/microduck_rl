# MicroDuck RL Lab 🦆🤖

강화학습(Reinforcement Learning) 초보자를 위한 로컬 실행 및 실험 환경입니다.
Gymnasium과 Stable-Baselines3, MuJoCo 물리 엔진을 기반으로 에이전트가 가상 환경에서 학습하는 전체 파이프라인(환경 점검 -> CartPole 기초 학습 -> MuJoCo 로봇 연속 제어 확장 -> 시각화 및 분석)을 제공합니다.

---

## 1. 프로젝트 구조

```text
microduck_rl/
├── microduck.xml           # 시그니처 2족 보행 오리 로봇 MuJoCo MJCF 물리 모델
├── microduck_env.py        # MicroDuck-v1 커스텀 Gymnasium MuJoCo 환경 (Waddle-Walk)
├── check_environment.py    # 환경 및 의존성 라이브러리 검증 스크립트
├── mujoco_compat.py        # Windows 한글 경로 MuJoCo C-runtime 호환 패치
├── launcher.py             # 마스터 대화형 콘솔 런처
├── train_cartpole.py       # CartPole-v1 PPO 학습 스크립트
├── evaluate_cartpole.py    # CartPole-v1 학습 모델 평가 및 비교 스크립트
├── train_mujoco.py         # MuJoCo 로봇 및 MicroDuck PPO 학습 스크립트
├── evaluate_mujoco.py      # MuJoCo 로봇 및 MicroDuck 평가 및 비교 스크립트
├── plot_results.py         # 8패널 통합 학습 결과 곡선 시각화 스크립트
├── visualize_agent.py      # 로봇 동작 화면 렌더링 및 GIF 애니메이션 녹화
├── requirements.txt        # 필요 라이브러리 목록
├── models/                 # 학습 완료된 모델 가중치 (.zip)
├── logs/                   # TensorBoard 로그 및 평가 데이터 (.npz)
└── videos/                 # 에이전트 동작 렌더링 결과 (.gif)
```

---

## 2. 개발 및 실행 환경

* **OS**: Windows 10 / 11
* **Python**: 3.10.11 (PyTorch, MuJoCo 호환 최적화)
* **RL Framework**: Gymnasium 1.3.0
* **Simulator**: MuJoCo 3.13.0
* **Algorithm**: PPO (Proximal Policy Optimization)
* **Library**: Stable-Baselines3 2.9.0, PyTorch 2.14.0

---

## 3. 설치 및 실행 가이드

### 0) 원클릭 통합 대화형 런처 및 웹 대시보드 (가장 추천! 👍)
```powershell
# 1. 인터랙티브 웹 대시보드 실행 (Streamlit, 브라우저 자동 오픈: localhost:8501)
python launch_dashboard.py

# 2. 콘솔 마스터 제어 런처 실행 (또는 run.bat 더블클릭)
python launcher.py
```
> [!TIP]
> `launch_dashboard.py`를 실행하면 세련된 다크 테마의 **웹 컨트롤 센터**가 브라우저에 열립니다. 7대 전체 로봇의 동작 비교 애니메이션(GIF), 8패널 종합 학습 곡선, 실시간 정책 인퍼런스 및 스텝별 텔레메트리 그래프, 네이티브 3D 뷰어 원클릭 런처를 브라우저에서 모두 조작할 수 있습니다.

### 1) 가상환경 활성화 (선택 사항)
```powershell
.\.venv\Scripts\Activate.ps1
```
*(가상환경 활성화를 누락하더라도 스크립트가 자동으로 `.venv`로 리디렉션하므로 걱정 없이 실행 가능합니다.)*

### 2) 환경 및 의존성 점검
```powershell
python check_environment.py
```

### 3) 1단계: CartPole-v1 학습 및 평가
```powershell
# 학습 (20,000 timesteps, 약 30초 소요)
python train_cartpole.py

# 성능 평가 (Random Baseline과 비교)
python evaluate_cartpole.py --episodes 5
```

### 4) 2단계: MuJoCo Hopper-v5 학습 및 평가
```powershell
# 학습 (50,000 timesteps, 약 1.2분 소요)
python train_mujoco.py --timesteps 50000

# 성능 평가 (Random Baseline과 비교)
python evaluate_mujoco.py --episodes 5
```

### 5) 3단계: 결과 시각화 및 동영상 생성
```powershell
# 학습 곡선 차트 생성 (logs/learning_curves.png)
python plot_results.py

# 학습 전/후 비교 애니메이션 GIF 생성 (videos/hopper_comparison.gif)
python visualize_agent.py

# 실시간 시뮬레이션 창 띄우기 (GUI 환경)
python visualize_agent.py --human
```

---

### 5) 3단계: 커스텀 오리 로봇 (MicroDuck-v1) 학습 및 시각화
```powershell
# MicroDuck-v1 학습 (60,000 timesteps)
python train_mujoco.py --env MicroDuck-v1 --timesteps 60000

# 성능 평가 (Random 대비 +177% 향상)
python evaluate_mujoco.py --env MicroDuck-v1 --episodes 5

# 실시간 3D 오리 뷰어 실행
python visualize_agent.py --env MicroDuck-v1 --human --loop

# 비교 애니메이션 녹화 (videos/microduck_comparison.gif)
python visualize_agent.py --env MicroDuck-v1
```

---

## 4. 실험 결과 요약

### 0) 🦆 MicroDuck-v1 (자체 제작 시그니처 2족 보행 오리 로봇)
* **학습 전 (Random)**: 평균 보상 `17.17`점 (평균 19스텝만에 뒤뚱거리다 전도)
* **학습 후 (PPO 60k)**: 평균 보상 `47.65`점 (최고 64.46점, **+177.5% 향상**, 물갈퀴 발을 이용한 전진 보행 습득)

### 1) Gymnasium CartPole-v1
* **학습 전 (Random)**: 평균 보상 `16.0`점 (약 15스텝 내로 막대가 쓰러짐)
* **학습 후 (PPO)**: 평균 보상 `500.0`점 (**최대 만점 달성**, +3,025% 향상)

### 2) MuJoCo Hopper-v5 (단각 로봇 균형 및 도약)
* **학습 전 (Random)**: 평균 보상 `18.04`점 (평균 23스텝만에 전도)
* **학습 후 (PPO 50k)**: 평균 보상 `400.98`점 (평균 140스텝 동안 전진 도약 유지, +2,122% 향상)

### 3) MuJoCo Walker2d-v5 (2족 보행 로봇 전진 보행)
* **학습 전 (Random)**: 평균 보상 `-0.12`점 (평균 19스텝만에 전도)
* **학습 후 (PPO 50k)**: 평균 보상 `377.30`점 (평균 191스텝 동안 두 다리로 전진 보행, +377점 향상)

### 4) MuJoCo Ant-v5 (4족 보행 로봇 다관절 보행)
* **학습 전 (Random)**: 평균 보상 `-21.39`점 (다리가 꼬이며 전도)
* **학습 후 (PPO 50k)**: 평균 보상 `93.94`점 (최고 168.67점, 4개 다리로 전진 기어가기 습득, +115점 향상)

### 5) MuJoCo Humanoid-v5 (17개 관절 휴머노이드 로봇)
* **학습 전 (Random)**: 평균 보상 `122.9`점 (평균 19스텝만에 전도)
* **학습 후 (PPO 50k)**: 평균 보상 `363.0`점 (평균 54스텝 이상 균형 유지, +240.1점 향상)

---

## 5. 기술적 특이사항 및 Windows 호환성 패치
1. **Windows 한글 경로 호환성**:
   Windows 작업 경로에 한글(`바탕 화면` 등)이 포함된 경우, MuJoCo C 엔진의 `mj_loadXML`이 경로를 파싱하지 못하는 문제를 `mujoco_compat.py`를 통해 Python UTF-8 파서와 `mujoco.MjModel.from_xml_string`으로 우회 해결했습니다.
2. **뷰어 마우스 카메라 제어 호환성**:
   Gymnasium의 `WindowViewer`가 최신 MuJoCo C 바인딩의 `mjv_moveCamera` 파라미터 개수와 불일치하여 발생하던 마우스 드래그 충돌을 `mujoco_compat.py`에서 래핑하여 마우스 회전 및 줌이 원활하도록 개선했습니다.
3. **가상환경 자동 부트스트랩**:
   가상환경 활성화를 누락하고 전역 Python으로 실행하더라도 `_bootstrap.py`가 자동으로 `.venv` 파이썬으로 리디렉션 실행합니다.
