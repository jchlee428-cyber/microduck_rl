"""
MicroDuck RL Lab - Interactive Web Dashboard (Streamlit Cloud Entrypoint)
========================================================================
Interactive web control center for observing, evaluating, and managing
all trained reinforcement learning robots (MicroDuck, MuJoCo suite, CartPole).
"""
try:
    import _bootstrap
except Exception:
    pass

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="MicroDuck RL Lab - AI Robot Control Center",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Futuristic Glassmorphism Theme */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFB703 0%, #FB8500 50%, #023047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #8E9AAF;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .robot-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #023047;
        color: #8ECAE6;
        border: 1px solid #219EBC;
    }
    .status-ok {
        color: #2EC4B6;
        font-weight: 600;
    }
    .status-warn {
        color: #E71D36;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def safe_image(path, caption=None):
    """Helper to render image with fallback for Streamlit version compatibility."""
    try:
        st.image(path, caption=caption, width="stretch")
    except Exception:
        st.image(path, caption=caption, use_container_width=True)

# ==========================================
# 2. Robot Specification Metadata
# ==========================================
ROBOT_CONFIGS = {
    "MicroDuck-v1": {
        "name": "MicroDuck-v1",
        "label": "🦆 MicroDuck-v1 (Signature Custom Duck)",
        "tag": "Featured Custom Biped Robot",
        "obs_dim": "23 dims (Pose, Rot, Velocity)",
        "action_dim": "6 Continuous Torques",
        "joints": "6 (Hip, Knee, Ankle x 2)",
        "model_file": "models/microduck_ppo.zip",
        "vec_norm": "models/microduck_vec_normalize.pkl",
        "gif_file": "videos/microduck_comparison.gif",
        "benchmark_reward": "47.65 pts (+177.5%)",
        "description": "자체 제작한 시그니처 2족 보행 오리 로봇입니다. 노란 몸통과 넓은 물갈퀴 발, 6개 관절 모터로 구성되어 전진 Waddle-Walk를 학습했습니다."
    },
    "Hopper-v5": {
        "name": "Hopper-v5",
        "label": "🦿 Hopper-v5 (1-Leg Hopping Robot)",
        "tag": "Gymnasium MuJoCo Benchmark",
        "obs_dim": "11 dims",
        "action_dim": "3 Continuous Torques",
        "joints": "3 (Thigh, Leg, Foot)",
        "model_file": "models/hopper_ppo.zip",
        "vec_norm": "models/hopper_vec_normalize.pkl",
        "gif_file": "videos/hopper_comparison.gif",
        "benchmark_reward": "400.98 pts (+2,122%)",
        "description": "외다리로 균형을 잡으며 앞으로 힘차게 도약하는 로봇입니다. 허벅지, 종아리, 발목 3개 모터로 지속적인 단각 점핑을 수행합니다."
    },
    "Walker2d-v5": {
        "name": "Walker2d-v5",
        "label": "🏃 Walker2d-v5 (2-Leg Bipedal Walking)",
        "tag": "Gymnasium MuJoCo Benchmark",
        "obs_dim": "17 dims",
        "action_dim": "6 Continuous Torques",
        "joints": "6 (Hip, Knee, Ankle x 2)",
        "model_file": "models/walker2d_ppo.zip",
        "vec_norm": "models/walker2d_vec_normalize.pkl",
        "gif_file": "videos/walker2d_comparison.gif",
        "benchmark_reward": "377.30 pts (안정 직진 보행)",
        "description": "인간의 하체를 모방한 2족 보행 로봇입니다. 두 다리를 번갈아 내딛으며 넘어지지 않고 안정적인 보폭으로 전진합니다."
    },
    "Ant-v5": {
        "name": "Ant-v5",
        "label": "🐜 Ant-v5 (4-Leg Quadruped Crawling)",
        "tag": "Gymnasium MuJoCo Benchmark",
        "obs_dim": "27 dims",
        "action_dim": "8 Continuous Torques",
        "joints": "8 (Hip, Ankle x 4 legs)",
        "model_file": "models/ant_ppo.zip",
        "vec_norm": "models/ant_vec_normalize.pkl",
        "gif_file": "videos/ant_comparison.gif",
        "benchmark_reward": "454.50 pts (1,000스텝 완주)",
        "description": "4개의 다리로 기어다니는 다족 보행 로봇입니다. 8개 관절을 유기적으로 협응하여 1,000스텝 동안 전복 없이 전진합니다."
    },
    "HalfCheetah-v5": {
        "name": "HalfCheetah-v5",
        "label": "🐆 HalfCheetah-v5 (Forward Sprint)",
        "tag": "Gymnasium MuJoCo Benchmark",
        "obs_dim": "17 dims",
        "action_dim": "6 Continuous Torques",
        "joints": "6 (Front/Back legs)",
        "model_file": "models/halfcheetah_ppo.zip",
        "vec_norm": "models/halfcheetah_vec_normalize.pkl",
        "gif_file": "videos/halfcheetah_comparison.gif",
        "benchmark_reward": "Sprint Control Trained",
        "description": "치타의 폭발적인 질주 모션을 학습하는 2D 4족 로봇입니다. 넘어짐 종료 없이 최고 전진 속도를 극대화합니다."
    },
    "Humanoid-v5": {
        "name": "Humanoid-v5",
        "label": "🦾 Humanoid-v5 (Full-Body Humanoid)",
        "tag": "Gymnasium MuJoCo SOTA Robot",
        "obs_dim": "376 dims",
        "action_dim": "17 Continuous Torques",
        "joints": "17 (전신 17개 관절 모터)",
        "model_file": "models/humanoid_ppo.zip",
        "vec_norm": "models/humanoid_vec_normalize.pkl",
        "gif_file": "videos/humanoid_comparison.gif",
        "benchmark_reward": "363.00 pts (직립 균형 유지)",
        "description": "가장 복잡한 17개 관절 전신 휴머노이드 로봇입니다. 376차원 고차원 관측 상태에서 직립 균형을 유지합니다."
    },
    "CartPole-v1": {
        "name": "CartPole-v1",
        "label": "🏋️ CartPole-v1 (Inverted Pendulum)",
        "tag": "Classic Control Foundation",
        "obs_dim": "4 dims",
        "action_dim": "Discrete(2) [Left, Right]",
        "joints": "1 Passive Cart-Pole Joint",
        "model_file": "models/cartpole_ppo.zip",
        "vec_norm": None,
        "gif_file": "videos/cartpole_comparison.gif",
        "benchmark_reward": "500.00 pts (만점 달성)",
        "description": "카트 위의 역진자 막대를 중심을 잡으며 좌우로 제어하는 고전 강화학습 대표 벤치마크입니다."
    }
}

# ==========================================
# 3. Header & Global Metrics
# ==========================================
st.markdown('<div class="main-title">🦆 MicroDuck RL Lab: AI Robot Control Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Gymnasium & MuJoCo 3.13 기반의 심층 강화학습(PPO) 로봇 제어 및 실시간 텔레메트리 웹 대시보드</div>', unsafe_allow_html=True)

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Total Active Robots", "7 Environments", "1 Custom + 6 Benchmarks")
with col_kpi2:
    st.metric("RL Algorithm", "PPO (Clip 0.2)", "Stable-Baselines3")
with col_kpi3:
    st.metric("Physics Simulation", "MuJoCo 3.13", "RK4 500Hz Integrator")
with col_kpi4:
    st.metric("Signature Robot", "MicroDuck-v1", "+177.5% Reward Boost")

st.markdown("---")

# ==========================================
# 4. Sidebar: Robot Selector & Hardware Inspector
# ==========================================
st.sidebar.markdown("### 🤖 Robot Selector")
selected_key = st.sidebar.selectbox(
    "관찰할 로봇을 선택하세요:",
    list(ROBOT_CONFIGS.keys()),
    format_func=lambda k: ROBOT_CONFIGS[k]["label"]
)

cfg = ROBOT_CONFIGS[selected_key]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Robot Specifications")
st.sidebar.markdown(f"**태그:** `{cfg['tag']}`")
st.sidebar.markdown(f"**관측 공간:** {cfg['obs_dim']}")
st.sidebar.markdown(f"**행동 공간:** {cfg['action_dim']}")
st.sidebar.markdown(f"**관절 구조:** {cfg['joints']}")
st.sidebar.markdown(f"**대표 성과:** **{cfg['benchmark_reward']}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Model Weights Status")
has_model = os.path.exists(cfg["model_file"])
if has_model:
    st.sidebar.markdown(f'<span class="status-ok">✔ Model Ready:</span> `{cfg["model_file"]}`', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<span class="status-warn">✖ Missing:</span> `{cfg["model_file"]}`', unsafe_allow_html=True)

if cfg["vec_norm"]:
    has_norm = os.path.exists(cfg["vec_norm"])
    if has_norm:
        st.sidebar.markdown(f'<span class="status-ok">✔ Normalizer Ready:</span> `{cfg["vec_norm"]}`', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<span class="status-warn">✖ Missing Normalizer</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Native 3D Simulation")
if st.sidebar.button("▶ Open Native 3D Viewer Window"):
    cmd = [sys.executable, "visualize_agent.py", "--env", cfg["name"], "--human", "--loop"]
    subprocess.Popen(cmd)
    st.sidebar.success(f"MuJoCo 3D 창 실행 명령 전송: {cfg['name']}")

# ==========================================
# 5. Main Dashboard Tabs
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Visual Showcase (애니메이션)",
    "📊 Performance Analytics (학습 곡선)",
    "🧪 Live Telemetry (실시간 평가)",
    "⚡ Interactive Training (브라우저 실시간 학습)",
    "📈 TensorBoard & System Info"
])

# ----------------------------------------------------
# Tab 1: Visual Showcase
# ----------------------------------------------------
with tab1:
    st.markdown(f"### 🎬 {cfg['label']} - Visual Motion Showcase")
    st.info(cfg["description"])

    col_vid, col_info = st.columns([1.5, 1])

    with col_vid:
        gif_path = cfg["gif_file"]
        if os.path.exists(gif_path):
            safe_image(gif_path, caption=f"[좌] 훈련 전 Random Baseline vs [우] 훈련 후 PPO Agent ({cfg['name']})")
        else:
            st.warning(f"애니메이션 파일이 아직 없습니다: {gif_path}")
            if st.button(f"🎥 {cfg['name']} 비교 애니메이션 지금 렌더링하기"):
                with st.spinner("애니메이션 녹화 중... (약 10~20초 소요)"):
                    cmd = [sys.executable, "visualize_agent.py", "--env", cfg["name"]]
                    subprocess.run(cmd)
                st.success("애니메이션 생성 완료! 페이지를 새로고침하면 표시됩니다.")
                st.rerun()

    with col_info:
        st.markdown("#### 🔬 물리 시뮬레이션 관찰 포인트")
        if cfg["name"] == "MicroDuck-v1":
            st.markdown("""
            * **Waddle-Walk 보행 메커니즘**:
              - 노란 몸통의 무게중심을 낮게 유지하며 두 개의 물갈퀴 발을 번갈아 디딤.
              - 훈련 전(랜덤)에는 19스텝만에 균형을 잃고 꽈당 넘어지지만, PPO 에이전트는 직진 토크를 분배하여 전진.
            * **생체 모방형 제어**:
              - 발목(Ankle)과 무릎(Knee) 모터의 부드러운 완충 동작.
            """)
        elif cfg["name"] == "Hopper-v5":
            st.markdown("""
            * **단각 연속 도약 (Single-Leg Hopping)**:
              - 발바닥 접지 시 무릎 관절을 굽혀 충격을 흡수하고, 신전 모멘텀으로 전방 도약.
              - 상체(Torso) 피치 각도를 직립 유지하여 전복 방지.
            """)
        elif cfg["name"] == "Walker2d-v5":
            st.markdown("""
            * **2족 교차 보행 (Bipedal Locomotion)**:
              - 좌우 다리가 180도 위상차를 가지고 교차 스윙.
              - 발바닥 충돌 반력을 이용해 추진력 생성.
            """)
        elif cfg["name"] == "Ant-v5":
            st.markdown("""
            * **4족 트로팅 보행 (Quadruped Crawling)**:
              - 대각선 다리 쌍(Diagonal Gait)을 번갈아 짚으며 안정적인 4족 이동.
              - 1,000스텝 완주 달성.
            """)
        elif cfg["name"] == "Humanoid-v5":
            st.markdown("""
            * **전신 17개 관절 협응 제어**:
              - 양팔, 몸통, 골반, 다리의 복합 자유도를 동시에 조율.
              - 376차원의 방대한 상태 공간에서 넘어지지 않고 직립 유지.
            """)
        else:
            st.markdown("""
            * **고전 제어 밸런싱**:
              - 막대의 각도와 각속도 오차를 즉각적으로 보상하는 반대 방향 카트 이동.
              - 최대 만점 500점 완벽 달성.
            """)

# ----------------------------------------------------
# Tab 2: Performance Analytics
# ----------------------------------------------------
with tab2:
    st.markdown("### 📊 Complete Benchmark Performance Analytics")

    # Interactive Robot Learning Curve from evaluations.npz
    robot_name = cfg["name"].split("-")[0].lower()
    eval_npz = os.path.join("logs", robot_name, "evaluations.npz")
    if os.path.exists(eval_npz):
        try:
            data = np.load(eval_npz)
            ts = data["timesteps"]
            res = data["results"]
            mean_r = np.mean(res, axis=1)
            df_curve = pd.DataFrame({
                "Timesteps": ts,
                f"{cfg['name']} PPO Mean Reward": mean_r
            }).set_index("Timesteps")
            st.markdown(f"#### 📈 {cfg['name']} 인터랙티브 학습 곡선 (Timesteps vs Reward)")
            st.line_chart(df_curve, color="#FFB703")
        except Exception:
            pass

    st.markdown("#### 🖼️ 8-패널 종합 학습 곡선 대시보드")
    chart_path = "logs/learning_curves.png"
    if os.path.exists(chart_path):
        safe_image(chart_path, caption="MicroDuck RL Lab - Complete 8-Panel Reinforcement Learning Curves")
    else:
        st.warning("학습 곡선 차트가 없습니다. 생성 중...")
        subprocess.run([sys.executable, "plot_results.py"])
        st.rerun()

    st.markdown("#### 🏆 전체 로봇 정량 벤치마크 비교표")
    benchmark_data = [
        {"Robot": "MicroDuck-v1", "Type": "Custom Biped", "Random Mean": 17.17, "PPO Mean": 47.65, "Improvement": "+177.5%", "Status": "★ Signature Robot"},
        {"Robot": "CartPole-v1", "Type": "Classic Control", "Random Mean": 16.00, "PPO Mean": 500.00, "Improvement": "+3,025.0%", "Status": "✔ Perfect Score"},
        {"Robot": "Hopper-v5", "Type": "1-Leg Hopping", "Random Mean": 18.04, "PPO Mean": 400.98, "Improvement": "+2,122.7%", "Status": "✔ Stable Hop"},
        {"Robot": "Walker2d-v5", "Type": "2-Leg Biped", "Random Mean": -0.12, "PPO Mean": 377.30, "Improvement": "+377.4 pts", "Status": "✔ Biped Walk"},
        {"Robot": "Ant-v5", "Type": "4-Leg Crawl", "Random Mean": -21.39, "PPO Mean": 454.50, "Improvement": "+475.8 pts", "Status": "✔ 1,000 Steps Finished"},
        {"Robot": "HalfCheetah-v5", "Type": "Cheetah Sprint", "Random Mean": -5.20, "PPO Mean": 240.00, "Improvement": "+245.2 pts", "Status": "✔ High-Speed Sprint"},
        {"Robot": "Humanoid-v5", "Type": "Full-Body 17J", "Random Mean": 122.90, "PPO Mean": 363.00, "Improvement": "+195.3%", "Status": "✔ Full-Body Balance"}
    ]
    df_bench = pd.DataFrame(benchmark_data)
    st.dataframe(df_bench, hide_index=True)

    st.markdown("#### 📈 PPO 평가 보상 비교 차트")
    chart_df = pd.DataFrame({
        "Robot": [b["Robot"] for b in benchmark_data],
        "PPO Reward": [b["PPO Mean"] for b in benchmark_data]
    }).set_index("Robot")
    st.bar_chart(chart_df, color="#FFB703")

# ----------------------------------------------------
# Tab 3: Live Telemetry & Evaluation
# ----------------------------------------------------
with tab3:
    st.markdown(f"### 🧪 Live Telemetry & Policy Rollout: {cfg['name']}")
    st.caption("웹 환경에서 즉시 에피소드 롤아웃을 실행하고, 스텝별 보상 및 생존 궤적을 실시간으로 분석합니다.")

    eval_col1, eval_col2 = st.columns([1, 3])
    with eval_col1:
        num_test_episodes = st.slider("평가 에피소드 수:", min_value=1, max_value=10, value=3)
        deterministic = st.checkbox("Deterministic Policy", value=True)
        start_eval = st.button("🚀 실시간 인퍼런스 롤아웃 시작", type="primary")

    with eval_col2:
        if start_eval:
            if not os.path.exists(cfg["model_file"]):
                st.error(f"모델 파일을 찾을 수 없습니다: {cfg['model_file']}")
            else:
                with st.spinner("RL 모델 및 물리 시뮬레이션 환경 로드 중..."):
                    import mujoco_compat
                    import microduck_env
                    import gymnasium as gym
                    from stable_baselines3 import PPO
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                progress_bar = st.progress(0)
                status_text = st.empty()

                def _make_eval_env():
                    return gym.make(cfg["name"])

                eval_vec = DummyVecEnv([_make_eval_env])
                if cfg["vec_norm"] and os.path.exists(cfg["vec_norm"]):
                    eval_vec = VecNormalize.load(cfg["vec_norm"], eval_vec)
                    eval_vec.training = False

                model = PPO.load(cfg["model_file"])

                ep_rewards = []
                ep_lengths = []
                all_step_rewards = []

                for ep in range(num_test_episodes):
                    status_text.text(f"에피소드 {ep + 1}/{num_test_episodes} 실행 중...")
                    obs = eval_vec.reset()
                    done = False
                    total_r = 0.0
                    steps = 0
                    step_rewards = []

                    while not done:
                        action, _ = model.predict(obs, deterministic=deterministic)
                        obs, reward, dones, info = eval_vec.step(action)
                        r = float(reward[0])
                        total_r += r
                        step_rewards.append(r)
                        steps += 1
                        done = bool(dones[0])
                        if steps >= 1000:
                            break

                    ep_rewards.append(total_r)
                    ep_lengths.append(steps)
                    all_step_rewards.append(step_rewards)
                    progress_bar.progress((ep + 1) / num_test_episodes)

                eval_vec.close()
                status_text.text("인퍼런스 완료!")

                res_c1, res_c2, res_c3 = st.columns(3)
                with res_c1:
                    st.metric("평균 에피소드 보상", f"{np.mean(ep_rewards):.2f} pts")
                with res_c2:
                    st.metric("최고 에피소드 보상", f"{np.max(ep_rewards):.2f} pts")
                with res_c3:
                    st.metric("평균 생존 스텝수", f"{np.mean(ep_lengths):.1f} steps")

                st.markdown("#### 📉 스텝별 실시간 보상 궤적 (Step Rewards)")
                max_len = max(len(sr) for sr in all_step_rewards)
                plot_data = {}
                for idx, sr in enumerate(all_step_rewards):
                    padded = sr + [np.nan] * (max_len - len(sr))
                    plot_data[f"Episode {idx+1}"] = padded
                df_steps = pd.DataFrame(plot_data)
                st.line_chart(df_steps)

# ----------------------------------------------------
# Tab 4: Interactive Browser Training Lab
# ----------------------------------------------------
with tab4:
    st.markdown("### ⚡ Interactive Browser Training Lab (브라우저 실시간 강화학습)")
    st.markdown("""
    설치나 복잡한 명령어 없이, **웹 브라우저에서 직접 PPO 하이퍼파라미터를 조절하고 실시간으로 에이전트를 학습**시킵니다!  
    학습이 진행됨에 따라 에피소드 보상이 상승하는 과정을 실시간 차트로 관찰해 보세요.
    """)

    col_ctrl, col_mon = st.columns([1, 1.6])

    with col_ctrl:
        st.markdown("#### ⚙️ 훈련 하이퍼파라미터 설정")
        train_env_choice = st.selectbox(
            "학습 대상 로봇/환경:",
            [
                "CartPole-v1 (초고속 1~3초 완성 · 교육 추천 🎯)",
                "MicroDuck-v1 (시그니처 2족 보행 오리 🦆)",
                "Hopper-v5 (MuJoCo 1족 호핑 🦿)",
                "Walker2d-v5 (MuJoCo 2족 보행 🏃)"
            ],
            index=0
        )

        env_id_map = {
            "CartPole-v1 (초고속 1~3초 완성 · 교육 추천 🎯)": "CartPole-v1",
            "MicroDuck-v1 (시그니처 2족 보행 오리 🦆)": "MicroDuck-v1",
            "Hopper-v5 (MuJoCo 1족 호핑 🦿)": "Hopper-v5",
            "Walker2d-v5 (MuJoCo 2족 보행 🏃)": "Walker2d-v5"
        }
        selected_train_env = env_id_map[train_env_choice]

        # 기본 스텝 수 추천
        default_steps = 3000 if selected_train_env == "CartPole-v1" else 2048
        max_steps = 10000 if selected_train_env == "CartPole-v1" else 6144
        step_increment = 1000 if selected_train_env == "CartPole-v1" else 2048

        train_timesteps = st.slider(
            "학습 타임스텝 수 (Total Timesteps):",
            min_value=1000 if selected_train_env == "CartPole-v1" else 2048,
            max_value=max_steps,
            value=default_steps,
            step=step_increment,
            help="클라우드 환경의 자원을 고려하여 브라우저에서 1~15초 내에 완료되는 범위로 최적화되었습니다."
        )

        lr_select = st.select_slider(
            "학습률 (Learning Rate):",
            options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
            value=3e-4,
            format_func=lambda x: f"{x:.0e}"
        )

        n_epochs = st.slider("PPO 에포크 수 (n_epochs):", min_value=3, max_value=10, value=5)

        start_train = st.button("🚀 브라우저에서 즉시 학습 시작", type="primary", use_container_width=True)

    with col_mon:
        st.markdown("#### 📡 실시간 훈련 모니터링")
        status_box = st.empty()
        prog_bar = st.progress(0)
        chart_box = st.empty()

        if not start_train:
            status_box.info("👈 좌측에서 하이퍼파라미터를 설정한 후 **[브라우저에서 즉시 학습 시작]** 버튼을 누르세요.")
            chart_box.markdown(
                """
                <div style="border: 2px dashed rgba(255,255,255,0.15); border-radius: 10px; padding: 40px; text-align: center; color: #8E9AAF;">
                    📈 학습이 시작되면 실시간 에피소드 보상 곡선이 이곳에 그려집니다.
                </div>
                """, unsafe_allow_html=True
            )
        else:
            try:
                import io
                import mujoco_compat
                import microduck_env
                import gymnasium as gym
                from stable_baselines3 import PPO
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.callbacks import BaseCallback

                status_box.info(f"⏳ `{selected_train_env}` 물리 시뮬레이션 환경 초기화 중...")

                # 1. 훈련 전 Random Baseline 빠른 평가
                raw_test_env = gym.make(selected_train_env)
                pre_rewards = []
                for _ in range(3):
                    o, _ = raw_test_env.reset()
                    d = False
                    ep_r = 0
                    while not d:
                        a = raw_test_env.action_space.sample()
                        o, r, term, trunc, _ = raw_test_env.step(a)
                        ep_r += float(r)
                        d = term or trunc
                    pre_rewards.append(ep_r)
                raw_test_env.close()
                pre_mean = float(np.mean(pre_rewards))

                # 2. Callback 정의
                class StreamlitCallback(BaseCallback):
                    def __init__(self, total_steps, p_bar, s_box, c_box):
                        super().__init__()
                        self.total_steps = total_steps
                        self.p_bar = p_bar
                        self.s_box = s_box
                        self.c_box = c_box
                        self.records = []
                        self.last_ep_len = 0
                        self.start_t = time.time()

                    def _on_step(self) -> bool:
                        buf = self.model.ep_info_buffer
                        if buf and len(buf) > self.last_ep_len:
                            self.last_ep_len = len(buf)
                            recent_r = [x["r"] for x in buf]
                            mean_rew = float(np.mean(recent_r[-5:]))
                            self.records.append({
                                "Timestep": self.num_timesteps,
                                "Mean Reward": mean_rew
                            })
                            df_hist = pd.DataFrame(self.records).set_index("Timestep")
                            self.c_box.line_chart(df_hist, color="#2EC4B6")

                        frac = min(1.0, float(self.num_timesteps) / float(self.total_steps))
                        self.p_bar.progress(frac)
                        elapsed = time.time() - self.start_t
                        fps = int(self.num_timesteps / max(0.1, elapsed))
                        self.s_box.markdown(
                            f"**🏃 학습 중:** `{self.num_timesteps:,} / {self.total_steps:,} steps` "
                            f"({int(frac*100)}%) | 완료 에피소드: `{self.last_ep_len}`개 | 속도: `{fps}` FPS"
                        )
                        return True

                # 3. 환경 생성
                train_env = Monitor(gym.make(selected_train_env))

                n_steps = 128 if selected_train_env == "CartPole-v1" else 512
                batch_size = 64

                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=lr_select,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=0.99,
                    verbose=0
                )

                cb = StreamlitCallback(train_timesteps, prog_bar, status_box, chart_box)
                start_time = time.time()
                model.learn(total_timesteps=train_timesteps, callback=cb)
                train_time = time.time() - start_time
                train_env.close()

                # 4. 훈련 후 즉시 평가
                eval_env = gym.make(selected_train_env)
                post_rewards = []
                for _ in range(3):
                    o, _ = eval_env.reset()
                    d = False
                    ep_r = 0
                    st_cnt = 0
                    while not d and st_cnt < 1000:
                        a, _ = model.predict(o, deterministic=True)
                        o, r, term, trunc, _ = eval_env.step(a)
                        ep_r += float(r)
                        st_cnt += 1
                        d = term or trunc
                    post_rewards.append(ep_r)
                eval_env.close()
                post_mean = float(np.mean(post_rewards))

                prog_bar.progress(1.0)
                status_box.success(f"🎉 브라우저 PPO 학습 완료! (총 소요 시간: {train_time:.1f}초)")

                # 5. 결과 비교 표시
                st.markdown("#### 🏆 학습 전후 성과 비교 (Before vs After)")
                m1, m2, m3 = st.columns(3)
                m1.metric("훈련 전 (랜덤 정책)", f"{pre_mean:.1f} pts")
                delta_pts = post_mean - pre_mean
                m2.metric("훈련 후 (브라우저 PPO)", f"{post_mean:.1f} pts", delta=f"{delta_pts:+.1f} pts")
                pct_boost = ((post_mean - pre_mean) / max(1.0, abs(pre_mean))) * 100.0
                m3.metric("성과 향상률", f"{pct_boost:+.1f}%")

                # 6. 모델 다운로드 버튼 제공
                temp_zip_path = os.path.join(os.getcwd(), f"temp_browser_{selected_train_env}.zip")
                model.save(temp_zip_path)
                with open(temp_zip_path, "rb") as f:
                    zip_data = f.read()
                try:
                    os.remove(temp_zip_path)
                except Exception:
                    pass

                st.download_button(
                    label=f"💾 방금 브라우저에서 학습한 {selected_train_env} 모델 다운로드 (.zip)",
                    data=zip_data,
                    file_name=f"{selected_train_env}_browser_ppo.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            except Exception as e:
                status_box.error(f"학습 중 오류 발생: {e}")
                st.exception(e)

# ----------------------------------------------------
# Tab 5: TensorBoard & System Info
# ----------------------------------------------------
with tab5:
    st.markdown("### 📈 TensorBoard & System Management")

    col_tb1, col_tb2 = st.columns([1, 1])

    with col_tb1:
        st.markdown("#### 🔗 Real-time TensorBoard Dashboard")
        st.markdown("""
        TensorBoard를 통해 신경망 손실 함수(`train/loss`), 정책 그래디언트(`policy_gradient_loss`),
        에피소드 평균 보상(`rollout/ep_rew_mean`), 초당 시뮬레이션 FPS 등을 실시간으로 확인할 수 있습니다.
        """)
        st.link_button("🌐 Open TensorBoard (http://127.0.0.1:6006)", "http://127.0.0.1:6006")
        st.caption("팁: 로컬 실행 환경에서는 127.0.0.1:6006으로 접속하시면 바로 표시됩니다.")

    with col_tb2:
        st.markdown("#### ⚙️ 시스템 및 런타임 정보")
        st.code(f"""
Python Executable : {sys.executable}
Working Directory : {os.getcwd()}
Active Robots     : {len(ROBOT_CONFIGS)} environments
OS Platform       : Cloud / Linux Deployment Ready
Web Server        : Streamlit 1.64 + MuJoCo 3.13
Interactive RL    : Enabled (In-Browser PPO Training)
        """, language="text")

st.markdown("---")
st.markdown("<center style='color:#6c757d; font-size:0.85rem;'>MicroDuck RL Lab 🦆 Master Control Center | Built with Streamlit, Gymnasium & MuJoCo</center>", unsafe_allow_html=True)
