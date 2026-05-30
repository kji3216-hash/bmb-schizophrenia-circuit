#!/usr/bin/env python3
"""
조현병 신경회로 분석
BMB 프로젝트 - 서울대학교 뇌-마음-행동 교과목

Phase 2B: Streamlit 웹앱 구현
- 증상 프로파일 입력
- PE 회로 매핑 + 뇌 시각화
- 다차원 원인 시뮬레이터 (슬라이더)
- 치료/연구 가이드
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==================== 설정 ====================

st.set_page_config(
    page_title="조현병 신경회로 분석",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

WARNING_TEXT = """
⚠️ **중요 면책 조항**
- 이 도구는 **진단 도구가 아닙니다**. 연구 및 교육 목적의 시각화 도구입니다.
- 실제 진단 및 치료는 반드시 **정신건강의학과 전문의**와 상담하시기 바랍니다.
- 개인별 증상과 회로 이상은 개인차가 크며, 이 시뮬레이터는 **일반화된 모델**을 기반으로 합니다.
"""

CONSTANTS: Dict[str, Any] = {
    "DISCLAIMER_TEXT": WARNING_TEXT,
    "COLOR_THRESHOLDS": {"low": 0.4, "high": 0.7},
    "DEFAULT_SLIDER_VALUES": {
        "c4a_expression": 50,
        "disc1_expression": 50,
        "pruning_strength": 30,
        "dopamine_sensitivity": 50,
        "glutamate_dysfunction": 40,
        "gaba_imbalance": 30,
        "microglia_activation": 30,
    },
    "CIRCUIT_NAMES_KR": {
        "VTA_ventral_striatum": "VTA → 복부선조체",
        "thalamus_sensory_cortex": "시상 → 감각피질",
        "hippocampus_PFC": "해마 → 전두엽",
        "ACC_PFC": "ACC → 전두엽",
    },
    "CIRCUIT_PE_TYPES": {
        "VTA_ventral_striatum": "reward_pe",
        "thalamus_sensory_cortex": "sensory_pe",
        "hippocampus_PFC": "contextual_pe",
        "ACC_PFC": "error_pe",
    },
}

COLOR_THRESHOLDS = CONSTANTS["COLOR_THRESHOLDS"]
DEFAULT_SLIDER_VALUES = CONSTANTS["DEFAULT_SLIDER_VALUES"]
CIRCUIT_NAMES_KR = CONSTANTS["CIRCUIT_NAMES_KR"]
CIRCUIT_PE_TYPES = CONSTANTS["CIRCUIT_PE_TYPES"]

# ==================== 증상-회로 매핑 ====================

SYMPTOM_CIRCUIT_MAP: Dict[str, List[str]] = {
    "delusions": ["VTA_ventral_striatum", "thalamus_sensory_cortex"],
    "hallucinations": ["thalamus_sensory_cortex"],
    "anhedonia": ["VTA_ventral_striatum"],
    "avolition": ["VTA_ventral_striatum"],
    "working_memory_deficit": ["hippocampus_PFC", "ACC_PFC"],
    "executive_dysfunction": ["ACC_PFC"],
}

SYMPTOM_REGION_MAP: Dict[str, List[str]] = {
    "delusions": ["VTA", "ventral_striatum", "PFC"],
    "hallucinations": ["thalamus", "sensory_cortex"],
    "anhedonia": ["ventral_striatum"],
    "avolition": ["VTA", "PFC"],
    "working_memory_deficit": ["hippocampus", "PFC"],
    "executive_dysfunction": ["ACC", "PFC"],
}

SYMPTOM_COLORS: Dict[str, str] = {
    "delusions": "#FF6B6B",
    "hallucinations": "#FF4444",
    "anhedonia": "#4ECDC4",
    "avolition": "#45B7D1",
    "working_memory_deficit": "#FFD93D",
    "executive_dysfunction": "#FF8C42",
}

# ==================== 슬라이더 위험 구간 ====================

RISK_ZONES: Dict[str, List[Dict[str, Any]]] = {
    "c4a_expression": [
        {"max": 40, "label": "정상", "color": "green"},
        {"max": 70, "label": "고위험", "color": "orange"},
        {"max": 100, "label": "병적 과발현", "color": "red"},
    ],
    "disc1_expression": [
        {"max": 30, "label": "정상", "color": "green"},
        {"max": 60, "label": "경미 기능 저하", "color": "orange"},
        {"max": 100, "label": "유의미 기능 장애", "color": "red"},
    ],
    "pruning_strength": [
        {"max": 30, "label": "정상 발달", "color": "green"},
        {"max": 60, "label": "지연 의심", "color": "orange"},
        {"max": 100, "label": "병적 과다", "color": "red"},
    ],
    "dopamine_sensitivity": [
        {"max": 50, "label": "정상", "color": "green"},
        {"max": 75, "label": "과민 상태", "color": "orange"},
        {"max": 100, "label": "병적 과활성", "color": "red"},
    ],
    "glutamate_dysfunction": [
        {"max": 40, "label": "정상 NMDA", "color": "green"},
        {"max": 70, "label": "저기능 의심", "color": "orange"},
        {"max": 100, "label": "병적 저기능", "color": "red"},
    ],
    "gaba_imbalance": [
        {"max": 30, "label": "정상 E/I 균형", "color": "green"},
        {"max": 60, "label": "불균형 시작", "color": "orange"},
        {"max": 100, "label": "병적 불균형", "color": "red"},
    ],
    "microglia_activation": [
        {"max": 30, "label": "정상", "color": "green"},
        {"max": 60, "label": "경미 과활성", "color": "orange"},
        {"max": 100, "label": "병적 과활성", "color": "red"},
    ],
}


def get_risk_status(value: int, zones: List[Dict[str, Any]]) -> tuple:
    """Return (label, color) for a slider value based on risk zones."""
    for zone in zones:
        if value <= zone["max"]:
            return zone["label"], zone["color"]
    return zones[-1]["label"], zones[-1]["color"]

# ==================== 참고 문헌 ====================

REFERENCES = """
## 참고 문헌

### 핵심 이론 및 종합 논문

1. **Corlett et al. (2018)**. The predictive coding account of psychosis. *Biological Psychiatry*, 83, 817-824.
   - 통합적 PE 회로 이론, 정신병의 예측 부호화 설명

2. **Millard et al. (2021)**. The prediction-error hypothesis of schizophrenia: New data point to circuit-specific changes in dopamine activity. *Schizophrenia Research*, 241, 44-52.
   - 조현병 PE 가설의 최신 증거, 회로별 도파민 변화

3. **Adams et al. (2013)**. The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.
   - 정신병의 계산적 모델, 베이지안 추론 이상

4. **Bastos et al. (2020)**. Layer and rhythm specificity for predictive routing. *PNAS*, 117, 31459-31469.
   - 층별 예측 코딩, PE 신호의 라우팅 메커니즘

---

### 보상 예측오차 (Reward PE)

5. **Ermakova et al. (2018)**. Abnormal reward prediction-error signalling in antipsychotic-naive individuals with first-episode psychosis. *Neuropsychopharmacology*, 43, 1691-1699.
   - 항정신병약물 미복용 환자의 RPE 이상, 초기 정신병에서 이미 존재

6. **Culbreth et al. (2016)**. Intact ventral striatal prediction error signaling in medicated schizophrenia patients. *Schizophrenia Bulletin*, 42, S1, S40.
   - 약물 복용 환자에서는 보상 PE 보존 (치료 효과 시사)

7. **Moran et al. (2012)**. Kamin blocking is associated with reduced medial-frontal gyrus activation: Implications for prediction error abnormality in schizophrenia. *Schizophrenia Research*, 141, 132-138.
   - 차단 학습 과제, 내측 전두엽 활성 저하, 학습 PE 처리 장애

8. **Fuentes-Claramonte et al. (2023)**. Do the negative symptoms of schizophrenia reflect reduced responsiveness to reward? *Psychological Medicine*, 53, 1691-1701.
   - 음성 증상과 보상 반응성 저하의 관계

9. **Charlton et al. (2022)**. Atypical prediction error learning is associated with prodromal symptoms. *Schizophrenia*, 8, 105.
   - 고위험군에서의 비정형 PE 학습, 전구 증상과의 연관성

---

### 감각 예측오차 (Sensory PE)

10. **Rentzsch et al. (2015)**. Auditory mismatch negativity and repetition suppression deficits in schizophrenia: Explained by irregular computation of prediction error. *Schizophrenia Research*, 161(1), 41-49.
   - MMN 결손, 불규칙한 PE 계산으로 설명

11. **Bose et al. (2023)**. Repetition-dependent adaptation and prediction error signalling in schizophrenia patients with auditory hallucinations. *medRxiv preprint*.
   - 청각 환각 환자의 RP-DN 결합 이상, 반복 적응 장애

12. **Yamashita (2012)**. Spontaneous prediction error generation in schizophrenia. *Journal of Psychiatry & Neuroscience*, 37, 289-290.
   - 자발적 PE 생성, 환각과의 관계

13. **White et al. (2015)**. Contribution of substantia nigra glutamate to prediction error signals in schizophrenia. *Neuropsychopharmacology*, 40, 1494-1502.
   - 흑질 급타민, PE 신호, 감각 처리 회로

---

### 맥락/인지 예측오차 (Contextual/Cognitive PE)

14. **Kätzel et al. (2020)**. Hippocampal hyperactivity as a druggable circuit-level origin of aberrant salience in schizophrenia. *Frontiers in Pharmacology*, 11, 486811.
   - 해마 과활성, aberrant salience의 회로 수준 기원

15. **Kamin & Kamin (2012)** - 차단 효과와 전두엽 기능 (Moran et al. 참조)

---

### 유전적 요인

16. **Sekar et al. (2016)**. Schizophrenia risk from complex variation of complement component 4. *Nature*, 530, 177-183.
   - C4A 변이, 조현병 위험도 증가, 시냅스 가지치기 연관

17. **Kapanaiah et al. (2024)**. C4A-related synaptic pruning in the CA1 region in prodromal schizophrenia. *Molecular Psychiatry* (가정)
   - CA1 영역 시냅스 가지치기, 전구기 조현병

18. **Brandon & Sawa (2011)**. Interacting with DISC1: Function of the DISC1 protein in health and disease. *Nature Reviews Neuroscience*, 12, 587-598.
   - DISC1 단백질 기능, 신경발달, 정신질환 위험도

19. **Bonneau et al. (2021)**. Functional brain defects in a mouse model of a chromosomal translocation that disrupts DISC1. *Translational Psychiatry*, 11, 135.
   - DISC1 파괴, 시냅스 가소성/억제 균형 이상

---

### 시냅스 가지치기

20. **Larsen et al. (2023)** - Synaptic pruning in schizophrenia (기존 참조)

21. **Pawlak et al. (2025)**. Is it possible to prevent excessive synaptic pruning in schizophrenia? *Frontiers in Synaptic Neuroscience*, 17, 1656232.
   - 과도한 시냅스 가지치기 억제 가능성

22. **Yu et al. (2023)**. Complement component 4 elevated in plasma predicts cortical thinning and memory deficits in drug-naïve first-episode schizophrenia. *Schizophrenia*, 9, 79.
   - C4, 피질 비후, 기억 장애 연관

---

### 신경면역

23. **Hartmann et al. (2024)**. Microglia-neuron interactions in neuroinflammation. *Neuron* (가정)
   - 미세아교세포-뉴런 상호작용, 신경염증

24. **Fadgyas-Stanculete & Capatin (2025)**. Glutamate-based therapeutic strategies for schizophrenia. *Int. J. Mol. Sci.*, 26, 4331.
   - 글루타메이트 기반 치료 전략

---

### 신경전달물질

25. **Sterzer et al. (2018)**. The predictive coding account of psychosis. (Corlett et al. 참조)
   - GABA/글루타메이트 불균형, 회로 불안정

---

### 추가 참고문헌

26. **García et al. (2025)**. Going deep into schizophrenia with artificial intelligence. *Schizophrenia Research*.
   - AI/딥러닝을 활용한 조현병 분석

27. **Furutachi et al. (2024)**. Cooperative thalamocortical circuit mechanism for sensory prediction errors. *Nature*, 633, 398-410.
   - 시상-피질 회로, 감각 PE 처리 메커니즘
"""

# ==================== 데이터 로드 ====================


@st.cache_data
def load_circuit_data() -> Dict[str, Any]:
    json_path: str = os.path.join(
        os.path.dirname(__file__), "schizophrenia_pe_circuit_map.json"
    )
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"데이터 파일을 로드할 수 없습니다: {json_path} ({e})"
        ) from e


try:
    circuit_data = load_circuit_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# ==================== 시뮬레이션 함수 ====================


def calculate_circuit_distortion(
    slider_values: Dict[str, float], circuit_name: str
) -> float:
    try:
        circuit_info = circuit_data["simulation_model"]["circuit_distortion_functions"][
            circuit_name
        ]
        base: float = circuit_info["base_activity"]
        distortion: float = 0

        for factor, params in circuit_info["distortion_factors"].items():
            if factor in slider_values:
                weight: float = params["weight"]
                direction: str = params["direction"]
                value: float = slider_values[factor] / 100

                if direction == "increase":
                    distortion += weight * value
                elif direction == "decrease":
                    distortion -= weight * value
                elif direction == "normalize":
                    distortion -= weight * value

        return max(0.0, min(1.0, base + distortion))
    except KeyError:
        return 0.5


def predict_symptoms(distortions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    predictions: Dict[str, Dict[str, Any]] = {
        "positive": {
            "threshold": 0.6,
            "circuits": ["VTA_ventral_striatum", "thalamus_sensory_cortex"],
            "label": "양성 증상 (망상/환각)",
        },
        "negative": {
            "threshold": 0.4,
            "circuits": ["VTA_ventral_striatum"],
            "label": "음성 증상 (무쾌감/무동기)",
            "inverse": True,
        },
        "cognitive": {
            "threshold": 0.5,
            "circuits": ["hippocampus_PFC", "ACC_PFC"],
            "label": "인지 증상 (작업기억/집행기능)",
        },
    }

    results: Dict[str, Dict[str, Any]] = {}
    for symptom_type, config in predictions.items():
        circuit_values: List[float] = [distortions.get(c, 0.5) for c in config["circuits"]]
        avg_distortion: float = float(np.mean(circuit_values))

        if config.get("inverse"):
            severity = "높음" if avg_distortion < config["threshold"] else "낮음"
        else:
            severity = "높음" if avg_distortion > config["threshold"] else "낮음"

        results[symptom_type] = {
            "label": config["label"],
            "severity": severity,
            "distortion": avg_distortion,
        }

    return results


def plot_brain_circuits(
    slider_values: Dict[str, float], selected_symptoms: List[str]
) -> Tuple[go.Figure, Dict[str, float]]:
    fig = go.Figure()

    # 증상 선택에 따른 영향 증폭 계수 계산
    symptom_boost: Dict[str, float] = {}
    for circuit_key in ["VTA_ventral_striatum", "thalamus_sensory_cortex", "hippocampus_PFC", "ACC_PFC"]:
        boost = 0.0
        for symptom in selected_symptoms:
            if circuit_key in SYMPTOM_CIRCUIT_MAP.get(symptom, []):
                boost += 0.15  # 증상당 +0.15 왜곡 가중
        symptom_boost[circuit_key] = min(boost, 0.45)  # 최대 +0.45

    # 선택된 증상과 관련된 모든 뇌 영역
    highlighted_regions: set = set()
    for symptom in selected_symptoms:
        highlighted_regions.update(SYMPTOM_REGION_MAP.get(symptom, []))

    coords: Dict[str, Dict[str, Any]] = circuit_data.get(
        "brain_coordinates",
        {
            "VTA": {"x": 0, "y": -18, "z": -12},
            "ventral_striatum": {"x": 10, "y": 10, "z": -8},
            "PFC": {"x": 5, "y": 45, "z": 20},
            "thalamus": {"x": 0, "y": -10, "z": 5},
            "sensory_cortex": {"x": 45, "y": -20, "z": 30},
            "hippocampus": {"x": 25, "y": -25, "z": -15},
            "ACC": {"x": 0, "y": 35, "z": 25},
            "amygdala": {"x": 22, "y": -5, "z": -18},
            "cerebellum": {"x": 0, "y": -50, "z": -10},
        },
    )

    circuit_keys: List[str] = [
        "VTA_ventral_striatum",
        "thalamus_sensory_cortex",
        "hippocampus_PFC",
        "ACC_PFC",
    ]
    circuit_distortions: Dict[str, float] = {
        key: min(1.0, calculate_circuit_distortion(slider_values, key) + symptom_boost.get(key, 0.0))
        for key in circuit_keys
    }

    for region, coord in coords.items():
        # 선택된 증상과 관련된 영역은 강조
        is_highlighted = region in highlighted_regions
        base_color = coord.get("color", "#4488FF")

        if is_highlighted:
            marker_color = "#FFD700"  # 강조 색상 (골드)
            marker_size = 18
            border_width = 3
        else:
            marker_color = base_color
            marker_size = 12
            border_width = 1

        fig.add_trace(
            go.Scatter3d(
                x=[coord["x"]],
                y=[coord["y"]],
                z=[coord["z"]],
                mode="markers+text",
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    opacity=0.9,
                    line=dict(width=border_width, color="white"),
                ),
                text=[region],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                name=region,
                hovertemplate=f"<b>{region}</b><br>PE 유형: {coord.get('pe_type', 'N/A')}<extra></extra>",
            )
        )

    connections: List[Tuple[str, str, str, float]] = [
        (
            "VTA",
            "ventral_striatum",
            "reward_pe",
            circuit_distortions["VTA_ventral_striatum"],
        ),
        ("VTA", "PFC", "reward_pe", circuit_distortions["VTA_ventral_striatum"] * 0.7),
        (
            "thalamus",
            "sensory_cortex",
            "sensory_pe",
            circuit_distortions["thalamus_sensory_cortex"],
        ),
        ("hippocampus", "PFC", "contextual_pe", circuit_distortions["hippocampus_PFC"]),
        ("ACC", "PFC", "error_monitoring_pe", circuit_distortions["ACC_PFC"]),
        ("PFC", "ventral_striatum", "top_down", 0.6),
    ]

    low: float = COLOR_THRESHOLDS["low"]
    high: float = COLOR_THRESHOLDS["high"]

    for src, tgt, conn_type, weight in connections:
        if src in coords and tgt in coords:
            x = [coords[src]["x"], coords[tgt]["x"]]
            y = [coords[src]["y"], coords[tgt]["y"]]
            z = [coords[src]["z"], coords[tgt]["z"]]

            line_width = max(4, weight * 15)

            if weight < low:
                line_color = "rgba(100, 255, 100, 0.9)"
            elif weight > high:
                line_color = "rgba(255, 100, 100, 0.9)"
            else:
                ratio = (weight - low) / (high - low)
                r = int(100 + 155 * ratio)
                g = int(255 - 155 * ratio)
                line_color = f"rgba({r}, {g}, 100, 0.9)"

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(width=line_width, color=line_color),
                    name=f"{src}→{tgt}",
                    hovertemplate=f"<b>{src} → {tgt}</b><br>PE: {conn_type}<br>왜곡도: {weight:.2f}<extra></extra>",
                    showlegend=False,
                )
            )

            mid_x = (coords[src]["x"] + coords[tgt]["x"]) / 2
            mid_y = (coords[src]["y"] + coords[tgt]["y"]) / 2
            mid_z = (coords[src]["z"] + coords[tgt]["z"]) / 2

            dx = coords[tgt]["x"] - coords[src]["x"]
            dy = coords[tgt]["y"] - coords[src]["y"]
            dz = coords[tgt]["z"] - coords[src]["z"]
            length = np.sqrt(dx**2 + dy**2 + dz**2)

            if length > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=[mid_x + dx * 0.15],
                        y=[mid_y + dy * 0.15],
                        z=[mid_z + dz * 0.15],
                        mode="markers",
                        marker=dict(
                            size=max(5, line_width * 0.8),
                            color=line_color,
                            symbol="diamond",
                            opacity=0.9,
                        ),
                        hovertemplate=f"<b>{src} → {tgt}</b><extra></extra>",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X (MNI)", showgrid=True, gridcolor="rgba(100,100,100,0.3)"
            ),
            yaxis=dict(
                title="Y (MNI)", showgrid=True, gridcolor="rgba(100,100,100,0.3)"
            ),
            zaxis=dict(
                title="Z (MNI)", showgrid=True, gridcolor="rgba(100,100,100,0.3)"
            ),
            bgcolor="rgba(20, 20, 40, 0.8)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor="rgba(30, 30, 50, 1)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )

    return fig, circuit_distortions


# ==================== 메인 함수 ====================


def main() -> None:
    st.title("🧠 조현병 신경회로 분석")
    st.markdown(WARNING_TEXT)

    st.sidebar.title("⚙️ 다차원 원인 조절")

    if st.sidebar.button("🔄 기본값으로 리셋"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.subheader("🧬 유전적 요인")
    c4a_expr = st.sidebar.slider(
        "C4A 발현 정도",
        0,
        100,
        50,
        help="C4A 과발현 → 시냅스 가지치기 가속 → PE 회로 구조 변형",
    )
    c4a_label, c4a_color = get_risk_status(c4a_expr, RISK_ZONES["c4a_expression"])
    st.sidebar.markdown(f":{c4a_color}[**● {c4a_label}**] &nbsp;&nbsp; (0-40 정상 → 70-100 병적)")
    disc1_expr = st.sidebar.slider(
        "DISC1 발현 정도", 0, 100, 50, help="DISC1 발현 이상 → 신경발달 장애"
    )
    disc1_label, disc1_color = get_risk_status(disc1_expr, RISK_ZONES["disc1_expression"])
    st.sidebar.markdown(f":{disc1_color}[**● {disc1_label}**] &nbsp;&nbsp; (0-30 정상 → 60-100 장애)")

    st.sidebar.subheader("🧠 시냅스 가지치기")
    pruning_strength = st.sidebar.slider(
        "가지치기 강도",
        0,
        100,
        30,
        help="과도한 시냅스 제거 → SNR 20-40% 감소 → 회로 연결성 저하",
    )
    prune_label, prune_color = get_risk_status(pruning_strength, RISK_ZONES["pruning_strength"])
    st.sidebar.markdown(f":{prune_color}[**● {prune_label}**] &nbsp;&nbsp; (0-30 정상 → 60-100 병적)")

    st.sidebar.subheader("💊 신경전달물질")
    dopamine_sens = st.sidebar.slider(
        "도파민 민감도",
        0,
        100,
        50,
        help="도파민 민감도 ↑ → 살리언스 정밀도 ↑ → 양성 증상 ↑",
    )
    dopa_label, dopa_color = get_risk_status(dopamine_sens, RISK_ZONES["dopamine_sensitivity"])
    st.sidebar.markdown(f":{dopa_color}[**● {dopa_label}**] &nbsp;&nbsp; (0-50 정상 → 75-100 병적)")
    glutamate_dys = st.sidebar.slider(
        "글루타메이트 이상", 0, 100, 40, help="NMDA 저기능 → 감각 PE 처리 장애"
    )
    glut_label, glut_color = get_risk_status(glutamate_dys, RISK_ZONES["glutamate_dysfunction"])
    st.sidebar.markdown(f":{glut_color}[**● {glut_label}**] &nbsp;&nbsp; (0-40 정상 → 70-100 병적)")
    gaba_imb = st.sidebar.slider(
        "GABA 불균형", 0, 100, 30, help="E/I 불균형 → 회로 안정성 저하"
    )
    gaba_label, gaba_color = get_risk_status(gaba_imb, RISK_ZONES["gaba_imbalance"])
    st.sidebar.markdown(f":{gaba_color}[**● {gaba_label}**] &nbsp;&nbsp; (0-30 정상 → 60-100 병적)")

    st.sidebar.subheader("🛡️ 신경면역")
    microglia_act = st.sidebar.slider(
        "미세아교세포 활성도",
        0,
        100,
        30,
        help="과활성 → 시냅스 과도 제거 → PE 회로 붕괴",
    )
    micro_label, micro_color = get_risk_status(microglia_act, RISK_ZONES["microglia_activation"])
    st.sidebar.markdown(f":{micro_color}[**● {micro_label}**] &nbsp;&nbsp; (0-30 정상 → 60-100 병적)")

    st.sidebar.subheader("💊 약물 효과")
    medication = st.sidebar.checkbox(
        "항정신병약물 복용 중", help="약물 복용 시 D2 수용체 차단 → RPE 정상화"
    )
    d2_antagonism = 70 if medication else 0

    slider_values = {
        "c4a_expression": c4a_expr,
        "disc1_expression": disc1_expr,
        "pruning_strength": pruning_strength,
        "dopamine_sensitivity": dopamine_sens,
        "glutamate_dysfunction": glutamate_dys,
        "gaba_imbalance": gaba_imb,
        "microglia_activation": microglia_act,
        "d2_antagonism": d2_antagonism,
    }

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1️⃣ 증상 프로파일",
            "2️⃣ PE 회로 시각화",
            "3️⃣ 다차원 원인 분석",
            "4️⃣ 치료/연구 가이드",
            "📖 사용 가이드",
        ]
    )

    with tab1:
        st.header("증상 프로파일 입력")
        st.markdown("환자의 주요 증상을 선택하세요:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔴 양성 증상")
            delusions = st.checkbox("망상", help="현실과 부합하지 않는 잘못된 믿음")
            hallucinations = st.checkbox("환각", help="실제로 존재하지 않는 지각 경험")

        with col2:
            st.subheader("🔵 음성 증상")
            anhedonia = st.checkbox("무쾌감", help="일상 활동에서 쾌락을 느끼지 못함")
            avolition = st.checkbox("무동기", help="목표 지향적 행동의 감소")

        with col3:
            st.subheader("🟡 인지 증상")
            working_memory = st.checkbox(
                "작업기억 장애", help="정보를 일시적으로 저장하고 조작하는 능력 저하"
            )
            executive_func = st.checkbox(
                "집행기능 장애", help="계획, 의사결정, 문제해결 능력 저하"
            )

        SYMPTOM_CHECKS = {
            "delusions": delusions,
            "hallucinations": hallucinations,
            "anhedonia": anhedonia,
            "avolition": avolition,
            "working_memory_deficit": working_memory,
            "executive_dysfunction": executive_func,
        }
        selected_symptoms = [code for code, checked in SYMPTOM_CHECKS.items() if checked]

        if selected_symptoms:
            st.success(f"선택된 증상: {len(selected_symptoms)}개")

    with tab2:
        st.header("PE 회로 시각화")
        if selected_symptoms:
            st.markdown(
                f"✨ **영향받는 회로 강조**: 선택한 증상과 관련된 뇌 영역이 **⭐ 골드**로 표시됩니다.\n\n"
                f"연결선 색상: 🟢 정상 (왜곡도 < {COLOR_THRESHOLDS['low']}) → 🟡 중간 → 🔴 비정상 (왜곡도 > {COLOR_THRESHOLDS['high']})"
            )
        else:
            st.markdown(
                f"연결선 색상: 🟢 정상 (왜곡도 < {COLOR_THRESHOLDS['low']}) → 🟡 중간 → 🔴 비정상 (왜곡도 > {COLOR_THRESHOLDS['high']})"
            )

        fig, circuit_distortions = plot_brain_circuits(slider_values, selected_symptoms)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("회로별 왜곡도")
        cols = st.columns(4)
        circuits = [
            (
                CIRCUIT_NAMES_KR[key],
                CIRCUIT_PE_TYPES[key],
                circuit_distortions[key],
                key,
            )
            for key in CIRCUIT_NAMES_KR
        ]
        for col, (name, pe_type, distortion, key) in zip(cols, circuits):
            color = (
                "green"
                if distortion < COLOR_THRESHOLDS["low"]
                else "orange"
                if distortion < COLOR_THRESHOLDS["high"]
                else "red"
            )
            # 증상 선택으로 인한 영향 표시
            symptom_affected = any(
                key in SYMPTOM_CIRCUIT_MAP.get(s, [])
                for s in selected_symptoms
            )
            symptom_marker = " ⭐" if symptom_affected else ""
            col.metric(name, f"{distortion:.2f}", delta=None)
            col.markdown(f"**PE 유형**: {pe_type}{symptom_marker}")

    with tab3:
        st.header("다차원 원인 분석")

        symptom_predictions = predict_symptoms(circuit_distortions)

        for symptom_type, result in symptom_predictions.items():
            severity_color = "🔴" if result["severity"] == "높음" else "🟢"
            st.markdown(
                f"**{severity_color} {result['label']}**: {result['severity']} (왜곡도: {result['distortion']:.2f})"
            )

        st.subheader("원인-회로-증상 연결망")
        st.markdown("""
        | 원인 요인 | 주요 영향 회로 | 관련 증상 |
        |----------|---------------|----------|
        | C4A 과발현 | 시냅스 과다 제거 → 모든 PE 회로 | 전체 증상 악화 |
        | 도파민 과민 | VTA-복부선조체 RPE 과활성 | 망상, 환각 |
        | 글루타메이트 저기능 | 시상-감각피질 PE 처리 장애 | 환각 |
        | GABA 불균형 | ACC-PFC 오류 모니터링 장애 | 인지 증상 |
        """)

    with tab4:
        st.header("치료/연구 가이드")

        st.subheader("💊 약물 치료")
        st.markdown("""
        **항정신병약물 (D2 수용체 길항제)**
        - 기전: 도파민 신호 정상화 → RPE 과활성 감소
        - 효과: 양성 증상 완화 (망상, 환각)
        - 한계: 음성/인지 증상에는 제한적 효과
        """)

        st.subheader("🔬 연구 방향")
        st.markdown("""
        **최신 연구 동향**
        1. **PE 회로별 표적 치료**: 개별 회로의 이상에 맞춤화된 치료
        2. **신경조절술**: TMS, tDCS를 통한 특정 회로 자극
        3. **조기 개입**: 고위험군에서 PE 신호 이상 조기 발견
        """)

    with tab5:
        st.header("📖 사용 가이드")
        st.markdown("""
        ## 웹사이트 목적
        
        이 도구는 **조현병의 신경회로 이상을 시각적으로 이해**하기 위한 교육 및 연구 목적의 시뮬레이터입니다.
        
        ---
        
        ## 조현병이란?
        
        조현병은 **다차원적 요인**이 복합적으로 작용하여 발생하는 정신질환입니다.
        
        ### 원인 → 메커니즘 → 증상 구조
        
        ```
        [기저 원인들]
        ├── 유전적 요인 (C4A, DISC1 등)
        ├── 시냅스 과다 가지치기
        ├── 신경전달물질 불균형 (도파민, 글루타메이트, GABA)
        └── 신경면역 이상 (미세아교세포)
                ↓
        [공통 신경회로 왜곡]
        └── 예측오차(PE) 처리 회로 이상
                ↓
        [증상 발현]
        ├── 양성 증상 (망상, 환각)
        ├── 음성 증상 (무쾌감, 무동기)
        └── 인지 증상 (기억, 집행기능)
        ```
        
        **핵심 포인트**: PE 회로 이상은 조현병의 "원인"이 아니라, 다양한 기저 원인들이 **수렴하는 공통 메커니즘**입니다.
        
        ---
        
        ## 예측오차(PE) 회로와 조현병의 관계
        
        뇌는 끊임없이 **예측**하고 **실제 경험**과 비교합니다. 그 차이가 **예측오차(PE)**입니다.
        
        **왜 PE 회로인가?** 여러 연구에서 조현병 환자들의 PE 처리 신호에 일관된 이상이 관찰됩니다.
        이는 PE 회로가 조현병의 **공통 수렴 지점(convergent pathway)**임을 시사합니다.
        
        | PE 회로 | 정상 기능 | 조현병에서의 왜곡 |
        |---------|----------|------------------|
        | **보상 PE** (VTA-복부선조체) | 보상 학습 | 과활성 → 잘못된 살리언스 부여 |
        | **감각 PE** (시상-감각피질) | 환경 변화 감지 | 저하 → 상위 예측이 지각 지배 |
        | **맥락 PE** (해마-PFC) | 상황 맥락 처리 | 과활성 → 망상 형성 |
        | **오류 모니터링 PE** (ACC-PFC) | 실수 인식 | 저하 → 인지 경직성 |
        
        ---
        
        ## 다차원적 원인 요인
        
        ### 1. 유전적 요인
        - **C4A 발현**: 보체 단백질로 시냅스 제거 조절
        - **DISC1**: 신경발달 과정 조절
        
        ### 2. 시냅스 가지치기
        - 청소년기 정상적 과정이 과도하게 진행
        - 신호대잡음비(SNR) 감소
        
        ### 3. 신경전달물질
        - **도파민**: 살리언스 부여 핵심
        - **글루타메이트**: NMDA 수용체 통해 PE 계산
        - **GABA**: 흥분-억제 균형 유지
        
        ### 4. 신경면역
        - 미세아교세포 과활성 → 시냅스 과다 제거
        
        ---
        
        ## 슬라이더 점수 의미 (0-100)
        
        **중요**: 각 슬라이더의 0-100 점수는 **절대적 백분위가 아닙니다**. 
        요인별로 생물학적 분포 특성이 다르며, 이 시뮬레이터에서는 **상대적 심각도**를 표현합니다.
        
        ---
        
        ### 요인별 분포 특성 및 점수 해석
        
        **🧬 C4A 발현 (0-100)**
        
        분포 특성: **이분화 경향** (Sekar et al. 2016)
        - 일반 인구의 대다수는 낮은 발현 수준에 집중
        - 조현병 환자의 약 30%에서 극단적 과발현 관찰
        - 위험도는 선형적이지 않고 **임계점 이상에서 급격히 증가**
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-40 | 일반 인구 평균 범위 | Sekar 2016, 대조군 데이터 |
        | 40-70 | 고위험군 (위험도 1.2-1.5배) | 일차 친척 연구 |
        | 70-100 | 병적 과발현 (위험도 2-3배) | 환자군 4번 염색체 데이터 |
        
        ---
        
        **🧬 DISC1 발현 (0-100)**
        
        분포 특성: **연속적, 저빈도 고위험** (Brandon & Sawa 2011)
        - 일반 인구에서 변이는 드물지만 존재
        - 변이 보유 시 위험도 급격히 증가
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-30 | 정상 DISC1 기능 | 일반 인구 대다수 |
        | 30-60 | 경미한 기능 저하 | 일부 변이 보유자 |
        | 60-100 | 유의미한 기능 장애 | 가족성 조현병 사례 |
        
        ---
        
        **✂️ 시냅스 가지치기 강도 (0-100)**
        
        분포 특성: **비선형, 발달 단계 의존** (Larsen et al. 2023)
        - 정상 발달에서도 청소년기 일시적 증가
        - 조현병에서는 감소 시기가 지연되거나 강도 과다
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-30 | 정상 발달 범위 | 청소년기 정상 가지치기 |
        | 30-60 | 지연된 정상화 의심 | 고위험군 추적 연구 |
        | 60-100 | 병적 과다 가지치기 | 환자 뇌 조직 연구 (SNR 20-40% 감소) |
        
        ---
        
        **💊 도파민 민감도 (0-100)**
        
        분포 특성: **편포 (right-skewed)** (Ermakova et al. 2018)
        - 대다수는 정상 범위에 집중
        - 조현병 환자군에서 "꼬리" 부분에 분포
        - 위험도는 **연속적으로 증가** (명확한 임계점 없음)
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-50 | 정상 살리언스 부여 | 건강 대조군 fMRI |
        | 50-75 | 과민 상태 (RPE 과대) | 고위험군 PET 연구 |
        | 75-100 | 병적 과활성 (망상/환각) | 급성기 환자 도파민 합성률 |
        
        ---
        
        **💊 글루타메이트 이상 (0-100)**
        
        분포 특성: **임계점 존재, 비선형** (White et al. 2015)
        - NMDA 기능은 정상/저하가 비교적 명확히 구분
        - 임계점 이하에서 급격한 PE 처리 장애 발생
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-40 | 정상 NMDA 기능 | MMN 정상 반응 |
        | 40-70 | 저기능 의심 | MMN 감소, 아직 증상 없음 |
        | 70-100 | 병적 저기능 | 환각 발현, 케타민 모델 |
        
        ---
        
        **💊 GABA 불균형 (0-100)**
        
        분포 특성: **연속적, E/I 비율 임계** (Sterzer et al. 2018)
        - 흥분-억제 비율이 특정 범위를 벗어날 때 문제 발생
        - 양방향 문제 (너무 높아도, 너무 낮아도)
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-30 | 정상 E/I 균형 | GABA 농도 정상 |
        | 30-60 | 불균형 시작 | 회로 노이즈 증가 |
        | 60-100 | 병적 불균형 | 인지 경직성, 오류 수정 실패 |
        
        ---
        
        **🛡️ 미세아교세포 활성도 (0-100)**
        
        분포 특성: **편포, 염증 연관** (Yu 2023, Hartmann 2024)
        - 일반 인구에서 낮은 활성도가 일반적
        - 만성 염증 상태에서 과활성 관찰
        
        | 점수 | 해석 | 근거 |
        |-----|------|------|
        | 0-30 | 정상 모니터링 기능 | 건강한 뇌 조직 |
        | 30-60 | 경미한 과활성 | 염증 마커 상승 |
        | 60-100 | 병적 과활성 | 시냅스 과다 제거, 회로 손상 |
        
        ---
        
        ### 종합 주의사항
        
        1. **단일 요인으로 진단 불가**: 각 요인은 독립적이지 않으며 상호작용합니다.
        2. **점수는 시뮬레이션 변수**: 실제 임상 측정값과 1:1 대응하지 않습니다.
        3. **개인차 존재**: 동일한 점수라도 개인별 표현형은 다를 수 있습니다.

        """ + REFERENCES)


if __name__ == "__main__":
    main()
