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

import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np

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

# ==================== 데이터 로드 ====================


def load_circuit_data():
    import os

    json_path = os.path.join(
        os.path.dirname(__file__), "schizophrenia_pe_circuit_map.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    circuit_data = load_circuit_data()
except FileNotFoundError:
    st.error("JSON 데이터 파일을 찾을 수 없습니다: schizophrenia_pe_circuit_map.json")
    st.stop()

# ==================== 시뮬레이션 함수 ====================


def calculate_circuit_distortion(slider_values, circuit_name):
    try:
        circuit_info = circuit_data["simulation_model"]["circuit_distortion_functions"][
            circuit_name
        ]
        base = circuit_info["base_activity"]
        distortion = 0

        for factor, params in circuit_info["distortion_factors"].items():
            if factor in slider_values:
                weight = params["weight"]
                direction = params["direction"]
                value = slider_values[factor] / 100

                if direction == "increase":
                    distortion += weight * value
                elif direction == "decrease":
                    distortion -= weight * value
                elif direction == "normalize":
                    distortion -= weight * value

        return max(0, min(1, base + distortion))
    except KeyError:
        return 0.5


def predict_symptoms(distortions):
    predictions = {
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

    results = {}
    for symptom_type, config in predictions.items():
        circuit_values = [distortions.get(c, 0.5) for c in config["circuits"]]
        avg_distortion = np.mean(circuit_values)

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


def plot_brain_circuits(slider_values, selected_symptoms):
    fig = go.Figure()

    coords = circuit_data.get(
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

    circuit_distortions = {
        "VTA_ventral_striatum": calculate_circuit_distortion(
            slider_values, "VTA_ventral_striatum"
        ),
        "thalamus_sensory_cortex": calculate_circuit_distortion(
            slider_values, "thalamus_sensory_cortex"
        ),
        "hippocampus_PFC": calculate_circuit_distortion(
            slider_values, "hippocampus_PFC"
        ),
        "ACC_PFC": calculate_circuit_distortion(slider_values, "ACC_PFC"),
    }

    for region, coord in coords.items():
        fig.add_trace(
            go.Scatter3d(
                x=[coord["x"]],
                y=[coord["y"]],
                z=[coord["z"]],
                mode="markers+text",
                marker=dict(size=12, color=coord.get("color", "#4488FF"), opacity=0.9),
                text=[region],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                name=region,
                hovertemplate=f"<b>{region}</b><br>PE 유형: {coord.get('pe_type', 'N/A')}<extra></extra>",
            )
        )

    connections = [
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

    for src, tgt, conn_type, weight in connections:
        if src in coords and tgt in coords:
            x = [coords[src]["x"], coords[tgt]["x"]]
            y = [coords[src]["y"], coords[tgt]["y"]]
            z = [coords[src]["z"], coords[tgt]["z"]]

            line_width = max(4, weight * 15)

            if weight < 0.4:
                line_color = "rgba(100, 255, 100, 0.9)"
            elif weight > 0.7:
                line_color = "rgba(255, 100, 100, 0.9)"
            else:
                ratio = (weight - 0.4) / 0.3
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


def main():
    st.title("🧠 조현병 신경회로 분석")
    st.markdown(WARNING_TEXT)

    st.sidebar.title("⚙️ 다차원 원인 조절")

    if st.sidebar.button("🔄 기본값으로 리셋"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    st.sidebar.subheader("🧬 유전적 요인")
    c4a_expr = st.sidebar.slider(
        "C4A 발현 정도",
        0,
        100,
        50,
        help="C4A 과발현 → 시냅스 가지치기 가속 → PE 회로 구조 변형",
    )
    disc1_expr = st.sidebar.slider(
        "DISC1 발현 정도", 0, 100, 50, help="DISC1 발현 이상 → 신경발달 장애"
    )

    st.sidebar.subheader("🧠 시냅스 가지치기")
    pruning_strength = st.sidebar.slider(
        "가지치기 강도",
        0,
        100,
        30,
        help="과도한 시냅스 제거 → SNR 20-40% 감소 → 회로 연결성 저하",
    )

    st.sidebar.subheader("💊 신경전달물질")
    dopamine_sens = st.sidebar.slider(
        "도파민 민감도",
        0,
        100,
        50,
        help="도파민 민감도 ↑ → 살리언스 정밀도 ↑ → 양성 증상 ↑",
    )
    glutamate_dys = st.sidebar.slider(
        "글루타메이트 이상", 0, 100, 40, help="NMDA 저기능 → 감각 PE 처리 장애"
    )
    gaba_imb = st.sidebar.slider(
        "GABA 불균형", 0, 100, 30, help="E/I 불균형 → 회로 안정성 저하"
    )

    st.sidebar.subheader("🛡️ 신경면역")
    microglia_act = st.sidebar.slider(
        "미세아교세포 활성도",
        0,
        100,
        30,
        help="과활성 → 시냅스 과도 제거 → PE 회로 붕괴",
    )

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

        selected_symptoms = []
        if delusions:
            selected_symptoms.append("delusions")
        if hallucinations:
            selected_symptoms.append("hallucinations")
        if anhedonia:
            selected_symptoms.append("anhedonia")
        if avolition:
            selected_symptoms.append("avolition")
        if working_memory:
            selected_symptoms.append("working_memory_deficit")
        if executive_func:
            selected_symptoms.append("executive_dysfunction")

        if selected_symptoms:
            st.success(f"선택된 증상: {len(selected_symptoms)}개")

    with tab2:
        st.header("PE 회로 시각화")
        st.markdown(
            "연결선 색상: 🟢 정상 (왜곡도 < 0.4) → 🟡 중간 → 🔴 비정상 (왜곡도 > 0.7)"
        )

        fig, circuit_distortions = plot_brain_circuits(slider_values, selected_symptoms)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("회로별 왜곡도")
        cols = st.columns(4)
        circuits = [
            (
                "VTA → 복부선조체",
                "reward_pe",
                circuit_distortions["VTA_ventral_striatum"],
            ),
            (
                "시상 → 감각피질",
                "sensory_pe",
                circuit_distortions["thalamus_sensory_cortex"],
            ),
            ("해마 → 전두엽", "contextual_pe", circuit_distortions["hippocampus_PFC"]),
            ("ACC → 전두엽", "error_pe", circuit_distortions["ACC_PFC"]),
        ]
        for col, (name, pe_type, distortion) in zip(cols, circuits):
            color = (
                "green" if distortion < 0.4 else "orange" if distortion < 0.7 else "red"
            )
            col.metric(name, f"{distortion:.2f}", delta=None)
            col.markdown(f"**PE 유형**: {pe_type}")

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
        
        ---
        
        ## 참고 문헌
        
        1. Millard et al. (2021). The prediction-error hypothesis of schizophrenia
        2. Ermakova et al. (2018). Abnormal reward prediction-error signalling
        3. Kapanaiah et al. (2024). C4A and synaptic pruning
        4. Larsen et al. (2023). Synaptic pruning in schizophrenia
        5. White et al. (2015). NMDA hypofunction
        """)


if __name__ == "__main__":
    main()
