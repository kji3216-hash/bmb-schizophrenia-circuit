# 🧠 조현병 예측오차-신경회로 통합 모델링

**BMB 과제 — 서울대학교 뇌-마음-행동 교과목**

> 14개 선행 연구의 문헌 고찰을 바탕으로, 조현병의 양성·음성·인지 증상을 예측오차(Prediction Error) 유형과 뇌 회로 수준에서 통합 매핑한 대화형 시각화 도구

---

## 📌 프로젝트 개요

조현병은 단일 원인이 아닌 **유전·시냅스·신경전달물질·신경면역** 등 다차원적 요인이 복합적으로 작용하는 질환입니다. 본 프로젝트는 이러한 위험 요인들이 뇌 내 예측오차(PE) 회로를 어떻게 왜곡시키는지, 그리고 그것이 증상으로 어떻게 연결되는지를 **상호작용적 웹 앱**으로 구현했습니다.

### 핵심 특징

| 기능 | 설명 |
|------|------|
| 🔬 **다차원 원인 시뮬레이터** | C4A 발현, 시냅스 가지치기, 도파민 민감도, 글루타메이트 이상, GABA 불균형, 미세아교세포 활성도 등 7가지 요인을 실시간 슬라이더로 조절 |
| 🧠 **3D 뇌 회로 가시화** | Plotly 기반 3D 인터랙티브 그래프로 VTA-ventral_striatum, thalamus-sensory_cortex, hippocampus-PFC, ACC-PFC 등 4개 주요 PE 회로의 왜곡도를 시각화 |
| 📈 **비선형 상전이 모델** | 로지스틱 함수 기반 Phase Transition 모델을 도입하여, 위험 요인이 임계점을 돌파할 때 회로 왜곡이 급격히 증가하는 양상을 선형 모델과 비교 |
| 💊 **약물 효과 시뮬레이션** | D2 수용체 차단 항정신병약물의 회로 정상화 효과를 실시간으로 확인 |

---

## 🛠️ 기술 스택

- **Frontend/UI**: Streamlit
- **3D Visualization**: Plotly Graph Objects
- **Scientific Computing**: NumPy
- **Data**: JSON-based circuit mapping knowledge structure
- **Deployment**: Streamlit Community Cloud

---

## 🚀 실행 방법

### 로컬 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 외부 접속 (ngrok)

```bash
bash run_external.sh
```

---

## 📚 문헌 기반

- **Sekar et al. (2016)** — C4A 보체 경로 & 시냅스 가지치기
- **Millard (2021), Ermakova (2018)** — 도파민 살리언스 이론
- **White (2015)** — 글루타메이트 NMDA 저기능 가설
- **Sterzer (2018)** — 감각 PE 처리 장애
- **Larsen (2023), Rentzsch (2015)** — 청소년기 시냅스 가지치기
- **Yu (2023), Hartmann (2024)** — 신경면역 & 미세아교세포

---

## 📊 프로젝트 구조

```
.
├── .streamlit/                  # Streamlit 설정
├── streamlit_app.py             # 메인 애플리케이션 (1,100+ 라인)
├── schizophrenia_pe_circuit_map.json  # 회로 매핑 지식 구조
├── requirements.txt             # 의존성
└── run_external.sh              # 외부 접속 실행 스크립트
```

---

## ⚠️ 면책 조항

이 도구는 **진단 도구가 아닙니다**. 연구 및 교육 목적의 시각화 도구이며, 실제 진단 및 치료는 반드시 정신건강의학과 전문의와 상담하시기 바랍니다.

---

*Created as part of Brain-Mind-Behavior (BMB) coursework, Seoul National University, 2026*
