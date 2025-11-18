import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 🔧 주요 개선 사항
# 1) 예시 데이터 구조를 논문 결과에 맞춰 재구성
#    - forest_care_density(숲가꾸기/조림 밀도) ↑ → fire_damage_area ↑
#    - conifer_ratio(침엽수 비율), stand_density(임분 밀도) ↑ → 피해 증가
#    - humidity(상대습도) ↑ → 피해 감소
# 2) "숲가꾸기 활동이 많을수록 피해가 커진다"를
#    - 산점도 + 회귀 직선 + 사용자의 현재 입력점 으로 시각적으로 강조
# 3) SHAP summary plot과 함께
#    - forest_care_density의 평균적 중요도가 높게 나오도록 데이터 구성
# 4) 임도 접근성 변수는 연구결과상 영향이 없으므로 모델에서 제외(설명만 언급)
# -------------------------------------------------------------------

st.set_page_config(page_title="숲가꾸기 기반 산불 위험 분석", layout="wide")
st.title("🌲 숲가꾸기 기반 산불 위험 분석")

st.markdown("""
이 앱은 **숲가꾸기(조림·숲가꾸기 면적/밀도)**, **수종 구성(침엽수 비율)**,  
**수관 밀도**, **지형(경사도)**, **임분 밀도**, **습도** 등이  
산불 **피해 면적(ha)** 에 어떤 영향을 줄 수 있는지 시뮬레이션합니다.

업로드한 논문·통계자료에서 공통적으로 나타난 경향을 반영하여,

- 숲가꾸기/조림 밀도가 높을수록 → 산불 피해가 **증가**하고  
- 침엽수 비율·임분 밀도가 높을수록 → 산불 피해가 **증가**하며  
- 습도가 높을수록 → 산불 피해가 **감소**하는

방향으로 예시 데이터를 구성하였습니다.
""")

st.info(
    "※ 임도 접근성 변수는 경북 분석 및 강릉 난곡 사례 연구에서 "
    "`산불 발생·피해와 통계적으로 유의한 관련이 없음`으로 나타나, "
    "본 시뮬레이션 모델에서는 제외하고 설명만 제공합니다."
)

# -------------------------------------------------------------------
# 1. 예시 데이터 (연구결과를 반영한 가상 데이터)
#    - 값 자체는 가상이나, 변수 간 관계는 논문 결과와 일관되게 구성
# -------------------------------------------------------------------
data = {
    # 숲가꾸기/조림 밀도 (예: ha/㎢ 또는 0~1 스케일)
    'forest_care_density': [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35,
                            0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70],
    # 수관밀도 (0~1) – 너무 낮거나 너무 높으면 피해가 커지는 패턴을 일부 반영
    'canopy_density':      [0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                            0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95],
    # 침엽수 비율 (0~1)
    'conifer_ratio':       [0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55,
                            0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.78],
    # 평균 경사도 (도)
    'slope_degree':        [5, 7, 10, 12, 15, 18, 20, 22, 24, 26,
                            28, 30, 32, 34, 36, 38, 40, 42, 43, 45],
    # 임분 밀도 (0~1)
    'stand_density':       [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62,
                            0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
    # 상대습도(%)
    'humidity':            [70, 68, 72, 65, 63, 60, 58, 55, 53, 50,
                            48, 45, 43, 40, 38, 35, 33, 30, 28, 25],
}

df = pd.DataFrame(data)

# 피해 면적 생성: 연구 결과 방향성을 반영한 가상의 함수 + 약간의 노이즈
# (숲가꾸기 밀도, 침엽수 비율, 임분 밀도는 +, 습도는 -)
rng = np.random.default_rng(42)
base = (
    500
    + 9000 * df['forest_care_density']      # 숲가꾸기 효과 (양의 영향)
    + 8000 * df['conifer_ratio']            # 침엽수 비율 (양의 영향)
    + 6000 * df['stand_density']            # 임분 밀도 (양의 영향)
    + 20   * df['slope_degree']             # 경사도 (완만한 양의 영향)
    - 30   * df['humidity']                 # 습도 (음의 영향)
)
noise = rng.normal(0, 400, size=len(df))
df['fire_damage_area'] = np.clip(base + noise, 50, None)  # 최소 50ha 이상으로 clip

# -------------------------------------------------------------------
# 2. 특징 및 타겟 설정
# -------------------------------------------------------------------
features = ['forest_care_density', 'canopy_density', 'conifer_ratio',
            'slope_degree', 'stand_density', 'humidity']
X = df[features]
y = df['fire_damage_area']

# 모델 학습
reg_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(
    random_state=42,
    n_estimators=300,
    max_depth=5
).fit(X, y)

# -------------------------------------------------------------------
# 3. 사이드바 입력 (사용자가 가정하는 산림 상태)
# -------------------------------------------------------------------
st.sidebar.header("🧪 시뮬레이션 입력")

forest_input = st.sidebar.slider("숲가꾸기/조림 밀도 (0~0.8)", 0.05, 0.80, 0.40, step=0.01)
canopy_input = st.sidebar.slider("수관 밀도 (0~1)", 0.3, 1.0, 0.75, step=0.01)
species_input = st.sidebar.slider("침엽수 비율 (0~1)", 0.0, 1.0, 0.55, step=0.01)
slope_input = st.sidebar.slider("평균 경사도 (도)", 0, 45, 25, step=1)
stand_input = st.sidebar.slider("임분 밀도 (0~1)", 0.2, 1.0, 0.65, step=0.01)
humidity_input = st.sidebar.slider("상대습도 (%)", 10, 100, 40, step=1)

# 입력 배열 생성
input_array = np.array([[forest_input, canopy_input, species_input,
                         slope_input, stand_input, humidity_input]])

# 예측
reg_pred = reg_model.predict(input_array)[0]
rf_pred = rf_model.predict(input_array)[0]

# -------------------------------------------------------------------
# 4. 예측 결과 출력
# -------------------------------------------------------------------
st.subheader("📊 예측 결과")
col1, col2 = st.columns(2)
with col1:
    st.metric("다중회귀 예측 산불 피해 면적 (ha)", f"{reg_pred:,.0f}")
with col2:
    st.metric("랜덤포레스트 예측 산불 피해 면적 (ha)", f"{rf_pred:,.0f}")

st.caption(
    "※ 수치는 연구결과의 '방향성'을 시뮬레이션하기 위한 가상의 값이며, "
    "실제 피해 규모를 그대로 반영하는 것은 아닙니다."
)

# -------------------------------------------------------------------
# 5. 시각화 1: 숲가꾸기 활동이 많을수록 피해가 커지는지 확인
#    - 실제(예시) 데이터 + 회귀 직선 + 사용자 입력점
# -------------------------------------------------------------------
st.subheader("🌲 숲가꾸기/조림 밀도와 산불 피해 면적의 관계")

# 회귀 직선을 그리기 위해 forest_care만 변화시키고 나머지는 평균값으로 고정
forest_range = np.linspace(df['forest_care_density'].min(),
                           df['forest_care_density'].max(), 50)
X_line = pd.DataFrame({
    'forest_care_density': forest_range,
    'canopy_density': np.full_like(forest_range, df['canopy_density'].mean()),
    'conifer_ratio': np.full_like(forest_range, df['conifer_ratio'].mean()),
    'slope_degree': np.full_like(forest_range, df['slope_degree'].mean()),
    'stand_density': np.full_like(forest_range, df['stand_density'].mean()),
    'humidity': np.full_like(forest_range, df['humidity'].mean()),
})
line_pred = reg_model.predict(X_line)

fig1 = go.Figure()

# ① 예시 데이터 산점도
fig1.add_trace(go.Scatter(
    x=df['forest_care_density'],
    y=df['fire_damage_area'],
    mode='markers',
    name='예시 관측값',
    marker=dict(size=8)
))

# ② 회귀 직선
fig1.add_trace(go.Scatter(
    x=forest_range,
    y=line_pred,
    mode='lines',
    name='선형회귀 추세선',
    line=dict(width=3, dash='solid')
))

# ③ 사용자의 현재 입력점
fig1.add_trace(go.Scatter(
    x=[forest_input],
    y=[rf_pred],
    mode='markers',
    name='현재 시나리오(RF 예측)',
    marker=dict(size=12, symbol='star')
))

fig1.update_layout(
    title="숲가꾸기/조림 밀도 증가에 따른 산불 피해 면적 변화 (다른 변수 평균 고정)",
    xaxis_title="숲가꾸기/조림 밀도 (0~0.8, 상대값)",
    yaxis_title="산불 피해 면적 (ha)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
위 그래프에서 **숲가꾸기/조림 밀도가 오른쪽(0.7 근처)으로 갈수록 회귀 직선이 뚜렷하게 상승**하는 것을 볼 수 있습니다.  
이는 업로드된 연구 결과와 마찬가지로 **“숲가꾸기 활동이 집중된 지역일수록 산불 피해가 커지는 경향”**을
시각적으로 확인할 수 있게 해 줍니다.
""")

# -------------------------------------------------------------------
# 6. 시각화 2: 다른 변수들을 포함한 SHAP 분석
# -------------------------------------------------------------------
st.subheader("🔎 변수 영향력 분석 (SHAP)")

# Tree 기반 모델이므로 TreeExplainer 사용
explainer = shap.Explainer(rf_model, X)
# additivity 체크 끄기
shap_values = explainer(X, check_additivity=False)

st.markdown("**(1) 전체 데이터 기준 변수 중요도 – forest_care_density가 상단에 오는지 확인해 보세요.**")

fig_summary, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig_summary)

st.markdown("**(2) 현재 입력값에 대한 변수별 기여도**")

sample_df = pd.DataFrame(input_array, columns=features)
sample_shap = explainer(sample_df, check_additivity=False)
shap_vals = sample_shap.values[0]

for i, f in enumerate(features):
    st.write(f"- {f}: 영향력 {shap_vals[i]:+.2f}")
    
# =========================
# (A) 하층습기 지수 계산
# =========================
df["understory_moisture_index"] = df["humidity"] * (1 - df["forest_care_density"])

understory_moisture_input = humidity_input * (1 - forest_input)

st.subheader("💧 숲가꾸기 → 하층식생 제거 → 습기 감소 경로 보기")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("현재 시나리오 숲가꾸기/조림 밀도", f"{forest_input:.2f}")
with c2:
    st.metric("현재 시나리오 상대습도(%)", f"{humidity_input:.0f}")
with c3:
    st.metric("추정 하층습기 지수", f"{understory_moisture_input:.1f}")

st.caption(
    "※ 하층습기 지수 = 습도 × (1 − 숲가꾸기 밀도)로 단순화한 지표입니다.\n"
    "숲가꾸기 밀도가 높을수록(하층식생이 많이 제거될수록) 같은 습도에서도 지수가 낮아집니다."
)

# (B) 하층습기 지수와 산불 피해 관계 시각화
fig_moist = go.Figure()
fig_moist.add_trace(go.Scatter(
    x=df["understory_moisture_index"],
    y=df["fire_damage_area"],
    mode="markers",
    name="예시 관측값",
    marker=dict(size=8)
))
fig_moist.add_trace(go.Scatter(
    x=[understory_moisture_input],
    y=[rf_pred],
    mode="markers",
    name="현재 시나리오",
    marker=dict(size=12, symbol="star")
))
fig_moist.update_layout(
    title="하층습기 지수와 산불 피해 면적의 관계",
    xaxis_title="하층습기 지수 (humidity × (1 − forest_care_density))",
    yaxis_title="산불 피해 면적 (ha)"
)
st.plotly_chart(fig_moist, use_container_width=True)

st.markdown("""
위 그래프에서 **하층습기 지수가 낮을수록(오른쪽이 아니라 왼쪽 방향)**  
예시 데이터들이 더 큰 피해 면적 쪽에 몰려 있다면,

> 숲가꾸기 → 활엽수 하층식생 제거 → 숲의 습기 감소 → 산불 피해 증가

라는 연구 결과를 직관적으로 보여주는 시각화가 됩니다.
""")

# -------------------------------------------------------------------
# 7. 원본(예시) 데이터 테이블
# -------------------------------------------------------------------
with st.expander("📂 학습에 사용된 예시 데이터 보기"):
    st.dataframe(df.style.format({"fire_damage_area": "{:.1f}"}))

# -------------------------------------------------------------------
# 8. 부가 설명
# -------------------------------------------------------------------
st.markdown("""
### 📌 해석 가이드

- 그래프와 SHAP 결과에서 **`forest_care_density`의 기울기와 중요도가 크고 양(+)의 방향**으로 나타나면  
  → *현재의 숲가꾸기·조림 방식이 산불 피해를 줄이기보다는 늘리는 쪽으로 작용할 수 있다*는 연구 결과와 일치합니다.

- 반대로 **humidity(습도)**는 음(-)의 기여를 보여야 하며,  
  **conifer_ratio(침엽수 비율)**, **stand_density(임분 밀도)**는 양(+)의 방향으로 나타나는지 확인해 보세요.
""")

