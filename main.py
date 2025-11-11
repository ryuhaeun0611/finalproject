import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="산불 피해 시뮬레이션", layout="wide")
st.title("🌲 다중 변수 기반 산불 피해 예측 시뮬레이션")

st.markdown("""
이 앱은 숲가꾸기 활동뿐 아니라 수종 구성, 수관 밀도, 경사 지형, 임분 밀도, 습도 등 다양한 변수가 산불 피해 면적에 어떤 영향을 줄 수 있는지를 시뮬레이션합니다.
사용자 입력에 따라 다중 회귀 및 랜덤포레스트 모델이 예측하며, SHAP 분석을 통해 변수의 영향력도 시각화합니다.
""")

# Sample data (가상 시뮬레이션용)
data = {
    'forest_care_area': [293, 279, 284, 306, 255, 211, 228, 209, 223, 226],
    'canopy_density': [0.85, 0.8, 0.83, 0.75, 0.7, 0.6, 0.62, 0.58, 0.65, 0.68],
    'species_type': [0.2, 0.3, 0.25, 0.4, 0.45, 0.6, 0.55, 0.65, 0.6, 0.5],
    'slope_degree': [15, 18, 12, 20, 25, 30, 28, 32, 35, 27],
    'stand_density': [0.7, 0.75, 0.72, 0.6, 0.55, 0.5, 0.52, 0.48, 0.5, 0.53],
    'humidity': [60, 58, 65, 55, 50, 40, 42, 38, 35, 45],
    'fire_damage_area': [137, 418, 378, 1480, 894, 3255, 2920, 766, 24797, 4992]
}
df = pd.DataFrame(data)

# Models
features = ['forest_care_area', 'canopy_density', 'species_type', 'slope_degree', 'stand_density', 'humidity']
X = df[features]
y = df['fire_damage_area']
reg_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(random_state=42).fit(X, y)

# Sidebar input
st.sidebar.header("🧪 시뮬레이션 입력")
forest_input = st.sidebar.slider("숲가꾸기 면적 (천 ha)", 200, 320, 250, step=5)
canopy_input = st.sidebar.slider("수관 밀도 (0~1)", 0.3, 1.0, 0.7, step=0.05)
species_input = st.sidebar.slider("침엽수 비율 (0~1)", 0.0, 1.0, 0.5, step=0.05)
slope_input = st.sidebar.slider("경사도 (도)", 0, 45, 20, step=1)
stand_input = st.sidebar.slider("임분 밀도 (0~1)", 0.2, 1.0, 0.6, step=0.05)
humidity_input = st.sidebar.slider("상대습도 (%)", 10, 100, 50, step=5)

# Prediction
input_array = np.array([[forest_input, canopy_input, species_input, slope_input, stand_input, humidity_input]])
reg_pred = reg_model.predict(input_array)[0]
rf_pred = rf_model.predict(input_array)[0]

# Results
st.subheader("📊 예측 결과")
st.markdown(f"**다중회귀 모델 예측 산불 피해 면적:** {reg_pred:.0f} ha")
st.markdown(f"**랜덤포레스트 모델 예측 산불 피해 면적:** {rf_pred:.0f} ha")

# Plot (2D scatter for partial view)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['forest_care_area'], y=df['fire_damage_area'],
    mode='markers', name='실제 데이터', marker=dict(size=8, color='green')
))
fig.add_trace(go.Scatter(
    x=[forest_input], y=[rf_pred], mode='markers', name='RF 예측점', marker=dict(size=10, color='red')
))
fig.update_layout(
    title='숲가꾸기 면적과 산불 피해 예측값 (기타 변수 고정)',
    xaxis_title='숲가꾸기 면적 (천 ha)',
    yaxis_title='예측 산불 피해 면적 (ha)'
)
st.plotly_chart(fig)

# SHAP 분석
st.subheader("🔎 변수 영향력 분석 (SHAP)")
explainer = shap.Explainer(rf_model, X)
shap_values = explainer(X)

# 요약 플롯 (전체 변수 중요도 시각화)
st.markdown("**전체 데이터 기반 변수 중요도 (summary plot)**")
fig_summary, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig_summary)

# 개별 입력값에 대한 SHAP 분석 (force plot 대체 텍스트 출력)
st.markdown("**현재 입력값에 대한 변수별 영향력:**")
sample_df = pd.DataFrame(input_array, columns=features)
sample_shap = explainer(sample_df)
shap_vals = sample_shap.values[0]

for i, f in enumerate(features):
    st.write(f"{f} → 영향력: {shap_vals[i]:+.2f}")

# Data
with st.expander("📂 원본 학습 데이터 보기"):
    st.dataframe(df)
