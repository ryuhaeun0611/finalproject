import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(page_title="산불 피해 시뮬레이션", layout="wide")
st.title("🌲 다중 변수 기반 산불 피해 예측 시뮬레이션")

st.markdown("""
이 앱은 숲가꾸기 활동뿐 아니라 수종 구성, 수관 밀도, 경사 지형 등 다양한 변수가 산불 피해 면적에 어떤 영향을 줄 수 있는지를 시뮬레이션합니다.
사용자 입력에 따라 다중 회귀 모델이 산불 피해를 예측합니다.
""")

# Sample data (가상 시뮬레이션용)
data = {
    'forest_care_area': [293, 279, 284, 306, 255, 211, 228, 209, 223, 226],  # 천 ha
    'canopy_density': [0.85, 0.8, 0.83, 0.75, 0.7, 0.6, 0.62, 0.58, 0.65, 0.68],  # 0~1
    'species_type': [0.2, 0.3, 0.25, 0.4, 0.45, 0.6, 0.55, 0.65, 0.6, 0.5],  # 침엽수 비율 (0~1)
    'slope_degree': [15, 18, 12, 20, 25, 30, 28, 32, 35, 27],  # 경사도 (°)
    'fire_damage_area': [137, 418, 378, 1480, 894, 3255, 2920, 766, 24797, 4992]  # ha
}
df = pd.DataFrame(data)

# Models
features = ['forest_care_area', 'canopy_density', 'species_type', 'slope_degree']
X = df[features]
y = df['fire_damage_area']
reg_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(random_state=42).fit(X, y)

# Sidebar input
st.sidebar.header("🧪 시뮬레이션 입력")
forest_input = st.sidebar.slider("숲가꾸기 면적 (천 ha)", 200, 320, 250, step=5)
canopy_input = st.sidebar.slider("수관 밀도 (0=없음 ~ 1=완전)", 0.3, 1.0, 0.7, step=0.05)
species_input = st.sidebar.slider("침엽수 비율 (0=전부 활엽수 ~ 1=전부 침엽수)", 0.0, 1.0, 0.5, step=0.05)
slope_input = st.sidebar.slider("경사도 (도)", 0, 45, 20, step=1)

# Prediction
input_array = np.array([[forest_input, canopy_input, species_input, slope_input]])
reg_pred = reg_model.predict(input_array)[0]
rf_pred = rf_model.predict(input_array)[0]

# Results
st.subheader("📊 예측 결과")
st.markdown(f"**다중회귀 모델 예측 산불 피해 면적:** {reg_pred:.0f} ha")
st.markdown(f"**랜덤포레스트 모델 예측 산불 피해 면적:** {rf_pred:.0f} ha")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=df['forest_care_area'], y=df['species_type'], z=df['fire_damage_area'],
    mode='markers', name='실제 데이터', marker=dict(size=5, color='green')
))
fig.add_trace(go.Scatter3d(
    x=[forest_input], y=[species_input], z=[rf_pred],
    mode='markers', name='RF 예측점', marker=dict(size=8, color='red')
))
fig.update_layout(scene=dict(
    xaxis_title='숲가꾸기 면적',
    yaxis_title='침엽수 비율',
    zaxis_title='예측 산불 피해(ha)'
), title='입력값 기반 산불 피해 예측')
st.plotly_chart(fig)

# Data
with st.expander("📂 원본 학습 데이터 보기"):
    st.dataframe(df)
