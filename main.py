import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(page_title="산불 피해 시뮬레이션", layout="wide")
st.title("🌲 숲가꾸기 활동이 산불 피해에 미치는 영향 시뮬레이션")

st.markdown("""
이 앱은 숲가꾸기 활동이 산불 피해 면적에 어떤 영향을 줄 수 있는지를 시뮬레이션합니다. 아래 슬라이더를 통해 숲가꾸기 면적을 조정하며 산불 피해 예상량 변화를 살펴보세요.
또한, 실제 통계 데이터를 기반으로 선형회귀 및 랜덤포레스트 회귀모형을 통해 시나리오 결과를 비교합니다.
""")

# Sample data (real values from 2014–2023 for simulation)
data = {
    'year': np.arange(2014, 2024),
    'forest_care_area': [293, 279, 284, 306, 255, 211, 228, 209, 223, 226],  # in thousand ha
    'fire_damage_area': [137, 418, 378, 1480, 894, 3255, 2920, 766, 24797, 4992]  # in ha
}
df = pd.DataFrame(data)

# Models
X = df[['forest_care_area']]
y = df['fire_damage_area']
lin_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(random_state=42).fit(X, y)

# Sidebar: User input for forest care scenario
st.sidebar.header("시뮬레이션 설정")
user_input = st.sidebar.slider(
    "시뮬레이션할 숲가꾸기 면적 (천 ha)",
    min_value=200,
    max_value=320,
    value=250,
    step=5
)

# Prediction
input_array = np.array([[user_input]])
lin_pred = lin_model.predict(input_array)[0]
rf_pred = rf_model.predict(input_array)[0]

# Display prediction results
st.subheader("📊 예측 결과")
st.markdown(f"**선형 회귀 모델 예측 산불 피해 면적:** {lin_pred:.0f} ha")
st.markdown(f"**랜덤포레스트 모델 예측 산불 피해 면적:** {rf_pred:.0f} ha")

# Plot with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['forest_care_area'], y=df['fire_damage_area'],
                         mode='markers', name='실제 데이터', marker=dict(color='green')))
fig.add_trace(go.Scatter(x=df['forest_care_area'], y=lin_model.predict(X),
                         mode='lines', name='선형 회귀선', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=[user_input], y=[lin_pred], mode='markers', name='선형 예측', marker=dict(color='blue', size=10)))
fig.add_trace(go.Scatter(x=[user_input], y=[rf_pred], mode='markers', name='RF 예측', marker=dict(color='red', size=10)))
fig.add_vline(x=user_input, line=dict(dash='dash', color='gray'), annotation_text='시뮬레이션 입력', annotation_position="top")
fig.update_layout(title="숲가꾸기 면적 vs 산불 피해 면적",
                  xaxis_title="숲가꾸기 면적 (천 ha)",
                  yaxis_title="산불 피해 면적 (ha)")
st.plotly_chart(fig, use_container_width=True)

# Optional: Display data
with st.expander("📂 원본 데이터 보기"):
    st.dataframe(df.style.format({"forest_care_area": "{:.0f}", "fire_damage_area": "{:.0f}"}))
