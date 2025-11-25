import streamlit as st
from sklearn.metrics import mean_absolute_error

st.title("📉 평균절대오차(MAE) 계산")

# 모델과 데이터가 Session State 에 있는지 확인
if "reg_model" not in st.session_state:
    st.error("❗ 메인 페이지에서 먼저 모델을 학습시켜 주세요.")
    st.stop()

reg_model = st.session_state["reg_model"]
rf_model = st.session_state["rf_model"]
X = st.session_state["X"]
y = st.session_state["y"]

# 예측
reg_pred_all = reg_model.predict(X)
rf_pred_all = rf_model.predict(X)

# MAE 계산
mae_reg = mean_absolute_error(y, reg_pred_all)
mae_rf = mean_absolute_error(y, rf_pred_all)

# 출력
st.subheader("📊 모델 성능 (MAE)")
st.write(f"**다중회귀 MAE:** {mae_reg:.2f} ha")
st.write(f"**랜덤포레스트 MAE:** {mae_rf:.2f} ha")
