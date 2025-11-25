from sklearn.metrics import mean_absolute_error

# 예측값 생성
reg_pred_all = reg_model.predict(X)
rf_pred_all = rf_model.predict(X)

# MAE 계산
mae_reg = mean_absolute_error(y, reg_pred_all)
mae_rf = mean_absolute_error(y, rf_pred_all)

st.subheader("📉 모델 성능 평가 (MAE)")
st.write(f"**다중회귀 MAE:** {mae_reg:.2f} ha")
st.write(f"**랜덤포레스트 MAE:** {mae_rf:.2f} ha")
