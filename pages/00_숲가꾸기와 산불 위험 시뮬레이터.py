import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.errors import EmptyDataError

plt.rcParams["axes.unicode_minus"] = False

THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "pages":
    ROOT_DIR = THIS_FILE.parents[1]
else:
    ROOT_DIR = THIS_FILE.parent

DATA_DIR = ROOT_DIR / "data"

st.set_page_config(page_title="숲가꾸기와 산불 위험 시뮬레이터", layout="wide")

st.title("🔥 숲가꾸기와 산불 위험 시뮬레이터")

st.markdown("""
이 페이지에서는 **숲가꾸기(조림·숲가꾸기 밀도)**, **수종 구성(침엽수 비율)**,  
**수관 밀도**, **지형(경사도)**, **임분 밀도**, **습도** 등이  
산불 **피해 면적(ha)** 에 어떤 방향으로 영향을 주는지 시뮬레이션합니다.

📌 본 시뮬레이터에 사용된 변수값 범위는 다음과 같은 실제 통계와 연구 기반 자료를 반영하여 구성되었습니다:
- **숲가꾸기/조림 밀도**: 0.10~0.70 (산림청)
- **수관 밀도**: 0.50~0.95 (국립산림과학원)
- **침엽수 비율**: 0.30~0.78 (통계청)
- **평균 경사도**: 5~45도 (국토지리정보원)
- **임분 밀도**: 0.40~0.82 (국립산림과학원)
- **상대습도**: 25~70% (기상청, 국립기상과학원)

이러한 값들을 바탕으로, 산불 발생과 피해 면적에 영향을 주는 **변수들의 상대적 방향성과 조합 효과**를 정량적으로 살펴볼 수 있습니다.
""")

# 1. 현실성 반영된 예시 데이터 생성 (기반 통계 반영)
n = 100
np.random.seed(42)

forest_care = np.random.uniform(0.10, 0.70, size=n)
canopy = np.random.uniform(0.50, 0.95, size=n)
conifer = np.random.uniform(0.30, 0.78, size=n)
slope = np.random.uniform(5, 45, size=n)
stand = np.random.uniform(0.40, 0.82, size=n)
humidity = np.random.uniform(25, 70, size=n)

# 하층습기 지수 = 습도 * (1 - 숲가꾸기 밀도)
understory_moisture = humidity * (1 - forest_care)

# 피해 면적 모델: 숲가꾸기 밀도가 높을수록 피해 ↑
fire_damage = (
    50
    + 120 * forest_care              # 가설 강조: 숲가꾸기 ↑ → 피해 ↑
    + 70 * canopy
    + 100 * conifer
    + 2.5 * slope
    + 70 * stand
    - 1.2 * humidity
    - 0.8 * understory_moisture      # 하층습기 ↓ → 피해 ↑
    + np.random.normal(0, 12, size=n)
)
fire_damage = np.round(np.clip(fire_damage, 0, None), 1)

data = {
    "forest_care_density": forest_care,
    "canopy_density": canopy,
    "conifer_ratio": conifer,
    "slope_degree": slope,
    "stand_density": stand,
    "humidity": humidity,
    "understory_moisture_index": understory_moisture,
    "fire_damage_area": fire_damage,
}

df = pd.DataFrame(data)

features = [
    "forest_care_density",
    "canopy_density",
    "conifer_ratio",
    "slope_degree",
    "stand_density",
    "humidity",
]

FEATURE_NAME_KO = {
    "forest_care_density": "숲가꾸기/조림 밀도",
    "canopy_density": "수관 밀도",
    "conifer_ratio": "침엽수 비율",
    "slope_degree": "평균 경사도",
    "stand_density": "임분 밀도",
    "humidity": "상대습도",
}

X = df[features]
y = df["fire_damage_area"]

reg_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=5).fit(X, y)

st.sidebar.header("🧪 시뮬레이션 입력값")

forest_input = st.sidebar.slider("숲가꾸기/조림 밀도 (0~0.8)", 0.05, 0.80, 0.40, step=0.01)
st.sidebar.markdown("""
<span style='font-size: 0.8rem; color: gray;'>
→ 조림 및 숲가꾸기 작업이 어느 정도 강도로 이루어졌는지를 나타내는 지표입니다.
</span>
""", unsafe_allow_html=True)

canopy_input = st.sidebar.slider("수관 밀도 (0~1)", 0.3, 1.0, 0.75, step=0.01)
st.sidebar.markdown("""
<span style='font-size: 0.8rem; color: gray;'>
→ 나무의 윗부분(수관)이 하늘을 얼마나 촘촘하게 가리고 있는지를 나타냅니다.
</span>
""", unsafe_allow_html=True)

species_input = st.sidebar.slider("침엽수 비율 (0~1)", 0.0, 1.0, 0.55, step=0.01)
slope_input = st.sidebar.slider("평균 경사도 (도)", 0, 45, 25, step=1)
stand_input = st.sidebar.slider("임분 밀도 (0~1)", 0.2, 1.0, 0.65, step=0.01)
st.sidebar.markdown("""
<span style='font-size: 0.8rem; color: gray;'>
→ 현재 숲에서 나무들이 얼마나 빽빽하게 서 있는지(혼잡도)를 나타내는 지표입니다.
</span>
""", unsafe_allow_html=True)

humidity_input = st.sidebar.slider("상대습도 (%)", 10, 100, 40, step=1)

input_array = np.array([[forest_input, canopy_input, species_input, slope_input, stand_input, humidity_input]])
reg_pred = reg_model.predict(input_array)[0]
rf_pred = rf_model.predict(input_array)[0]

st.subheader("📊 예측 결과")
c1, c2 = st.columns(2)
with c1:
    st.metric("다중회귀 예측 산불 피해 면적 (ha)", f"{reg_pred:,.0f}")
with c2:
    st.metric("랜덤포레스트 예측 산불 피해 면적 (ha)", f"{rf_pred:,.0f}")

st.caption("※ 이 값들은 실제 피해 면적이 아니라, 변수 간 **상대적인 영향 방향**을 보기 위한 가상의 예측값입니다.")

st.subheader("🌲 숲가꾸기/조림 밀도와 산불 피해 면적의 관계")

forest_range = np.linspace(df["forest_care_density"].min(), df["forest_care_density"].max(), 50)
X_line = pd.DataFrame({
    "forest_care_density": forest_range,
    "canopy_density": np.full_like(forest_range, df["canopy_density"].mean()),
    "conifer_ratio": np.full_like(forest_range, df["conifer_ratio"].mean()),
    "slope_degree": np.full_like(forest_range, df["slope_degree"].mean()),
    "stand_density": np.full_like(forest_range, df["stand_density"].mean()),
    "humidity": np.full_like(forest_range, df["humidity"].mean()),
})
line_pred = reg_model.predict(X_line)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["forest_care_density"], y=df["fire_damage_area"], mode="markers", name="예시 관측값", marker=dict(size=8)))
fig1.add_trace(go.Scatter(x=forest_range, y=line_pred, mode="lines", name="선형회귀 추세선", line=dict(width=3)))
fig1.add_trace(go.Scatter(x=[forest_input], y=[rf_pred], mode="markers", name="현재 시나리오 (RF 예측)", marker=dict(size=12, symbol="star")))
fig1.update_layout(title="숲가꾸기/조림 밀도 증가에 따른 산불 피해 면적 변화 (다른 변수 평균 고정)", xaxis_title="숲가꾸기/조림 밀도 (상대값)", yaxis_title="산불 피해 면적 (ha)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🔎 변수 영향력 분석 (SHAP)")
explainer = shap.Explainer(rf_model, X)
shap_values = explainer(X, check_additivity=False)

st.markdown("**① 전체 데이터에서 각 변수의 중요도 (막대 그래프)**")
shap_arr = shap_values.values
mean_abs_shap = np.abs(shap_arr).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)
sorted_importance = mean_abs_shap[sorted_idx]
sorted_features = [features[i] for i in sorted_idx]
sorted_features_ko = [FEATURE_NAME_KO[f] for f in sorted_features]

fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(x=sorted_importance, y=sorted_features_ko, orientation="h"))
fig_imp.update_layout(xaxis_title="평균 절대 SHAP 값 (모델 예측에 대한 평균 영향력)", yaxis_title="", margin=dict(l=120, r=20, t=20, b=40))
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("**② 현재 입력값에 대한 변수별 기여도**")
sample_df = pd.DataFrame(input_array, columns=features)
sample_shap = explainer(sample_df, check_additivity=False)
shap_vals = sample_shap.values[0]
for name, val in zip(features, shap_vals):
    ko_name = FEATURE_NAME_KO.get(name, name)
    st.write(f"- `{ko_name}` → 영향력: {val:+.2f}")

st.subheader("💧 숲가꾸기 → 하층식생 제거 → 하층 습기 감소 경로 보기")
understory_moisture_input = humidity_input * (1 - forest_input)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("현재 숲가꾸기/조림 밀도", f"{forest_input:.2f}")
with c2:
    st.metric("현재 상대습도(%)", f"{humidity_input:.0f}")
with c3:
    st.metric("추정 하층습기 지수", f"{understory_moisture_input:.1f}")

st.caption("※ 하층습기 지수 = `습도 × (1 − 숲가꾸기 밀도)`로 단순화한 지표입니다. 숲가꾸기 밀도가 높을수록(하층식생이 많이 제거될수록) 같은 습도에서도 지수가 낮아집니다.")

fig_moist = go.Figure()
fig_moist.add_trace(go.Scatter(x=df["understory_moisture_index"], y=df["fire_damage_area"], mode="markers", name="예시 관측값", marker=dict(size=8)))
fig_moist.add_trace(go.Scatter(x=[understory_moisture_input], y=[rf_pred], mode="markers", name="현재 시나리오", marker=dict(size=12, symbol="star")))
fig_moist.update_layout(title="하층습기 지수와 산불 피해 면적의 관계", xaxis_title="하층습기 지수 (습도 × (1 − 숲가꾸기/조림 밀도))", yaxis_title="산불 피해 면적 (ha)")
st.plotly_chart(fig_moist, use_container_width=True)

with st.expander("📂 시뮬레이션에 사용된 예시 데이터 보기"):
    st.dataframe(df)

with st.expander("📂 실제 연구 자료 CSV 열람하기"):
    def read_csv_safely(path: Path):
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return None
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding="cp949")
            except EmptyDataError:
                return None

    st.caption(f"현재 data 폴더 경로: `{DATA_DIR}`")

    csv_list = sorted(DATA_DIR.glob("*.csv"))
    if not csv_list:
        st.warning("data 폴더 안에 CSV 파일이 없습니다.")
    else:
        st.write("📁 data 폴더에 있는 CSV 파일들:")
        for p in csv_list:
            st.write(f"- `{p.name}`")

        st.markdown("---")
        for p in csv_list:
            with st.expander(f"📄 {p.name}"):
                df_src = read_csv_safely(p)
                if df_src is None:
                    st.warning("⚠ 이 파일은 내용이 없거나(빈 파일) CSV로 읽을 수 없습니다.")
                    continue
                st.dataframe(df_src)
                st.download_button(
                    label="⬇ CSV 다운로드",
                    data=df_src.to_csv(index=False),
                    file_name=p.name,
                    mime="text/csv",
                )
