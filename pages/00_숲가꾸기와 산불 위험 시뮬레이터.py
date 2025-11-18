import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------
# (옵션) 마이너스 깨짐 방지만 유지
# -------------------------------------------
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# -------------------------------------------
# 프로젝트 루트 / data 폴더 경로 계산
#   - 이 파일이 pages/ 안에 있어도, main.py 옆 data/를 찾도록 처리
# -------------------------------------------
THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "pages":
    ROOT_DIR = THIS_FILE.parents[1]   # .../finalproject
else:
    ROOT_DIR = THIS_FILE.parent       # .../finalproject

DATA_DIR = ROOT_DIR / "data"

# -------------------------------------------
# 페이지 설정
# -------------------------------------------
st.set_page_config(page_title="숲가꾸기와 산불 위험 시뮬레이터", layout="wide")

st.title("🔥 숲가꾸기와 산불 위험 시뮬레이터")

st.markdown("""
이 페이지에서는 **숲가꾸기(조림·숲가꾸기 밀도)**, **수종 구성(침엽수 비율)**,  
**수관 밀도**, **지형(경사도)**, **임분 밀도**, **습도** 등이  
산불 **피해 면적(ha)** 에 어떤 방향으로 영향을 주는지 시뮬레이션합니다.

업로드한 논문·통계자료에서 공통적으로 나타난 경향을 반영하여,

- 숲가꾸기/조림 밀도 ↑ → 산불 피해 **증가**  
- 침엽수 비율·임분 밀도 ↑ → 산불 피해 **증가**  
- 습도 ↑ → 산불 피해 **감소**  

가 되도록 예시 데이터를 구성했습니다.
""")

st.info(
    "※ 수치는 실제 피해 규모를 그대로 반영하는 것이 아니라, "
    "연구 결과의 **‘방향성’**을 시각적으로 이해하기 위한 가상의 데이터입니다."
)

# -------------------------------------------
# 1. 예시 데이터 (연구결과를 반영한 가상 데이터)
# -------------------------------------------
data = {
    "forest_care_density": [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35,
                            0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70],
    "canopy_density":      [0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                            0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95],
    "conifer_ratio":       [0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55,
                            0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.78],
    "slope_degree":        [5, 7, 10, 12, 15, 18, 20, 22, 24, 26,
                            28, 30, 32, 34, 36, 38, 40, 42, 43, 45],
    "stand_density":       [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62,
                            0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
    "humidity":            [70, 68, 72, 65, 63, 60, 58, 55, 53, 50,
                            48, 45, 43, 40, 38, 35, 33, 30, 28, 25],
}

df = pd.DataFrame(data)

# 피해 면적 생성 (방향성만 반영)
rng = np.random.default_rng(42)
base = (
    500
    + 9000 * df["forest_care_density"]   # 숲가꾸기/조림 밀도 ↑ → 피해 ↑
    + 8000 * df["conifer_ratio"]         # 침엽수 비율 ↑ → 피해 ↑
    + 6000 * df["stand_density"]         # 임분 밀도 ↑ → 피해 ↑
    + 20   * df["slope_degree"]          # 경사도 ↑ → 약한 피해 ↑
    - 30   * df["humidity"]              # 습도 ↑ → 피해 ↓
)
noise = rng.normal(0, 400, size=len(df))
df["fire_damage_area"] = np.clip(base + noise, 50, None)

# -------------------------------------------
# 2. 특징 / 타깃 및 한글 매핑
# -------------------------------------------
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
rf_model = RandomForestRegressor(
    random_state=42,
    n_estimators=300,
    max_depth=5,
).fit(X, y)

# -------------------------------------------
# 3. 사이드바 입력
# -------------------------------------------
st.sidebar.header("🧪 시뮬레이션 입력값")

forest_input = st.sidebar.slider("숲가꾸기/조림 밀도 (0~0.8)", 0.05, 0.80, 0.40, step=0.01)
st.sidebar.markdown(
    "<span style='font-size: 0.8rem; color: gray;'>"
    "→ 조림 및 숲가꾸기 작업이 어느 정도 강도로 이루어졌는지를 나타내는 지표입니다.**"
    "</span>",
    unsafe_allow_html=True,
)

canopy_input = st.sidebar.slider("수관 밀도 (0~1)", 0.3, 1.0, 0.75, step=0.01)
st.sidebar.markdown(
    "<span style='font-size: 0.8rem; color: gray;'>"
    "→ 나무의 윗부분(수관)이 하늘을 얼마나 촘촘하게 가리고 있는지를 나타냅니다."
    "</span>",
    unsafe_allow_html=True,
)

species_input = st.sidebar.slider("침엽수 비율 (0~1)", 0.0, 1.0, 0.55, step=0.01)

slope_input = st.sidebar.slider("평균 경사도 (도)", 0, 45, 25, step=1)

stand_input = st.sidebar.slider("임분 밀도 (0~1)", 0.2, 1.0, 0.65, step=0.01)
st.sidebar.markdown(
    "<span style='font-size: 0.8rem; color: gray;'>"
    "→ 현재 숲에서 나무들이 얼마나 빽빽하게 서 있는지(혼잡도)를 나타내는 지표입니다."
    "</span>",
    unsafe_allow_html=True,
)

humidity_input = st.sidebar.slider("상대습도 (%)", 10, 100, 40, step=1)

input_array = np.array(
    [[forest_input, canopy_input, species_input, slope_input, stand_input, humidity_input]]
)

reg_pred = reg_model.predict(input_array)[0]
rf_pred = rf_model.predict(input_array)[0]

# -------------------------------------------
# 4. 예측 결과
# -------------------------------------------
st.subheader("📊 예측 결과")

c1, c2 = st.columns(2)
with c1:
    st.metric("다중회귀 예측 산불 피해 면적 (ha)", f"{reg_pred:,.0f}")
with c2:
    st.metric("랜덤포레스트 예측 산불 피해 면적 (ha)", f"{rf_pred:,.0f}")

st.caption(
    "※ 이 값들은 실제 피해 면적이 아니라, 변수 간 **상대적인 영향 방향**을 보기 위한 가상의 예측값입니다."
)

# -------------------------------------------
# 5. 숲가꾸기/조림 밀도 vs 피해 면적
# -------------------------------------------
st.subheader("🌲 숲가꾸기/조림 밀도와 산불 피해 면적의 관계")

forest_range = np.linspace(df["forest_care_density"].min(),
                           df["forest_care_density"].max(), 50)

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
fig1.add_trace(go.Scatter(
    x=df["forest_care_density"],
    y=df["fire_damage_area"],
    mode="markers",
    name="예시 관측값",
    marker=dict(size=8),
))
fig1.add_trace(go.Scatter(
    x=forest_range,
    y=line_pred,
    mode="lines",
    name="선형회귀 추세선",
    line=dict(width=3),
))
fig1.add_trace(go.Scatter(
    x=[forest_input],
    y=[rf_pred],
    mode="markers",
    name="현재 시나리오 (RF 예측)",
    marker=dict(size=12, symbol="star"),
))

fig1.update_layout(
    title="숲가꾸기/조림 밀도 증가에 따른 산불 피해 면적 변화 (다른 변수 평균 고정)",
    xaxis_title="숲가꾸기/조림 밀도 (상대값)",
    yaxis_title="산불 피해 면적 (ha)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
그래프에서 **숲가꾸기/조림 밀도(가로축)가 커질수록 회귀선과 예시 데이터 점들이 위쪽(피해 증가)으로 이동**하는지를 보면,  
논문에서 말하는 **“숲가꾸기 활동이 오히려 산불 피해를 키울 수 있다”**는 방향성을 직관적으로 확인할 수 있습니다.
""")

# -------------------------------------------
# 6. SHAP 변수 영향력 분석 – Plotly 막대 그래프(완전 한글)
# -------------------------------------------
st.subheader("🔎 변수 영향력 분석 (SHAP)")

explainer = shap.Explainer(rf_model, X)
shap_values = explainer(X, check_additivity=False)

st.markdown("**① 전체 데이터에서 각 변수의 중요도 (막대 그래프)**")

# SHAP 값으로부터 변수 중요도(평균 절대값) 계산
shap_arr = shap_values.values  # (샘플 수, 변수 수)
mean_abs_shap = np.abs(shap_arr).mean(axis=0)

# 중요도 작은 것 → 큰 것 순으로 정렬 (가장 큰 값이 y축 맨 위로 가게 하기 위해)
sorted_idx = np.argsort(mean_abs_shap)  # 오름차순
sorted_importance = mean_abs_shap[sorted_idx]
sorted_features = [features[i] for i in sorted_idx]
sorted_features_ko = [FEATURE_NAME_KO[f] for f in sorted_features]

# Plotly 막대 그래프 (한국어, 중요도 큰 순이 위쪽)
fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    x=sorted_importance,
    y=sorted_features_ko,
    orientation="h"
))
fig_imp.update_layout(
    xaxis_title="평균 절대 SHAP 값 (모델 예측에 대한 평균 영향력)",
    yaxis_title="",  # '변수 이름' 문구 제거
    margin=dict(l=120, r=20, t=20, b=40),
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("**② 현재 입력값에 대한 변수별 기여도**")

sample_df = pd.DataFrame(input_array, columns=features)
sample_shap = explainer(sample_df, check_additivity=False)
shap_vals = sample_shap.values[0]

for name, val in zip(features, shap_vals):
    ko_name = FEATURE_NAME_KO.get(name, name)
    st.write(f"- `{ko_name}` → 영향력: {val:+.2f}")

# -------------------------------------------
# 7. 하층습기 지수(understory moisture) 시각화
# -------------------------------------------
st.subheader("💧 숲가꾸기 → 하층식생 제거 → 하층 습기 감소 경로 보기")

df["understory_moisture_index"] = df["humidity"] * (1 - df["forest_care_density"])
understory_moisture_input = humidity_input * (1 - forest_input)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("현재 숲가꾸기/조림 밀도", f"{forest_input:.2f}")
with c2:
    st.metric("현재 상대습도(%)", f"{humidity_input:.0f}")
with c3:
    st.metric("추정 하층습기 지수", f"{understory_moisture_input:.1f}")

st.caption(
    "※ 하층습기 지수 = `습도 × (1 − 숲가꾸기 밀도)`로 단순화한 지표입니다. "
    "숲가꾸기 밀도가 높을수록(하층식생이 많이 제거될수록) 같은 습도에서도 지수가 낮아집니다."
)

fig_moist = go.Figure()
fig_moist.add_trace(go.Scatter(
    x=df["understory_moisture_index"],
    y=df["fire_damage_area"],
    mode="markers",
    name="예시 관측값",
    marker=dict(size=8),
))
fig_moist.add_trace(go.Scatter(
    x=[understory_moisture_input],
    y=[rf_pred],
    mode="markers",
    name="현재 시나리오",
    marker=dict(size=12, symbol="star"),
))
fig_moist.update_layout(
    title="하층습기 지수와 산불 피해 면적의 관계",
    xaxis_title="하층습기 지수 (습도 × (1 − 숲가꾸기/조림 밀도))",
    yaxis_title="산불 피해 면적 (ha)",
)
st.plotly_chart(fig_moist, use_container_width=True)

# -------------------------------------------
# 8. 데이터 표 – 예시 데이터 + 실제 CSV
# -------------------------------------------

# 8-1) 예시 데이터
with st.expander("📂 시뮬레이션에 사용된 예시 데이터 보기"):
    st.dataframe(df)

# 8-2) 실제 연구 자료 CSV 열람하기
with st.expander("📂 실제 연구 자료 CSV 열람하기"):

    CSV_FILES = {
        "과거 10년간 산불통계 (연도별)": "과거 10년간 산불통계_연도.csv",
        "과거 10년간 산불통계 (지역별)": "과거 10년간 산불통계_지역.csv",
        "산림청 국유림경영정보 (산림조사)": "산림청_국유림경영정보_산림조사.csv",
        "산림청 임도시설 현황": "산림청_임도시설 현황.csv",
    }

    def read_csv_safely(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp949")

    st.write(f"🔎 현재 data 폴더 경로: `{DATA_DIR}`")

    for title, filename in CSV_FILES.items():
        file_path = DATA_DIR / filename

        with st.expander(f"📄 {title}"):
            if file_path.exists():
                df_src = read_csv_safely(file_path)
                st.dataframe(df_src)

                st.download_button(
                    label="⬇ CSV 다운로드",
                    data=df_src.to_csv(index=False),
                    file_name=filename,
                    mime="text/csv",
                )
            else:
                st.warning(f"⚠ `{filename}` 파일을 data 폴더에서 찾을 수 없습니다.")
