import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# ---------------------------------------------------------
# 0. 경로 설정 (pages/ 안에서도 잘 작동하도록)
# ---------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "pages":
    ROOT_DIR = THIS_FILE.parents[1]   # .../finalproject
else:
    ROOT_DIR = THIS_FILE.parent

OUTPUT_DIR = ROOT_DIR / "output"
DATA_PATH = OUTPUT_DIR / "fire_region_with_roads.csv"

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="실제 데이터 기반 산불-임도 모델",
    layout="wide"
)

st.title("📈 실제 CSV 기반 산불 피해 예측 모델")

st.markdown("""
이 페이지는 전처리된 **실제 공공 데이터**  
(`output/fire_region_with_roads.csv`)를 사용하여,

- 지역별 **산불 피해 면적(damage_ha)** 을  
- **발생 건수(fire_count)** 와 **임도 연장(road_length)** 으로 예측하는  
**기본 회귀 모델(선형회귀 + 랜덤포레스트)** 을 학습하고 평가합니다.
""")

st.markdown("---")

# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"`{DATA_PATH}` 파일을 찾을 수 없습니다. 전처리 스크립트를 먼저 실행해 주세요.")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

df_raw = load_data(DATA_PATH)

st.subheader("📂 전처리된 실제 데이터 미리보기")

st.caption(f"파일 위치: `{DATA_PATH}`")
st.dataframe(df_raw.head())

st.markdown("""
- `region_name` : 지역 이름  
- `year` : 연도 (있을 경우)  
- `fire_count` : 산불 발생 건수  
- `damage_ha` : 산불 피해 면적(ha)  
- `road_length` : 임도 연장 길이 (km 등, 원자료 단위에 따름)
""")

st.markdown("---")

# ---------------------------------------------------------
# 3. 모델링용 데이터 준비
# ---------------------------------------------------------
st.subheader("🧹 모델링용 데이터 정리")

df = df_raw.copy()

# 필수 컬럼 체크
required_cols = ["fire_count", "damage_ha"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"모델 학습에 필요한 컬럼이 없습니다: {missing}")
    st.stop()

# road_length는 없을 수도 있으므로 옵션
has_road = "road_length" in df.columns

# 숫자형으로 강제 변환
for col in ["fire_count", "damage_ha"] + (["road_length"] if has_road else []):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 유효한 데이터만 사용: 피해 면적과 발생 건수 둘 다 0 이상 & NaN 제거
cond = df["damage_ha"].notna() & df["fire_count"].notna()
if has_road:
    cond = cond & df["road_length"].notna()

df_model = df.loc[cond].copy()

st.write(f"✅ 모델링에 사용할 행 수: {len(df_model)} / 원본 {len(df)}")

if len(df_model) < 10:
    st.warning("모델을 학습하기에 데이터가 너무 적을 수 있습니다. 결과를 해석할 때 주의하세요.")

with st.expander("📂 모델링용 데이터 확인"):
    st.dataframe(df_model)

# ---------------------------------------------------------
# 4. 특성(X) / 타깃(y) 설정
# ---------------------------------------------------------
st.subheader("🎯 타깃 변수 및 입력 변수 설정")

target = "damage_ha"

if has_road:
    default_features = ["fire_count", "road_length"]
else:
    default_features = ["fire_count"]

# 사용자가 쓸 특성 선택할 수 있게 (확장 가능)
all_candidate_features = [c for c in df_model.columns if c in ["fire_count", "road_length", "year"]]
features = st.multiselect(
    "입력 변수(피처)를 선택하세요.",
    options=all_candidate_features,
    default=default_features
)

if not features:
    st.error("최소 1개 이상의 입력 변수를 선택해야 합니다.")
    st.stop()

st.write(f"✅ 선택된 입력 변수: `{features}`")
st.write(f"🎯 예측할 타깃 변수: `damage_ha` (산불 피해 면적)")

X = df_model[features]
y = df_model[target]

# ---------------------------------------------------------
# 5. 학습/검증 데이터 분할
# ---------------------------------------------------------
st.subheader("✂ 학습/검증 데이터 분할")

test_size = st.slider("검증 데이터 비율", 0.1, 0.5, 0.2, step=0.05)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

st.write(f"- 학습 데이터: {len(X_train)}개")
st.write(f"- 검증 데이터: {len(X_test)}개")

# ---------------------------------------------------------
# 6. 모델 학습 (선형회귀 + 랜덤포레스트)
# ---------------------------------------------------------
st.subheader("🤖 모델 학습 및 평가")

# 선형회귀
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# 랜덤포레스트
rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=5
)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# 평가 지표
def eval_reg(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

metrics_lin = eval_reg(y_test, y_pred_lin)
metrics_rf = eval_reg(y_test, y_pred_rf)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📐 선형회귀 (Linear Regression)")
    st.write(f"- R²: `{metrics_lin['R²']:.3f}`")
    st.write(f"- MAE: `{metrics_lin['MAE']:.3f}`")

with col2:
    st.markdown("#### 🌲 랜덤포레스트 (Random Forest Regressor)")
    st.write(f"- R²: `{metrics_rf['R²']:.3f}`")
    st.write(f"- MAE: `{metrics_rf['MAE']:.3f}`")

st.caption("※ R²는 1에 가까울수록, MAE는 0에 가까울수록 좋은 성능입니다.")

# ---------------------------------------------------------
# 7. 실제값 vs 예측값 시각화
# ---------------------------------------------------------
st.subheader("📊 실제값 vs 예측값 비교")

result_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred_lin": y_pred_lin,
    "y_pred_rf": y_pred_rf,
})
result_df.reset_index(drop=True, inplace=True)

# 랜덤포레스트 기준으로 산점도
fig_scatter = px.scatter(
    result_df,
    x="y_true",
    y="y_pred_rf",
    labels={
        "y_true": "실제 피해 면적 (ha)",
        "y_pred_rf": "예측 피해 면적 (ha, RF)"
    },
    title="랜덤포레스트 기준: 실제값 vs 예측값"
)
fig_scatter.add_shape(
    type="line",
    x0=result_df["y_true"].min(),
    y0=result_df["y_true"].min(),
    x1=result_df["y_true"].max(),
    y1=result_df["y_true"].max(),
    line=dict(dash="dash")
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# 8. 임도 연장 vs 피해 면적 관계 보기 (있을 때만)
# ---------------------------------------------------------
if has_road:
    st.subheader("🛣 임도 연장과 산불 피해 면적의 관계")

    fig_rl = px.scatter(
        df_model,
        x="road_length",
        y="damage_ha",
        color="fire_count",
        labels={
            "road_length": "임도 연장 길이",
            "damage_ha": "산불 피해 면적 (ha)",
            "fire_count": "산불 발생 건수"
        },
        title="임도 연장 길이 vs 산불 피해 면적 (색: 발생 건수)"
    )
    st.plotly_chart(fig_rl, use_container_width=True)

    st.markdown("""
위 그래프를 통해,
- **임도 연장이 길수록 피해 면적이 줄어드는지**,  
- 혹은 **별 상관이 없는지 / 오히려 피해가 큰 지역이 존재하는지**  
직접 눈으로 확인할 수 있습니다.
""")

# ---------------------------------------------------------
# 9. 피처 중요도 (랜덤포레스트 기준)
# ---------------------------------------------------------
st.subheader("🔎 입력 변수 중요도 (랜덤포레스트 기준)")

importances = rf_reg.feature_importances_
imp_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=True)

fig_imp = px.bar(
    imp_df,
    x="importance",
    y="feature",
    orientation="h",
    labels={
        "importance": "중요도 (feature importance)",
        "feature": "입력 변수"
    },
    title="랜덤포레스트 모델에서 본 변수 중요도"
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("""
- 중요도가 높은 변수일수록,  
  **산불 피해 면적 예측에 더 큰 영향을 미친다**는 뜻입니다.
- 예를 들어, `fire_count`가 가장 크다면  
  “발생 건수가 많은 지역일수록 피해 면적도 큰 경향” 을 의미합니다.
- `road_length`의 중요도가 낮게 나오거나 부정적인 관계를 보인다면,  
  **“임도가 많다고 해서 산불 피해가 줄어들지는 않는다”**라는  
  연구 결과와도 연결시켜 해석할 수 있습니다.
""")
