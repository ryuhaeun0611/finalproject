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
- **숲가꾸기/조림 밀도**: 0.10~0.70
- **수관 밀도**: 0.50~0.95
- **침엽수 비율**: 0.30~0.78
- **평균 경사도**: 5~45도
- **임분 밀도**: 0.40~0.82
- **상대습도**: 25~70%

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

# [이후 코드는 동일하게 유지]
