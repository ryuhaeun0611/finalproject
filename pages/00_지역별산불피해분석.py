import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import plotly.express as px


# ------------------------------------------------------------
# 1. Load region_df (이미 전처리된 파일)
# ------------------------------------------------------------
st.header("📊 지역별 표준화 + 유클리드 거리 기반 클러스터링")

uploaded = st.file_uploader("region_df.csv 파일 업로드", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # ------------------------------------------------------------
    # 2. 클러스터링에 사용할 변수 선택
    # ------------------------------------------------------------
    st.subheader("🔧 클러스터링 변수 선택")

    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "region"]

    selected_vars = st.multiselect(
        "사용할 변수 선택 (예: 숲가꾸기 강도, 피해면적 등)",
        numeric_cols,
        default=["forest_care_intensity", "fire_damage_per_road_km"]
    )

    if len(selected_vars) < 2:
        st.warning("두 개 이상의 변수를 선택하세요.")
        st.stop()

    X = df[selected_vars].copy()

    # ------------------------------------------------------------
    # 3. 표준화(Z-score)
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.write("표준화된 데이터 (일부):")
    st.dataframe(pd.DataFrame(X_scaled, columns=selected_vars).head())

    # ------------------------------------------------------------
    # 4. 클러스터 개수 선택
    # ------------------------------------------------------------
    k = st.slider("클러스터 개수 선택", 2, 8, 3)

    # ------------------------------------------------------------
    # 5. 계층적 클러스터링 (Euclidean 기반)
    # ------------------------------------------------------------
    cluster_model = AgglomerativeClustering(
        n_clusters=k,
        affinity="euclidean",
        linkage="ward"
    )
    clusters = cluster_model.fit_predict(X_scaled)
    df["cluster"] = clusters

    st.subheader("📌 클러스터링 결과 미리보기")
    st.dataframe(df[["region", "cluster"] + selected_vars])

    # ------------------------------------------------------------
    # 6. 2D 시각화: 숲가꾸기 강도 vs 산불 피해
    # ------------------------------------------------------------
    st.subheader("📉 클러스터링 시각화")

    # 기본 추천 조합
    if "forest_care_intensity" in df.columns and "total_fire_area_ha" in df.columns:
        xcol = "forest_care_intensity"
        ycol = "total_fire_area_ha"
    else:
        xcol = selected_vars[0]
        ycol = selected_vars[1]

    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color="cluster",
        hover_data=["region"],
        title="표준화 + 유클리드 거리 기반 클러스터링 결과",
        labels={xcol: xcol, ycol: ycol}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # 7. 클러스터별 통계 요약
    # ------------------------------------------------------------
    st.subheader("📊 클러스터별 평균값 비교")

    cluster_stats = df.groupby("cluster")[selected_vars].mean()
    st.dataframe(cluster_stats.style.highlight_max(axis=0))

    # ------------------------------------------------------------
    # 8. 해석 안내
    # ------------------------------------------------------------
    st.markdown("""
    ### 🧾 해석 포인트
    - 클러스터는 **표준화된 변수들의 유클리드 거리**를 기반으로 형성됩니다.
    - 같은 클러스터 안에 있다는 것은  
      → *숲가꾸기 강도, 산불 피해 규모, 임도 연장 등에서 전체적인 절대값 구조가 유사하다*는 의미입니다.
    - 이는 “코사인 유사도”처럼 단순 패턴이 아니라  
      **실제 정책적으로 중요한 절대적 차이**를 반영합니다.
    - x-y 산점도에서 오른쪽 위(숲가꾸기 강도↑ 피해규모↑)에 있는 클러스터는  
      정책 타깃이 필요한 ‘고위험군 지역’으로 볼 수 있습니다.
    """)
