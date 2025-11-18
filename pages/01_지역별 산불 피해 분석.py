import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

st.set_page_config(page_title="지역별 산불 피해 클러스터링", layout="wide")

st.title("🗺️ 지역별 산불 피해 클러스터링 (표준화 + 유클리드 거리)")

st.markdown("""
전처리 스크립트로 만든 **`region_df.csv`**를 업로드하면,  
다음 변수들을 기반으로 **표준화(Z-score) + 유클리드 거리**로 클러스터링합니다.

- 예: `forest_care_intensity`, `fire_damage_per_ha`,  
      `total_fire_area_ha`, `total_fire_count`,  
      `total_road_km_2011_2020`, `fire_damage_per_road_km` 등
""")

uploaded_region = st.file_uploader("📁 region_df.csv 파일 업로드", type=["csv"])

if uploaded_region is not None:
    region_df = pd.read_csv(uploaded_region)

    st.subheader("원본 region_df 미리보기")
    st.dataframe(region_df.head())

    numeric_cols = region_df.select_dtypes(include=[float, int]).columns.tolist()

    st.subheader("🔧 클러스터링에 사용할 변수 선택")

    default_candidates = [c for c in numeric_cols if c in [
        "forest_care_intensity",
        "fire_damage_per_ha",
        "total_fire_area_ha",
        "total_fire_count",
        "total_road_km_2011_2020",
        "fire_damage_per_road_km",
    ]]
    if len(default_candidates) >= 2:
        default_vars = default_candidates[:3]
    else:
        default_vars = numeric_cols[:3]

    selected_vars = st.multiselect(
        "사용할 변수 선택 (2개 이상 권장)",
        numeric_cols,
        default=default_vars
    )

    if len(selected_vars) < 2:
        st.warning("두 개 이상의 변수를 선택해야 의미 있는 클러스터링이 가능합니다.")
    else:
        X_region = region_df[selected_vars].copy()

        # 1) 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_region)

        st.markdown("**표준화된 데이터 (일부)**")
        st.dataframe(
            pd.DataFrame(X_scaled, columns=selected_vars).head()
        )

        # 2) 클러스터 개수 선택
        k = st.slider("클러스터 개수 선택", 2, 8, 3, 1)

        # 3) 계층적 클러스터링 (유클리드 거리, Ward)
        cluster_model = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward",
        )
        clusters = cluster_model.fit_predict(X_scaled)
        region_df["cluster"] = clusters

        st.subheader("📌 지역별 클러스터 배정 결과")
        show_cols = ["cluster"]
        if "region" in region_df.columns:
            show_cols = ["region", "cluster"]
        show_cols.extend(selected_vars)
        show_cols = list(dict.fromkeys(show_cols))
        st.dataframe(region_df[show_cols].sort_values("cluster"))

        # 4) 2D 시각화
        st.subheader("📉 클러스터링 시각화")

        if "forest_care_intensity" in region_df.columns:
            xcol = "forest_care_intensity"
        else:
            xcol = selected_vars[0]

        if "fire_damage_per_ha" in region_df.columns:
            ycol = "fire_damage_per_ha"
        elif "total_fire_area_ha" in region_df.columns:
            ycol = "total_fire_area_ha"
        else:
            ycol = selected_vars[1] if len(selected_vars) > 1 else selected_vars[0]

        fig_region = px.scatter(
            region_df,
            x=xcol,
            y=ycol,
            color="cluster",
            hover_data=["region"] if "region" in region_df.columns else None,
            title="표준화 + 유클리드 거리 기반 지역 클러스터링 결과",
            labels={xcol: xcol, ycol: ycol},
        )
        st.plotly_chart(fig_region, use_container_width=True)

        st.subheader("📊 클러스터별 평균값 비교 (선택 변수 기준)")
        cluster_stats = region_df.groupby("cluster")[selected_vars].mean()
        st.dataframe(cluster_stats.style.highlight_max(axis=0))

        st.markdown("""
        ### 🧾 클러스터 해석 가이드
        - 같은 클러스터에 속한 지역은 **선택한 변수들의 절대 수준을 고려했을 때** 서로 비슷한 구조를 가집니다.
        - 예를 들어, `forest_care_intensity`와 `fire_damage_per_ha`를 함께 사용했다면,
          - **숲가꾸기 강도가 높고 피해도 큰 고위험군 클러스터**
          - **숲가꾸기는 적지만 피해가 큰 예외적 클러스터**
          - **둘 다 낮은 저위험군 클러스터**
          등을 구분할 수 있습니다.
        - 이는 코사인 유사도와 달리, **실제 규모(절대값)의 차이**를 반영한다는 점이 중요합니다.
        """)
else:
    st.info("지역 클러스터링을 사용하려면 먼저 `region_df.csv`를 업로드해 주세요.")

