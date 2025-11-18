import streamlit as st
import base64

st.set_page_config(
    page_title="숲가꾸기와 산불 피해",
    layout="wide"
)

st.title("🌲 숲가꾸기 활동이 산불 피해에 미치는 영향")

# ---------------------------------------------------------
# 👥 앱 제작자
# ---------------------------------------------------------
st.markdown("""
### 👥 앱 제작자  
**20908 류지민 · 20909 류하은 · 20923 최보경**
""")

st.markdown("----")

# ---------------------------------------------------------
# 💡 PDF embed 함수
# ---------------------------------------------------------
def display_pdf(file_path):
    """PDF를 base64로 인코딩하여 브라우저에 표시."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></ifr
