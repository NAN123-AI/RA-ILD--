# app.py
import streamlit as st
import numpy as np

st.set_page_config(page_title="RA-ILD风险预测模型", layout="wide")

# ===== 背景肺图 (全屏 + 内容重叠) =====
st.markdown("""
<style>
/* 背景图片 */
.bg-img {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: -1;
    opacity: 0.3;  /* 可调透明度 */
}

/* 页面内容浮在图片上 */
.content {
    position: relative;
    z-index: 1;
    max-width: 900px;
    margin: 50px auto;
    padding: 30px;
    background: rgba(255,255,255,0.8); /* 半透明背景 */
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

/* 标题文字 */
h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: #1a1a1a;
    font-weight: bold;
}
</style>

<img class="bg-img" src="lung.png">
""", unsafe_allow_html=True)

# ===== 页面内容 =====
st.markdown('<div class="content">', unsafe_allow_html=True)

# 页面标题
st.title("RA-ILD风险预测模型（论文一致版）")
st.markdown("基于多因素Logistic回归模型（AUC=0.959）")

# 输入区
col1, col2 = st.columns(2)
with col1:
    age = st.slider("年龄", 30, 90, 60)
    smoke = st.radio("吸烟史", ["否", "是"])
    smoke_val = 1 if smoke == "是" else 0
    il22 = st.slider("IL-22 (pg/ml)", 100, 350, 220)

with col2:
    mcvab = st.slider("MCV-Ab", 0, 1000, 500)
    mchc = st.slider("MCHC", 260, 350, 320)

# 中心化
age_c = (age - 60)/10
il22_c = (il22 - 220)/50
mcvab_c = (mcvab - 500)/100
mchc_c = (mchc - 320)/10

# Logistic回归计算
z = (
    -0.032
    -0.059 * il22_c
    +0.110 * age_c
    +4.288 * smoke_val
    +0.006 * mcvab_c
    -0.124 * mchc_c
)
risk = 1 / (1 + np.exp(-z))

# 输出预测结果
st.subheader("📊 预测结果")
st.metric("RA-ILD风险概率", f"{risk:.2%}")
if risk < 0.2:
    st.success("低风险")
elif risk < 0.5:
    st.warning("中等风险")
else:
    st.error("高风险")
st.progress(float(risk))

# IL-22提示
st.subheader("🧬 IL-22临床提示")
if il22 < 243.06:
    st.error("IL-22 < 243 → 高风险提示")
else:
    st.success("IL-22 ≥ 243 → 相对低风险")

# 模型说明
st.markdown("""
---
### 📚 模型说明
- 多因素Logistic回归模型
- AUC = 0.959
- 敏感度 = 97.0%
- 特异度 = 87.9%
""")

st.markdown('</div>', unsafe_allow_html=True)
