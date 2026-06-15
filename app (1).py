"""
AIDEOM-VN Web App – Streamlit
Chạy: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import io
import subprocess
import sys
from pathlib import Path

# ── CẤU HÌNH TRANG ───────────────────────────────────────────────
st.set_page_config(
    page_title="VN AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS tùy chỉnh ────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #fafafa !important; }
.metric-card {
    background: #1a1d27; border-radius: 12px; padding: 20px;
    border-left: 4px solid #ff4b4b; margin-bottom: 10px;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #ff4b4b; }
.metric-label { font-size: 0.85rem; color: #888; margin-bottom: 4px; }
.metric-delta { font-size: 0.85rem; color: #00c853; }
h1 { font-size: 2.8rem !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# ── DỮ LIỆU ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    macro_csv = """year,GDP_trillion_VND,GDP_billion_USD,GDP_growth_pct,digital_economy_share_GDP_pct,FDI_disbursed_billion_USD,GDP_per_capita_USD
2020,8044.4,346.6,2.91,12.0,19.98,3521
2021,8487.5,366.1,2.58,12.7,19.74,3717
2022,9513.3,408.8,8.02,14.3,22.40,4163
2023,10221.8,430.0,5.05,16.5,23.18,4347
2024,11511.9,476.3,7.09,18.3,25.35,4700
2025,12847.6,514.0,8.02,19.5,27.60,5026"""

    sectors_csv = """sector_id,sector_name_vi,growth_rate_2024_pct,labor_million,export_billion_USD,ai_readiness_0_100,automation_risk_pct,spillover_coef_0_1
1,Nông-Lâm-Thủy sản,3.27,13.2,40.5,15,18,0.35
2,CN Chế biến,9.64,11.5,290.9,55,42,0.78
3,Xây dựng,7.45,4.8,2.5,20,25,0.42
4,Khai khoáng,-1.2,0.3,8.2,30,55,0.30
5,Bán buôn-bán lẻ,7.10,7.8,5.5,48,38,0.55
6,Tài chính-NH,7.36,0.55,1.2,72,52,0.85
7,Logistics,9.93,1.95,3.1,42,35,0.72
8,CNTT-Truyền thông,7.85,0.62,178.0,88,28,0.92
9,Giáo dục,6.42,2.15,0.0,38,22,0.65
10,Y tế,6.85,0.75,0.0,45,18,0.60"""

    regions_csv = """region_id,region_name_vi,grdp_per_capita_million_VND,fdi_registered_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,gini_coef,rd_intensity_pct,internet_penetration_pct
1,Trung du MN phía Bắc,57.0,3.5,38,22,21.5,0.405,0.18,72
2,Đồng bằng sông Hồng,152.3,20.0,78,68,36.8,0.358,0.85,92
3,Bắc Trung Bộ + DH,87.5,8.2,55,40,27.5,0.372,0.32,84
4,Tây Nguyên,68.9,0.8,32,18,18.2,0.412,0.15,68
5,Đông Nam Bộ,158.9,18.5,82,75,42.5,0.385,0.78,94
6,Đồng bằng sông CL,80.5,2.1,48,30,16.8,0.392,0.22,78"""

    df_macro   = pd.read_csv(io.StringIO(macro_csv))
    df_sectors = pd.read_csv(io.StringIO(sectors_csv))
    df_regions = pd.read_csv(io.StringIO(regions_csv))
    return df_macro, df_sectors, df_regions

df_macro, df_sectors, df_regions = load_data()

# ── HỖ TRỢ CHẠY SCRIPT MÔ HÌNH ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

SCRIPT_PAGES = {
    "🗺️ Bài 4 — LP ngành-vùng": {
        "title": "Bài 4 — LP phân bổ ngân sách số theo vùng",
        "desc": "LP 24 biến phân bổ 50,000 tỷ cho 6 vùng × 4 hạng mục",
        "script": "bai04_lp_region.py",
        "image": "bai04_ket_qua.png",
    },
    "📋 Bài 5 — MIP 15 dự án": {
        "title": "Bài 5 — MIP lựa chọn tối ưu 15 dự án",
        "desc": "MIP lựa chọn tối ưu trong 15 dự án chuyển đổi số quốc gia",
        "script": "bai05_mip_projects.py",
        "image": "bai05_ket_qua.png",
    },
    "🎯 Bài 7 — Pareto đa mục tiêu": {
        "title": "Bài 7 — Pareto front 4 mục tiêu",
        "desc": "Pareto front 4 mục tiêu: GDP, bình đẳng, môi trường, an ninh",
        "script": "bai07_pareto.py",
        "image": "bai07_ket_qua.png",
    },
    "⏳ Bài 8 — Động 2026-2035": {
        "title": "Bài 8 — Tối ưu động 2026-2035",
        "desc": "Tối ưu động phân bổ vốn 10 năm, U=ρ^t·ln(C_t)",
        "script": "bai08_dynamic.py",
        "image": "bai08_ket_qua.png",
    },
    "👷 Bài 9 — Lao động & AI": {
        "title": "Bài 9 — Lao động & AI",
        "desc": "LP tối đa NetJob: đầu tư AI vs đào tạo lại 8 ngành",
        "script": "bai09_labor.py",
        "image": "bai09_ket_qua.png",
    },
    "🎲 Bài 10 — Stochastic SP": {
        "title": "Bài 10 — Stochastic LP",
        "desc": "Stochastic LP 2 giai đoạn, 4 kịch bản, VSS & EVPI",
        "script": "bai10_stochastic.py",
        "image": "bai10_ket_qua.png",
    },
    "🤖 Bài 11 — Q-learning RL": {
        "title": "Bài 11 — Q-Learning",
        "desc": "Q-Learning 81 trạng thái, 5 hành động, 20,000 episodes",
        "script": "bai11_qlearning.py",
        "image": "bai11_ket_qua.png",
    },
}

@st.cache_data(show_spinner=False)
def _run_script(script_name: str, mtime: float):
    script_path = BASE_DIR / script_name
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.stdout, proc.stderr, proc.returncode

def render_script_page(page_name: str):
    cfg = SCRIPT_PAGES[page_name]
    st.title(cfg["title"])
    st.info(f"📌 {cfg['desc']}")

    script_path = BASE_DIR / cfg["script"]
    if not script_path.exists():
        st.error(f"Không tìm thấy file mô hình: {script_path.name}")
        return

    with st.spinner(f"Đang chạy {script_path.name}..."):
        stdout, stderr, code = _run_script(cfg["script"], script_path.stat().st_mtime)

    img_path = BASE_DIR / cfg["image"]
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    if code != 0:
        st.error(f"Script trả về mã lỗi {code}")
    if stderr.strip():
        with st.expander("Cảnh báo / lỗi từ console"):
            st.code(stderr)
    if stdout.strip():
        with st.expander("Xem log giải mô hình"):
            st.code(stdout, language="text")

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇻🇳 AIDEOM-VN")
    st.markdown("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    st.divider()
    page = st.radio("", [
        "🏠 Trang chủ",
        "📈 Bài 1 — Cobb-Douglas + AI",
        "💰 Bài 2 — LP ngân sách số",
        "🏆 Bài 3 — Priority 10 ngành",
        "🗺️ Bài 4 — LP ngành-vùng",
        "📋 Bài 5 — MIP 15 dự án",
        "🌐 Bài 6 — TOPSIS 6 vùng",
        "🎯 Bài 7 — Pareto đa mục tiêu",
        "⏳ Bài 8 — Động 2026-2035",
        "👷 Bài 9 — Lao động & AI",
        "🎲 Bài 10 — Stochastic SP",
        "🤖 Bài 11 — Q-learning RL",
        "🔗 Bài 12 — AIDEOM tích hợp",
    ], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════
# TRANG CHỦ
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Trang chủ":
    st.markdown("# VN AIDEOM-VN")
    st.markdown("### *AI-Driven Decision Optimization Model for Vietnam*")
    st.markdown("Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020-2025.")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "+8,02%")
    col2.metric("Kinh tế số / GDP", "≈19,5%", "+1,2 dpt")
    col3.metric("FDI giải ngân 2025", "27,6 tỷ USD", "+8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "+6,9%")

    st.divider()
    st.markdown("### 📚 12 bài toán theo 4 cấp độ")

    with st.expander("🟢 Cấp độ DỄ — Làm quen mô hình", expanded=True):
        st.markdown("""
| Bài | Nội dung | Kỹ thuật |
|-----|----------|----------|
| **Bài 1** | Hàm sản xuất Cobb-Douglas mở rộng + AI | Growth accounting, dự báo GDP 2030 |
| **Bài 2** | LP phân bổ ngân sách 4 hạng mục | scipy.optimize, shadow price |
| **Bài 3** | Chỉ số ưu tiên 10 ngành | Min-max norm, weighted scoring |
""")

    with st.expander("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển"):
        st.markdown("""
| Bài | Nội dung | Kỹ thuật |
|-----|----------|----------|
| **Bài 4** | LP phân bổ 50,000 tỷ theo 6 vùng × 4 hạng mục | LP 24 biến, equity constraint |
| **Bài 5** | MIP lựa chọn 15 dự án chuyển đổi số | Binary variables, scipy.milp |
| **Bài 6** | TOPSIS xếp hạng 6 vùng đầu tư AI | MCDM, Entropy weights |
""")

    with st.expander("🟠 Cấp độ KHÁ KHÓ — Tối ưu nâng cao"):
        st.markdown("""
| Bài | Nội dung | Kỹ thuật |
|-----|----------|----------|
| **Bài 7** | Pareto 4 mục tiêu xung đột | Weighted-sum scalarization, SLSQP |
| **Bài 8** | Tối ưu động vốn 2026-2035 | Dynamic optimization, ρ·ln(C) |
| **Bài 9** | Tác động AI tới lao động | LP 16 biến, NetJob constraint |
""")

    with st.expander("🔴 Cấp độ KHÓ — Bất định & học máy"):
        st.markdown("""
| Bài | Nội dung | Kỹ thuật |
|-----|----------|----------|
| **Bài 10** | Stochastic LP 2 giai đoạn, 4 kịch bản | VSS, EVPI, minimax regret |
| **Bài 11** | Q-Learning chính sách kinh tế | MDP 81 states, tabular RL |
| **Bài 12** | AIDEOM-VN tích hợp 6 module | Dashboard 5 kịch bản chính sách |
""")

    st.info("👈 Chọn bài từ menu bên trái để bắt đầu")

# ══════════════════════════════════════════════════════════════════
# BÀI 1
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Bài 1 — Cobb-Douglas + AI":
    st.title("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng")
    st.markdown("**Mô hình:** `Y = A · K^α · L^β · D^γ · AI^δ · H^θ`")

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("#### Tham số hệ số co giãn")
        alpha = st.slider("α (Vốn K)", 0.20, 0.45, 0.33, 0.01)
        beta  = st.slider("β (Lao động L)", 0.30, 0.55, 0.42, 0.01)
        gamma = st.slider("γ (Số hóa D)", 0.05, 0.20, 0.10, 0.01)
        delta = st.slider("δ (AI)", 0.03, 0.15, 0.08, 0.01)
        theta = st.slider("θ (Nhân lực H)", 0.03, 0.15, 0.07, 0.01)
        total = alpha+beta+gamma+delta+theta
        if abs(total-1.0)>0.02:
            st.warning(f"⚠️ Tổng = {total:.2f} ≠ 1.0 (lợi suất không đổi)")
        else:
            st.success(f"✓ Tổng = {total:.2f} ≈ 1.0")

    Y  = df_macro['GDP_trillion_VND'].values
    K  = np.array([16500,17800,19600,21300,23500,25900])
    L  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
    D  = df_macro['digital_economy_share_GDP_pct'].values
    AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
    H  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
    years = df_macro['year'].values

    A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    A_mean = A.mean()
    Y_hat  = A_mean * K**alpha * L**beta * D**gamma * AI**delta * H**theta
    mape   = np.mean(np.abs((Y-Y_hat)/Y))*100

    with col2:
        fig, axes = plt.subplots(1,2,figsize=(12,4),facecolor='#0e1117')
        for ax in axes:
            ax.set_facecolor('#0e1117'); ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#333')

        axes[0].plot(years, A, 'o-', color='#ff4b4b', lw=2.5, ms=9)
        axes[0].fill_between(years, A, alpha=0.2, color='#ff4b4b')
        axes[0].set_title('TFP A_t theo năm', color='white', fontsize=12)
        axes[0].set_xlabel('Năm', color='white'); axes[0].set_ylabel('TFP', color='white')

        axes[1].plot(years, Y,     'o-', color='#00c853', lw=2.5, ms=8, label='Thực tế')
        axes[1].plot(years, Y_hat, 's--', color='#ff9800', lw=2, ms=7, label=f'Dự báo (MAPE={mape:.2f}%)')
        axes[1].set_title('GDP: Thực tế vs Dự báo', color='white', fontsize=12)
        axes[1].set_xlabel('Năm', color='white'); axes[1].set_ylabel('nghìn tỷ VND', color='white')
        axes[1].legend(facecolor='#1a1d27', labelcolor='white')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.markdown("#### Dự báo GDP 2030")
    c1,c2,c3 = st.columns(3)
    D_2030  = c1.slider("Kinh tế số 2030 (%)", 20, 40, 30)
    AI_2030 = c2.slider("DN số 2030 (nghìn)", 80, 150, 100)
    H_2030  = c3.slider("Nhân lực đào tạo 2030 (%)", 30, 45, 35)

    K_f=K[-1]; L_f=L[-1]; A_f=A[-1]
    for yr in range(2026,2031):
        K_f*=1.06; L_f*=1.01; A_f*=1.012
        t=(yr-2025)/5
        D_f=D[-1]+t*(D_2030-D[-1]); AI_f=AI[-1]+t*(AI_2030-AI[-1]); H_f=H[-1]+t*(H_2030-H[-1])
    GDP_2030=A_f*K_f**alpha*L_f**beta*D_f**gamma*AI_f**delta*H_f**theta
    cagr=(GDP_2030/Y[-1])**(1/5)-1

    co1,co2,co3 = st.columns(3)
    co1.metric("GDP 2030 (nghìn tỷ VND)", f"{GDP_2030:,.0f}", f"+{cagr*100:.2f}%/năm")
    co2.metric("GDP 2030 (tỷ USD)", f"{GDP_2030/26:,.0f}", "tỷ USD")
    co3.metric("MAPE dự báo", f"{mape:.2f}%", "Độ chính xác mô hình")

    with st.expander("📊 Phân rã tăng trưởng (Growth Accounting)"):
        dY=np.diff(np.log(Y)); cK=alpha*np.diff(np.log(K)); cL=beta*np.diff(np.log(L))
        cD=gamma*np.diff(np.log(D)); cAI=delta*np.diff(np.log(AI)); cH=theta*np.diff(np.log(H))
        cTFP=dY-cK-cL-cD-cAI-cH
        periods=[f"{years[i]}-{years[i+1]}" for i in range(5)]
        df_ga=pd.DataFrame({'Giai đoạn':periods,'GDP%':(dY*100).round(2),
            'K':(cK*100).round(2),'L':(cL*100).round(2),'D':(cD*100).round(2),
            'AI':(cAI*100).round(2),'H':(cH*100).round(2),'TFP':(cTFP*100).round(2)})
        st.dataframe(df_ga, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# BÀI 2
# ══════════════════════════════════════════════════════════════════
elif page == "💰 Bài 2 — LP ngân sách số":
    st.title("Bài 2 — LP Phân bổ ngân sách số")
    st.markdown("**max Z = 0.85·x₁ + 1.20·x₂ + 0.95·x₃ + 1.35·x₄**")

    budget = st.slider("Tổng ngân sách (nghìn tỷ VND)", 80, 200, 100, 5)

    c_obj=np.array([-0.85,-1.20,-0.95,-1.35])
    A_ub=np.array([[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],
                   [0.35,-0.65,0.35,-0.65]])
    b_ub=np.array([budget,-25,-15,-20,-10,0])
    res=linprog(c_obj,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method='highs')

    if res.success:
        x=res.x; Z=-res.fun
        labels=['Hạ tầng số','AI & Dữ liệu','Nhân lực số','R&D']
        c1,c2,c3,c4=st.columns(4)
        cols=[c1,c2,c3,c4]; colors_m=['#1976D2','#FF9800','#4CAF50','#9C27B0']
        for i,(col,lbl,xi) in enumerate(zip(cols,labels,x)):
            col.metric(lbl, f"{xi:.1f} nghìn tỷ", f"{xi/budget*100:.1f}%")

        st.success(f"**Z* = {Z:.2f} nghìn tỷ VND GDP tăng thêm** | Ngân sách dùng: {x.sum():.0f}/{budget}")

        fig,axes=plt.subplots(1,2,figsize=(12,4),facecolor='#0e1117')
        for ax in axes:
            ax.set_facecolor('#0e1117'); ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#333')
        axes[0].bar(labels,x,color=colors_m,alpha=0.9,edgecolor='none')
        for i,v in enumerate(x): axes[0].text(i,v+0.3,f'{v:.1f}',ha='center',color='white',fontweight='bold')
        axes[0].set_title('Phân bổ tối ưu',color='white'); axes[0].set_ylabel('nghìn tỷ VND',color='white')

        budgets=np.arange(70,210,10); Zs=[]
        for B in budgets:
            b2=b_ub.copy(); b2[0]=B
            r2=linprog(c_obj,A_ub=A_ub,b_ub=b2,bounds=[(0,None)]*4,method='highs')
            Zs.append(-r2.fun if r2.success else np.nan)
        axes[1].plot(budgets,Zs,'o-',color='#ff4b4b',lw=2.5,ms=6)
        axes[1].axvline(budget,color='#ff9800',ls='--',alpha=0.8,label=f'B={budget}')
        axes[1].set_title('Đường cong Z*(B)',color='white'); axes[1].set_xlabel('Ngân sách',color='white')
        axes[1].set_ylabel('Z*',color='white'); axes[1].legend(facecolor='#1a1d27',labelcolor='white')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# BÀI 3
# ══════════════════════════════════════════════════════════════════
elif page == "🏆 Bài 3 — Priority 10 ngành":
    st.title("Bài 3 — Chỉ số ưu tiên ngành")

    st.markdown("#### Điều chỉnh trọng số")
    c1,c2,c3,c4 = st.columns(4)
    w1=c1.slider("a₁ Tăng trưởng",0.05,0.40,0.15,0.05)
    w2=c1.slider("a₂ Năng suất",0.05,0.40,0.15,0.05)
    w3=c2.slider("a₃ Lan tỏa",0.05,0.40,0.20,0.05)
    w4=c2.slider("a₄ Xuất khẩu",0.05,0.40,0.15,0.05)
    w5=c3.slider("a₅ Việc làm",0.05,0.30,0.10,0.05)
    w6=c3.slider("a₆ AI Readiness",0.05,0.40,0.20,0.05)
    w7=c4.slider("a₇ Rủi ro (trừ)",0.05,0.30,0.15,0.05)
    total_w=w1+w2+w3+w4+w5+w6+w7
    if abs(total_w-1.0)>0.05: c4.warning(f"Tổng={total_w:.2f}")
    else: c4.success(f"Tổng={total_w:.2f} ✓")

    prod=np.array([103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1])
    df_s=df_sectors.copy(); df_s['productivity']=prod
    def nm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    def nb(x): return (x.max()-x)/(x.max()-x.min()+1e-10)
    X=np.column_stack([nm(df_s['growth_rate_2024_pct']),nm(df_s['productivity']),
                       nm(df_s['spillover_coef_0_1']),nm(df_s['export_billion_USD']),
                       nm(df_s['labor_million']),nm(df_s['ai_readiness_0_100'])])
    Xb=nb(df_s['automation_risk_pct'])
    w=np.array([w1,w2,w3,w4,w5,w6])
    w_norm=w/w.sum()*(1-w7)
    priority=X@w_norm + w7*Xb
    df_s['Priority']=priority
    df_sorted=df_s.sort_values('Priority',ascending=False).reset_index(drop=True)
    df_sorted['Hạng']=range(1,11)

    fig,ax=plt.subplots(figsize=(10,5),facecolor='#0e1117')
    ax.set_facecolor('#0e1117'); ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')
    colors_p=plt.cm.RdYlGn(np.linspace(0.3,0.9,10))
    bars=ax.barh(df_sorted['sector_name_vi'][::-1],df_sorted['Priority'][::-1],
                 color=colors_p,alpha=0.9)
    ax.set_title('Xếp hạng Priority ngành',color='white',fontsize=13)
    ax.set_xlabel('Điểm Priority',color='white')
    for bar,v in zip(bars,df_sorted['Priority'][::-1]):
        ax.text(v+0.002,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',color='white',fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Bảng xếp hạng")
    st.dataframe(df_sorted[['Hạng','sector_name_vi','Priority','growth_rate_2024_pct',
                             'ai_readiness_0_100','automation_risk_pct']].rename(columns={
        'sector_name_vi':'Ngành','Priority':'Điểm','growth_rate_2024_pct':'Tăng trưởng %',
        'ai_readiness_0_100':'AI Ready','automation_risk_pct':'Rủi ro TĐH %'}),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# BÀI 6 – TOPSIS
# ══════════════════════════════════════════════════════════════════
elif page == "🌐 Bài 6 — TOPSIS 6 vùng":
    st.title("Bài 6 — TOPSIS Xếp hạng 6 vùng đầu tư AI")
    regs=['TDMN-PB','ĐBSH','BTB+DHMT','Tây Nguyên','Đông Nam Bộ','ĐBSCL']
    crit=['grdp_per_capita_million_VND','fdi_registered_billion_USD','digital_index_0_100',
          'ai_readiness_0_100','trained_labor_pct','rd_intensity_pct','internet_penetration_pct','gini_coef']
    is_b=[True]*7+[False]

    st.markdown("#### Trọng số TOPSIS")
    cols=st.columns(8)
    ws=[cols[i].number_input(c[:6],0.0,1.0,v,0.05,key=f"w{i}")
        for i,c,v in zip(range(8),['GRDP','FDI','Digital','AI','LĐ ĐT','R&D','Internet','Gini'],
                          [0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])]
    w=np.array(ws); w=w/w.sum()

    X=df_regions[crit].values.astype(float)
    R=X/np.sqrt((X**2).sum(axis=0)); V=R*w
    Ap=np.where(is_b,V.max(axis=0),V.min(axis=0))
    An=np.where(is_b,V.min(axis=0),V.max(axis=0))
    Sp=np.sqrt(((V-Ap)**2).sum(axis=1)); Sn=np.sqrt(((V-An)**2).sum(axis=1))
    C=Sn/(Sp+Sn)

    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor='#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117'); ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#333')
    idx=np.argsort(C)[::-1]
    colors_t=['#ff4b4b' if i==0 else '#ff9800' if i<=2 else '#555' for i in range(6)]
    axes[0].barh([regs[i] for i in idx[::-1]],[C[i] for i in idx[::-1]],
                 color=[colors_t[k] for k in range(5,-1,-1)],alpha=0.9)
    axes[0].set_title('Xếp hạng TOPSIS',color='white',fontsize=12)
    axes[0].set_xlabel('C* (gần 1 = tốt hơn)',color='white')

    # Radar
    cats=['GRDP/người','FDI','Digital','AI Ready','Lao động ĐT','R&D','Internet']
    X_norm=(X[:,:-1]-X[:,:-1].min(0))/(X[:,:-1].max(0)-X[:,:-1].min(0)+1e-10)
    angles=np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()
    angles+=[angles[0]]
    ax2=plt.subplot(122,projection='polar',facecolor='#0e1117')
    ax2.set_facecolor('#0e1117'); ax2.tick_params(colors='white')
    colors_radar=['#ff4b4b','#4CAF50','#2196F3','#FF9800','#9C27B0','#00BCD4']
    for i in [idx[0],idx[1],idx[2]]:
        vals=X_norm[i].tolist()+[X_norm[i][0]]
        ax2.plot(angles,vals,color=colors_radar[i],lw=2,label=regs[i])
        ax2.fill(angles,vals,alpha=0.1,color=colors_radar[i])
    ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(cats,color='white',size=8)
    ax2.set_title('Top 3 vùng (radar)',color='white',pad=15)
    ax2.legend(loc='lower right',facecolor='#1a1d27',labelcolor='white',fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    df_topsis=pd.DataFrame({'Vùng':regs,'C*':C.round(4),'Hạng':pd.Series(C).rank(ascending=False).astype(int)})
    st.dataframe(df_topsis.sort_values('C*',ascending=False),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════
# BÀI 12 – AIDEOM tích hợp
# ══════════════════════════════════════════════════════════════════
elif page == "🔗 Bài 12 — AIDEOM tích hợp":
    st.title("Bài 12 — AIDEOM-VN: Hệ thống tích hợp 5 kịch bản")

    scenarios={'S1 Truyền thống':[0.70,0.10,0.10,0.10],
               'S2 Số hóa nhanh':[0.25,0.45,0.15,0.15],
               'S3 AI dẫn dắt':[0.20,0.20,0.45,0.15],
               'S4 Bao trùm số':[0.30,0.20,0.10,0.40],
               'S5 Tối ưu LP':[None,None,None,None]}

    Y=df_macro['GDP_trillion_VND'].values
    D=df_macro['digital_economy_share_GDP_pct'].values
    K=np.array([16500,17800,19600,21300,23500,25900])
    L=np.array([53.6,50.5,51.7,52.4,52.9,53.4])
    AI=np.array([55.6,60.2,65.4,67.0,73.8,80.1])
    H=np.array([24.1,26.1,26.2,27.0,28.4,29.2])
    alpha_,beta_,gamma_,delta_,theta_=0.33,0.42,0.10,0.08,0.07
    A_hist=Y/(K**alpha_*L**beta_*D**gamma_*AI**delta_*H**theta_)

    res_s={}
    for sname,alloc in scenarios.items():
        D_t=30 if 'Số hóa' in sname else 26
        AI_t=120 if 'AI' in sname else 100
        H_t =35  if 'Bao trùm' in sname else 33
        K_s,L_s,A_s=K[-1],L[-1],A_hist[-1]
        for _ in range(5): K_s*=1.06; L_s*=1.01; A_s*=1.012
        D_5=D[-1]+D_t-D[-1]; AI_5=AI[-1]+AI_t-AI[-1]; H_5=H[-1]+H_t-H[-1]
        gdp=A_s*K_s**alpha_*L_s**beta_*D_5**gamma_*AI_5**delta_*H_5**theta_
        cagr=(gdp/Y[-1])**(1/5)-1
        if alloc[0] is None:
            c_obj=np.array([-0.85,-1.20,-0.95,-1.35])
            A_ub=np.array([[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]])
            b_ub=np.array([100,-25,-15,-20,-10,0])
            r=linprog(c_obj,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method='highs')
            gain=-r.fun if r.success else 0
        else:
            beta_lp=np.array([0.85,1.20,0.95,1.35])
            gain=float(np.dot(beta_lp, np.array(alloc)*100))
        res_s[sname]={'GDP_2030':gdp,'CAGR':cagr,'gain':gain}

    # Bảng KPI
    st.markdown("### 📊 Bảng tổng hợp KPI 2030")
    rows=[]
    for s,v in res_s.items():
        rows.append({'Kịch bản':s,'GDP 2030 (nghìn tỷ)':f"{v['GDP_2030']:,.0f}",
                     'CAGR (%/năm)':f"{v['CAGR']*100:.2f}%",'GDP gain (tỷ)':f"{v['gain']:,.0f}"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # Biểu đồ so sánh
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor='#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117'); ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#333')
    names=list(res_s.keys()); short=['S1','S2','S3','S4','S5*']
    gdps=[res_s[n]['GDP_2030'] for n in names]
    gains=[res_s[n]['gain'] for n in names]
    colors_s=['#795548','#2196F3','#9C27B0','#4CAF50','#ff4b4b']
    bars=axes[0].bar(short,gdps,color=colors_s,alpha=0.9,edgecolor='none')
    axes[0].axhline(Y[-1],color='#ff9800',ls='--',alpha=0.7,label=f'GDP 2025={Y[-1]:,.0f}')
    axes[0].set_title('GDP 2030 theo kịch bản',color='white',fontsize=12)
    axes[0].set_ylabel('nghìn tỷ VND',color='white'); axes[0].legend(facecolor='#1a1d27',labelcolor='white')
    for bar,v in zip(bars,gdps): axes[0].text(bar.get_x()+bar.get_width()/2,v+50,f'{v:,.0f}',
                                               ha='center',color='white',fontsize=8,fontweight='bold')
    axes[1].bar(short,gains,color=colors_s,alpha=0.9,edgecolor='none')
    axes[1].set_title('GDP gain từ đầu tư số',color='white',fontsize=12)
    axes[1].set_ylabel('Tỷ VND',color='white')
    plt.tight_layout(); st.pyplot(fig); plt.close()


elif page in SCRIPT_PAGES:
    render_script_page(page)

else:
    st.error("Trang không tồn tại hoặc chưa được cấu hình.")
