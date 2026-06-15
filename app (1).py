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
# ══════════════════════════════════════════════════════════════════
# BÀI 12 – AIDEOM tích hợp (4 tabs giống của thầy)
# ══════════════════════════════════════════════════════════════════
elif page == "🔗 Bài 12 — AIDEOM tích hợp":
    st.title("Bài 12 — AIDEOM-VN: Hệ thống tích hợp 5 kịch bản")

    Y  = df_macro["GDP_trillion_VND"].values
    D  = df_macro["digital_economy_share_GDP_pct"].values
    K  = np.array([16500,17800,19600,21300,23500,25900])
    L  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
    AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
    H  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
    a_,b_,g_,d_,t_ = 0.33,0.42,0.10,0.08,0.07
    A_hist = Y/(K**a_*L**b_*D**g_*AI**d_*H**t_)

    tab1,tab2,tab3,tab4 = st.tabs([
        "📊 Tổng quan (M1-M2)",
        "💰 Phân bổ (M3)",
        "🎯 5 Kịch bản (M6)",
        "⚠️ Cảnh báo rủi ro (M4-M5)"
    ])

    with tab1:
        st.markdown("## M1 — Dự báo kinh tế (Cobb-Douglas)")
        if st.button("▶ Chạy M1", type="primary"):
            with st.spinner("Đang tính toán..."):
                A_mean=A_hist.mean()
                Y_hat=A_mean*K**a_*L**b_*D**g_*AI**d_*H**t_
                mape=float(np.mean(np.abs((Y-Y_hat)/Y))*100)
                Ks,Ls,As=K[-1],L[-1],A_hist[-1]
                for _ in range(5): Ks*=1.06; Ls*=1.01; As*=1.012
                GDP2030=As*Ks**a_*Ls**b_*(D[-1]+10)**g_*(AI[-1]+14)**d_*(H[-1]+5)**t_
                c1,c2,c3=st.columns(3)
                c1.metric("MAPE (Cobb-Douglas)",f"{mape:.2f}%")
                c2.metric("Ā (TFP trung bình)",f"{A_hist.mean():.4f}")
                c3.metric("Y 2030 dự báo",f"{GDP2030:,.0f} ng.tỷ")
                dY=np.diff(np.log(Y)); cK=a_*np.diff(np.log(K)); cL=b_*np.diff(np.log(L))
                cD=g_*np.diff(np.log(D)); cAI=d_*np.diff(np.log(AI)); cH=t_*np.diff(np.log(H))
                cTFP=dY-cK-cL-cD-cAI-cH
                means={"TFP (A)":cTFP.mean()*100,"Vốn (K)":cK.mean()*100,
                       "Lao động (L)":cL.mean()*100,"Số hóa (D)":cD.mean()*100,
                       "AI":cAI.mean()*100,"Nhân lực số (H)":cH.mean()*100}
                fig,ax=plt.subplots(figsize=(10,4),facecolor="#0e1117")
                ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
                for sp in ax.spines.values(): sp.set_color("#333")
                cols_bar=["#26C6DA","#FFA726","#EF5350","#AB47BC","#42A5F5","#66BB6A"]
                bars=ax.bar(list(means.keys()),list(means.values()),color=cols_bar,alpha=0.9,edgecolor="none")
                ax.set_title("Phân rã đóng góp tăng trưởng 2020-2025",color="white",fontsize=12)
                ax.set_xlabel("Yếu tố",color="white"); ax.set_ylabel("Đóng_góp_pct",color="white")
                ax.axhline(0,color="#555",lw=0.8)
                for bar,v in zip(bars,means.values()):
                    ax.text(bar.get_x()+bar.get_width()/2,v+0.05 if v>=0 else v-0.15,
                            f"{v:.2f}%",ha="center",color="white",fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("👆 Click **Chạy M1** để tính toán và hiển thị kết quả")

        st.divider()
        st.markdown("## M2 — Đánh giá sẵn sàng số (TOPSIS)")
        if st.button("▶ Chạy M2", type="primary"):
            with st.spinner("Đang tính TOPSIS..."):
                regs=["TDMN-PB","ĐBSH","BTB+DHMT","Tây Nguyên","Đông Nam Bộ","ĐBSCL"]
                crit=["grdp_per_capita_million_VND","fdi_registered_billion_USD",
                      "digital_index_0_100","ai_readiness_0_100","trained_labor_pct",
                      "rd_intensity_pct","internet_penetration_pct","gini_coef"]
                is_b=[True]*7+[False]; w=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
                X=df_regions[crit].values.astype(float)
                R2=X/np.sqrt((X**2).sum(0)); V=R2*w
                Ap=np.where(is_b,V.max(0),V.min(0)); An=np.where(is_b,V.min(0),V.max(0))
                Sp=np.sqrt(((V-Ap)**2).sum(1)); Sn=np.sqrt(((V-An)**2).sum(1)); C=Sn/(Sp+Sn)
                idx=np.argsort(C)[::-1]
                fig,ax=plt.subplots(figsize=(8,4),facecolor="#0e1117")
                ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
                for sp in ax.spines.values(): sp.set_color("#333")
                cols_r=["#ff4b4b","#ff9800","#4CAF50","#555","#555","#555"]
                ax.barh([regs[i] for i in idx[::-1]],[C[i] for i in idx[::-1]],
                        color=[cols_r[k] for k in range(5,-1,-1)],alpha=0.9)
                ax.set_title("TOPSIS – Xếp hạng sẵn sàng AI",color="white",fontsize=12)
                ax.set_xlabel("C* (gần 1 = tốt hơn)",color="white")
                plt.tight_layout(); st.pyplot(fig); plt.close()
                c1,c2,c3=st.columns(3)
                for i,col in enumerate([c1,c2,c3]):
                    col.metric(f"#{i+1} ưu tiên AI",regs[idx[i]],f"C*={C[idx[i]]:.4f}")
        else:
            st.info("👆 Click **Chạy M2** để xếp hạng 6 vùng theo TOPSIS")

    with tab2:
        st.markdown("## M3 — Tối ưu phân bổ ngân sách (LP 24 biến)")
        budget_m3=st.slider("Tổng ngân sách (tỷ VND)",30000,80000,50000,5000)
        if st.button("▶ Chạy M3", type="primary"):
            with st.spinner("Đang giải LP..."):
                R_lp,J_lp,N_lp=6,4,24
                beta_lp=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],
                                   [1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],
                                   [0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
                Al,bl=[],[]
                Al.append(np.ones(N_lp)); bl.append(budget_m3)
                for r in range(R_lp):
                    row=np.zeros(N_lp); row[r*J_lp:(r+1)*J_lp]=-1; Al.append(row); bl.append(-budget_m3/R_lp*0.5)
                for r in range(R_lp):
                    row=np.zeros(N_lp); row[r*J_lp:(r+1)*J_lp]=1; Al.append(row); bl.append(budget_m3/R_lp*1.8)
                row=np.zeros(N_lp)
                for r in range(R_lp): row[r*J_lp+3]=-1
                Al.append(row); bl.append(-budget_m3*0.24)
                res=linprog(-beta_lp.flatten(),A_ub=np.array(Al),b_ub=np.array(bl),bounds=[(0,None)]*N_lp,method="highs")
                if res.status==0:
                    x_opt=res.x.reshape(R_lp,J_lp); Z=-res.fun
                    st.success(f"✅ Z* = **{Z:,.0f} tỷ VND** GDP gain")
                    regs6=["TDMN-PB","ĐBSH","BTB+DHMT","Tây Nguyên","ĐNB","ĐBSCL"]
                    items4=["I Hạ tầng","D Số hóa","AI","H Nhân lực"]
                    df_alloc=pd.DataFrame(x_opt.round(0),index=regs6,columns=items4)
                    df_alloc["TỔNG"]=df_alloc.sum(axis=1)
                    st.dataframe(df_alloc,use_container_width=True)
                    fig,ax=plt.subplots(figsize=(10,4),facecolor="#0e1117")
                    ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
                    for sp in ax.spines.values(): sp.set_color("#333")
                    cols_j=["#1976D2","#FF9800","#9C27B0","#4CAF50"]
                    for j,(item,col) in enumerate(zip(items4,cols_j)):
                        ax.bar(np.arange(R_lp)+j*0.2,x_opt[:,j],0.2,label=item,color=col,alpha=0.85)
                    ax.set_xticks(np.arange(R_lp)+0.3); ax.set_xticklabels(regs6,rotation=20,color="white")
                    ax.set_title("Phân bổ tối ưu theo vùng",color="white",fontsize=12)
                    ax.set_ylabel("Tỷ VND",color="white"); ax.legend(facecolor="#1a1d27",labelcolor="white")
                    plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("👆 Click **Chạy M3** để giải LP phân bổ ngân sách")

    with tab3:
        st.markdown("## M6 — So sánh 5 kịch bản chính sách")
        if st.button("▶ Chạy M6 – So sánh 5 kịch bản", type="primary"):
            with st.spinner("Đang tính 5 kịch bản..."):
                scen_cfg={"S1 Truyền thống":{"alloc":[0.70,0.10,0.10,0.10],"D30":22,"AI30":90,"H30":31,"c":"#795548"},
                          "S2 Số hóa nhanh":{"alloc":[0.25,0.45,0.15,0.15],"D30":30,"AI30":95,"H30":33,"c":"#2196F3"},
                          "S3 AI dẫn dắt":  {"alloc":[0.20,0.20,0.45,0.15],"D30":26,"AI30":120,"H30":33,"c":"#9C27B0"},
                          "S4 Bao trùm số": {"alloc":[0.30,0.20,0.10,0.40],"D30":28,"AI30":95,"H30":37,"c":"#4CAF50"},
                          "S5 Tối ưu LP":   {"alloc":None,                  "D30":30,"AI30":110,"H30":35,"c":"#ff4b4b"}}
                rows2=[]; gdps=[]; gains=[]; short=[]; cols_sc=[]
                for sname,cfg in scen_cfg.items():
                    Ks,Ls,As=K[-1],L[-1],A_hist[-1]
                    for _ in range(5): Ks*=1.06; Ls*=1.01; As*=1.012
                    Dt=D[-1]+(cfg["D30"]-D[-1]); AIt=AI[-1]+(cfg["AI30"]-AI[-1]); Ht=H[-1]+(cfg["H30"]-H[-1])
                    gdp=As*Ks**a_*Ls**b_*Dt**g_*AIt**d_*Ht**t_; cagr=(gdp/Y[-1])**(1/5)-1
                    if cfg["alloc"] is None:
                        c_o=np.array([-0.85,-1.20,-0.95,-1.35])
                        A_u=np.array([[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]])
                        b_u=np.array([100,-25,-15,-20,-10,0])
                        r2=linprog(c_o,A_ub=A_u,b_ub=b_u,bounds=[(0,None)]*4,method="highs")
                        gain=-r2.fun if r2.success else 0
                    else:
                        gain=float(np.dot([0.85,1.20,0.95,1.35],np.array(cfg["alloc"])*100))
                    rows2.append({"Kịch bản":sname,"GDP 2030 (nghìn tỷ)":f"{gdp:,.0f}",
                                  "CAGR (%/năm)":f"{cagr*100:.2f}%","GDP gain (tỷ)":f"{gain:,.0f}"})
                    gdps.append(gdp); gains.append(gain); short.append(sname[:2]); cols_sc.append(cfg["c"])

                st.dataframe(pd.DataFrame(rows2),use_container_width=True,hide_index=True)
                fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor="#0e1117")
                for ax in axes:
                    ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
                    for sp in ax.spines.values(): sp.set_color("#333")
                lb=["S1","S2","S3","S4","S5*"]
                bars=axes[0].bar(lb,gdps,color=cols_sc,alpha=0.9,edgecolor="none")
                axes[0].axhline(Y[-1],color="#ff9800",ls="--",alpha=0.7,label=f"GDP 2025={Y[-1]:,.0f}")
                axes[0].set_title("GDP 2030 theo kịch bản",color="white",fontsize=12)
                axes[0].set_ylabel("nghìn tỷ VND",color="white"); axes[0].legend(facecolor="#1a1d27",labelcolor="white")
                for bar,v in zip(bars,gdps):
                    axes[0].text(bar.get_x()+bar.get_width()/2,v+50,f"{v:,.0f}",ha="center",color="white",fontsize=8,fontweight="bold")
                axes[1].bar(lb,gains,color=cols_sc,alpha=0.9,edgecolor="none")
                axes[1].set_title("GDP gain từ đầu tư số",color="white",fontsize=12); axes[1].set_ylabel("Tỷ VND",color="white")
                plt.tight_layout(); st.pyplot(fig); plt.close()
                best=list(scen_cfg.keys())[int(np.argmax(gains))]
                st.success(f"🏆 Kịch bản GDP gain cao nhất: **{best}**")
        else:
            st.info("👆 Click **Chạy M6** để so sánh 5 kịch bản")

    with tab4:
        st.markdown("## M4 — Mô phỏng lao động & M5 — Cảnh báo rủi ro")
        bl_m4=st.slider("Ngân sách lao động (tỷ VND)",10000,50000,30000,5000)
        if st.button("▶ Chạy M4+M5", type="primary"):
            with st.spinner("Đang mô phỏng..."):
                sec=["Nông-Lâm-TS","CN Chế biến","Xây dựng","Bán buôn-lẻ","Tài chính-NH","Logistics","CNTT-TT","Giáo dục"]
                risk2=np.array([0.18,0.42,0.25,0.38,0.52,0.35,0.28,0.22])
                a1j=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
                b1j=np.array([45.,28.,35.,32.,22.,30.,20.,55.])
                c1j=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
                N8=8; Nv2=2*N8; net=a1j-c1j*risk2
                c_o2=np.concatenate([-net,-b1j]); Al3,bl3=[],[]
                Al3.append(np.ones(Nv2)); bl3.append(bl_m4)
                for i in range(N8):
                    row=np.zeros(Nv2); row[i]=-net[i]; row[N8+i]=-b1j[i]; Al3.append(row); bl3.append(0)
                for i in range(N8):
                    row=np.zeros(Nv2); row[i]=c1j[i]*risk2[i]; row[N8+i]=-b1j[i]; Al3.append(row); bl3.append(0)
                rl=linprog(c_o2,A_ub=np.array(Al3),b_ub=np.array(bl3),bounds=[(0,None)]*Nv2,method="highs")
                if rl.status==0:
                    xAI2=rl.x[:N8]; xH2=rl.x[N8:]
                    NJ=net*xAI2+b1j*xH2; Disp=c1j*risk2*xAI2
                    c1m,c2m,c3m=st.columns(3)
                    c1m.metric("Tổng NetJob",f"{NJ.sum():,.0f}","việc làm ròng")
                    c2m.metric("Bị thay thế",f"{Disp.sum():,.0f}","việc")
                    c3m.metric("Đào tạo lại",f"{(b1j*xH2).sum():,.0f}","việc nâng cấp")
                    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor="#0e1117")
                    for ax in axes:
                        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
                        for sp in ax.spines.values(): sp.set_color("#333")
                    cols_nj=["#4CAF50" if v>0 else "#F44336" for v in NJ]
                    axes[0].barh(sec[::-1],NJ[::-1],color=cols_nj[::-1],alpha=0.9)
                    axes[0].axvline(0,color="white",lw=0.8)
                    axes[0].set_title("NetJob ròng theo ngành",color="white",fontsize=12)
                    axes[0].set_xlabel("Số việc làm ròng",color="white")
                    rsk_l=["Cyber risk","Automation","Dependency"]
                    rsk_v=[float(np.mean(risk2[:6])*0.8), float(np.mean(risk2)), 0.25]
                    rsk_c=["#F44336" if v>0.4 else "#FF9800" if v>0.2 else "#4CAF50" for v in rsk_v]
                    br2=axes[1].bar(rsk_l,rsk_v,color=rsk_c,alpha=0.9,width=0.5)
                    axes[1].axhline(0.4,color="#F44336",ls="--",alpha=0.6,label="Ngưỡng cao")
                    axes[1].axhline(0.2,color="#FF9800",ls="--",alpha=0.6,label="Ngưỡng TB")
                    axes[1].set_title("Cảnh báo rủi ro (M5)",color="white",fontsize=12)
                    axes[1].set_ylabel("Mức độ rủi ro [0-1]",color="white")
                    axes[1].legend(facecolor="#1a1d27",labelcolor="white")
                    for bar,v in zip(br2,rsk_v):
                        axes[1].text(bar.get_x()+bar.get_width()/2,v+0.01,f"{v:.3f}",ha="center",color="white",fontsize=11,fontweight="bold")
                    plt.tight_layout(); st.pyplot(fig); plt.close()
                    if NJ.min()<0:
                        st.error(f"🚨 Ngành {sec[int(np.argmin(NJ))]} có NetJob ÂM — cần tăng đào tạo lại ngay!")
                    else:
                        st.success("✅ Tất cả ngành đều có NetJob DƯƠNG — chính sách đang cân bằng")
        else:
            st.info("👆 Click **Chạy M4+M5** để phân tích rủi ro lao động")

# ── CÁC BÀI KHÁC chỉ hiển thị placeholder ───────────────────────
elif page == "🗺️ Bài 4 — LP ngành-vùng":
    exec(open("bai04_lp_region.py", encoding="utf-8").read())

elif page == "📋 Bài 5 — MIP 15 dự án":
    exec(open("bai05_mip_projects.py", encoding="utf-8").read())

elif page == "🎯 Bài 7 — Pareto đa mục tiêu":
    exec(open("bai07_pareto.py", encoding="utf-8").read())

elif page == "⏳ Bài 8 — Động 2026-2035":
    exec(open("bai08_dynamic.py", encoding="utf-8").read())

elif page == "👷 Bài 9 — Lao động & AI":
    exec(open("bai09_labor.py", encoding="utf-8").read())

elif page == "🎲 Bài 10 — Stochastic SP":
    exec(open("bai10_stochastic.py", encoding="utf-8").read())

elif page == "🤖 Bài 11 — Q-learning RL":
    exec(open("bai11_qlearning.py", encoding="utf-8").read())
Bài này đã được giải đầy đủ trong file **`bai{num:02d}_*.py`** trong bộ file đã tải.

**Để chạy trên Colab:**
1. Mở file `bai{num:02d}_*.py` bằng TextEdit
2. **Cmd+A** → **Cmd+C**
3. Dán vào cell Colab → **Shift+Enter**
""")
        with st.expander("Xem mô hình toán học"):
            models={
                4:"**max Z = Σ β_{jr}·x_{jr}**\n- 24 biến: x[6 vùng][4 hạng mục]\n- Ràng buộc C1-C5: ngân sách, sàn/trần vùng, công bằng số hóa",
                5:"**max Σ B_i·y_i**, y_i ∈ {0,1}\n- 15 biến nhị phân\n- Ràng buộc: ngân sách, loại trừ, tiên quyết, cân đối lĩnh vực",
                7:"**max f₁ (GDP), min f₂ (BĐ), min f₃ (phát thải), min f₄ (rủi ro)**\n- 200 nghiệm Pareto\n- Chọn nghiệm thỏa hiệp bằng TOPSIS",
                8:"**max Σ ρ^t·ln(C_t)**, ρ=0.97, T=10 năm\n- 50 biến: I_K, I_D, I_AI, I_H, C mỗi năm\n- Ràng buộc ngân sách: C_t + ΣI ≤ Y_t",
                9:"**max Σ NetJob_i = NewJob + UpgradeJob − DisplacedJob**\n- 16 biến: x_AI, x_H × 8 ngành\n- Ràng buộc: Displaced ≤ RetrainingCapacity",
                10:"**max β·x + Σ p_s·β_s·y_s**\n- x ≤ 65,000 tỷ (first-stage)\n- y^s ≤ 15,000 tỷ (second-stage)\n- y_AI^s ≤ 0.5·x_H",
                11:"**Q-Learning**: Q(s,a) += α[r + γ·max Q(s',a') − Q(s,a)]\n- 81 trạng thái (3⁴), 5 hành động\n- α=0.1, γ=0.95, ε giảm từ 1.0→0.05",
            }
            st.code(models.get(num,"Xem file .py"), language="")

# ═══ PHẦN MỞ RỘNG – GHI ĐÈ PHẦN PLACEHOLDER ═══
# (không dùng – đã xử lý ở phần else phía trên)
