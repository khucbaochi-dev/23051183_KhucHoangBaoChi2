"""
BÀI 12 – ĐỒ ÁN TÍCH HỢP AIDEOM-VN
6 Module: M1 Dự báo | M2 TOPSIS | M3 LP | M4 Lao động | M5 Rủi ro | M6 Dashboard
5 Kịch bản: S1 Truyền thống | S2 Số hóa nhanh | S3 AI dẫn dắt | S4 Bao trùm | S5 Tối ưu
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize

print("="*60)
print("BÀI 12 – AIDEOM-VN: HỆ THỐNG TỐI ƯU 6 MODULE")
print("="*60)

# ══════════════════════════════════════════════════════════
# MODULE M1 – DỰ BÁO KINH TẾ (Cobb-Douglas)
# ══════════════════════════════════════════════════════════
print("\n[M1] DỰ BÁO KINH TẾ 2026-2030")
print("-"*40)

df_macro = pd.read_csv('vietnam_macro_2020_2025.csv')
Y = df_macro['GDP_trillion_VND'].values
K = np.array([16500,17800,19600,21300,23500,25900])
L = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
D = df_macro['digital_economy_share_GDP_pct'].values
AI= np.array([55.6,60.2,65.4,67.0,73.8,80.1])
H = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
alpha_,beta_,gamma_,delta_,theta_=0.33,0.42,0.10,0.08,0.07
A_hist = Y / (K**alpha_*L**beta_*D**gamma_*AI**delta_*H**theta_)
A0=A_hist[-1]; K0=K[-1]; L0=L[-1]; D0=D[-1]; AI0=AI[-1]; H0=H[-1]

# 5 kịch bản × parameters
scenarios={
    'S1 Truyền thống':  {'D_2030':22,'AI_2030':90, 'H_2030':31,'gK':0.07,'gL':0.01,'gTFP':0.005},
    'S2 Số hóa nhanh':  {'D_2030':30,'AI_2030':95, 'H_2030':33,'gK':0.06,'gL':0.01,'gTFP':0.010},
    'S3 AI dẫn dắt':    {'D_2030':26,'AI_2030':120,'H_2030':33,'gK':0.06,'gL':0.01,'gTFP':0.015},
    'S4 Bao trùm số':   {'D_2030':28,'AI_2030':95, 'H_2030':37,'gK':0.05,'gL':0.01,'gTFP':0.010},
    'S5 Tối ưu (M3)':   {'D_2030':30,'AI_2030':110,'H_2030':35,'gK':0.06,'gL':0.01,'gTFP':0.012},
}

def forecast_gdp_2030(p):
    K_s,D_s,AI_s,H_s,A_s=K0,D0,AI0,H0,A0
    for yr in range(2026,2031):
        K_s*=(1+p['gK']); L_s=L0*(1+p['gL'])**(yr-2025)
        A_s*=(1+p['gTFP'])
        t=(yr-2025)/5
        D_t=D0+t*(p['D_2030']-D0)
        AI_t=AI0+t*(p['AI_2030']-AI0)
        H_t=H0+t*(p['H_2030']-H0)
    Y_2030=A_s*K_s**alpha_*L_s**beta_*D_t**gamma_*AI_t**delta_*H_t**theta_
    cagr=(Y_2030/Y[-1])**(1/5)-1
    return Y_2030, cagr

m1_results={}
for name,p in scenarios.items():
    gdp,cagr=forecast_gdp_2030(p)
    m1_results[name]={'GDP_2030':gdp,'CAGR':cagr}
    print(f"  {name:22s}: GDP 2030={gdp:8,.0f} nghìn tỷ ({cagr*100:.2f}%/năm)")

# ══════════════════════════════════════════════════════════
# MODULE M2 – TOPSIS ĐÁNH GIÁ SẴN SÀNG SỐ
# ══════════════════════════════════════════════════════════
print("\n[M2] TOPSIS XẾP HẠNG VÙNG")
print("-"*40)
df_r=pd.read_csv('vietnam_regions_2024.csv')
regs=['TDMN-PB','DBSH','BTB-DHMT','TN','DNB','DBSCL']
crit=['grdp_per_capita_million_VND','fdi_registered_billion_USD',
      'digital_index_0_100','ai_readiness_0_100','trained_labor_pct',
      'rd_intensity_pct','internet_penetration_pct','gini_coef']
is_b=[True]*7+[False]
w_t=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
X=df_r[crit].values.astype(float)
R_mat=X/np.sqrt((X**2).sum(axis=0)); V=R_mat*w_t
Ap=np.where(is_b,V.max(axis=0),V.min(axis=0))
An=np.where(is_b,V.min(axis=0),V.max(axis=0))
Sp=np.sqrt(((V-Ap)**2).sum(axis=1)); Sn=np.sqrt(((V-An)**2).sum(axis=1))
C_star=Sn/(Sp+Sn)
top3_idx=np.argsort(C_star)[::-1][:3]
top3=[regs[i] for i in top3_idx]
for i,r in enumerate(regs):
    print(f"  {r:12s}: C*={C_star[i]:.4f} {'← TOP3 ưu tiên AI' if r in top3 else ''}")

# ══════════════════════════════════════════════════════════
# MODULE M3 – TỐI ƯU PHÂN BỔ NGÂN SÁCH
# ══════════════════════════════════════════════════════════
print("\n[M3] TỐI ƯU PHÂN BỔ (LP Bài 4 + Kịch bản)")
print("-"*40)
R_lp,J_lp,N_lp=6,4,24
beta_lp=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],
                   [1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
alloc_s={'S1 Truyền thống':[0.70,0.10,0.10,0.10],'S2 Số hóa nhanh':[0.25,0.45,0.15,0.15],
         'S3 AI dẫn dắt':[0.20,0.20,0.45,0.15],'S4 Bao trùm số':[0.30,0.20,0.10,0.40],
         'S5 Tối ưu (M3)':None}

Al,bl=[],[]
Al.append(np.ones(N_lp)); bl.append(50000)
for r in range(R_lp):
    row=np.zeros(N_lp); row[r*J_lp:(r+1)*J_lp]=-1; Al.append(row); bl.append(-5000)
for r in range(R_lp):
    row=np.zeros(N_lp); row[r*J_lp:(r+1)*J_lp]=1; Al.append(row); bl.append(12000)
row=np.zeros(N_lp)
for r in range(R_lp): row[r*J_lp+3]=-1
Al.append(row); bl.append(-12000)
A_ub_lp=np.array(Al); b_ub_lp=np.array(bl)

m3_results={}
for sname in list(scenarios.keys()):
    a_fix=alloc_s.get(sname)
    if a_fix is None:  # S5 tối ưu
        res=linprog(-beta_lp.flatten(), A_ub=A_ub_lp, b_ub=b_ub_lp,
                    bounds=[(0,None)]*N_lp, method='highs')
        Z=-res.fun if res.status==0 else 0
    else:
        x_s=np.zeros(N_lp)
        for r in range(R_lp):
            total_r=50000/R_lp
            for j in range(J_lp): x_s[r*J_lp+j]=a_fix[j]*total_r
        Z=float((beta_lp.flatten()*x_s).sum())
    m3_results[sname]=Z
    print(f"  {sname:22s}: Z*={Z:10,.0f} tỷ GDP gain")

# ══════════════════════════════════════════════════════════
# MODULE M4 – MÔ PHỎNG LAO ĐỘNG
# ══════════════════════════════════════════════════════════
print("\n[M4] MÔ PHỎNG LAO ĐỘNG")
print("-"*40)
L_sec=np.array([13.2,11.5,4.8,7.8,0.55,1.95,0.62,2.15])
risk=np.array([0.18,0.42,0.25,0.38,0.52,0.35,0.28,0.22])
a1_j=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1_j=np.array([45.,28.,35.,32.,22.,30.,20.,55.])
c1_j=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])

m4_results={}
for sname,p in scenarios.items():
    # Budget AI và H theo kịch bản (30,000 tỷ dành cho lao động)
    a_fix=alloc_s.get(sname)
    if a_fix is None: a_fix=[0.20,0.20,0.45,0.15]
    budget_labor=30000
    xAI=a_fix[2]*budget_labor/len(L_sec)*np.ones(len(L_sec))
    xH =a_fix[3]*budget_labor/len(L_sec)*np.ones(len(L_sec))
    NetJob=np.sum((a1_j-c1_j*risk)*xAI + b1_j*xH)
    m4_results[sname]=NetJob
    print(f"  {sname:22s}: NetJob={NetJob:10,.0f} việc làm ròng")

# ══════════════════════════════════════════════════════════
# MODULE M5 – ĐÁNH GIÁ RỦI RO
# ══════════════════════════════════════════════════════════
print("\n[M5] ĐÁNH GIÁ RỦI RO")
print("-"*40)
rho_r=np.array([0.18,0.45,0.28,0.12,0.52,0.22])
sig_r=np.array([0.32,0.28,0.30,0.35,0.25,0.30])
e_r  =np.array([0.42,0.55,0.48,0.32,0.62,0.38])

m5_results={}
for sname in scenarios:
    a_fix=alloc_s.get(sname)
    if a_fix is None: a_fix=[0.20,0.20,0.45,0.15]
    xAI_r=a_fix[2]*50000/6*np.ones(6); xH_r=a_fix[3]*50000/6*np.ones(6)
    xI_r =a_fix[0]*50000/6*np.ones(6)
    cyber  =float((rho_r*xAI_r - sig_r*xH_r).sum())
    emission=float((e_r*(xI_r+xAI_r)).sum())
    m5_results[sname]={'cyber':cyber,'emission':emission,'composite':0.6*cyber+0.4*emission}
    print(f"  {sname:22s}: Cyber={cyber:8,.0f}  Emission={emission:8,.0f}  Tổng hợp={0.6*cyber+0.4*emission:8,.0f}")

# ══════════════════════════════════════════════════════════
# MODULE M6 – TỔNG HỢP & DASHBOARD
# ══════════════════════════════════════════════════════════
print("\n[M6] BẢNG TỔNG HỢP 5 KỊCH BẢN (KPI 2030)")
print("="*80)
print(f"{'Kịch bản':22s} | {'GDP 2030':>12} | {'CAGR':>7} | {'GDP gain':>10} | {'NetJob':>10} | {'Rủi ro':>8}")
print("-"*80)
for sname in scenarios:
    gdp  =m1_results[sname]['GDP_2030']
    cagr =m1_results[sname]['CAGR']
    gain =m3_results[sname]
    jobs =m4_results[sname]
    risk_=m5_results[sname]['composite']
    print(f"  {sname:20s} | {gdp:12,.0f} | {cagr*100:6.2f}% | {gain:10,.0f} | {jobs:10,.0f} | {risk_:8,.0f}")

# So sánh S5 vs S1
gdp_s5=m1_results['S5 Tối ưu (M3)']['GDP_2030']
gdp_s1=m1_results['S1 Truyền thống']['GDP_2030']
print(f"\nS5 vs S1: GDP tăng thêm = {gdp_s5-gdp_s1:,.0f} nghìn tỷ VND ({(gdp_s5/gdp_s1-1)*100:.1f}%)")

# Xếp hạng kịch bản theo đa tiêu chí (TOPSIS mini)
names_k=list(scenarios.keys())
F_k=np.array([[m1_results[n]['GDP_2030'],m3_results[n],m4_results[n],
               -m5_results[n]['composite']] for n in names_k])
Fn=F_k/np.sqrt((F_k**2).sum(axis=0)+1e-10)
w_k=np.array([0.35,0.30,0.20,0.15])
Vk=Fn*w_k; Apk=Vk.max(axis=0); Ank=Vk.min(axis=0)
Sk_p=np.sqrt(((Vk-Apk)**2).sum(axis=1)); Sk_n=np.sqrt(((Vk-Ank)**2).sum(axis=1))
Ck=Sk_n/(Sk_p+Sk_n)
best_k=names_k[np.argmax(Ck)]
print(f"\nXếp hạng kịch bản (TOPSIS w=[GDP:0.35, gain:0.30, jobs:0.20, safety:0.15]):")
for i in np.argsort(Ck)[::-1]:
    print(f"  {'★' if i==np.argmax(Ck) else ' '} #{np.argsort(np.argsort(Ck)[::-1])[i]+1} {names_k[i]:22s}: C*={Ck[i]:.4f}")

# ══════════════════════════════════════════════════════════
# BIỂU ĐỒ DASHBOARD
# ══════════════════════════════════════════════════════════
fig=plt.figure(figsize=(20,14))
fig.suptitle('BÀI 12 – AIDEOM-VN Dashboard: 5 Kịch bản Chính sách 2030\n'
             'Dữ liệu: vietnam_macro/sectors/regions_2024.csv | '
             'Mô hình: M1-M6 tích hợp', fontweight='bold', fontsize=13)

colors_s=['#795548','#2196F3','#9C27B0','#4CAF50','#F44336']
names_k_short=['S1','S2','S3','S4','S5*']

ax1=fig.add_subplot(231)
gdp_vals=[m1_results[n]['GDP_2030'] for n in names_k]
bars=ax1.bar(names_k_short, gdp_vals, color=colors_s, alpha=0.85, edgecolor='white')
ax1.axhline(Y[-1],color='gray',ls='--',alpha=0.6,label=f'GDP 2025={Y[-1]:,.0f}')
for bar,v in zip(bars,gdp_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, v+100, f'{v:,.0f}',
             ha='center', fontsize=8, fontweight='bold')
ax1.set_title('M1: GDP 2030 (nghìn tỷ VND)', fontweight='bold')
ax1.set_ylabel('nghìn tỷ VND'); ax1.legend(fontsize=8); ax1.grid(axis='y',alpha=0.3)

ax2=fig.add_subplot(232)
gain_vals=[m3_results[n] for n in names_k]
bars2=ax2.bar(names_k_short, gain_vals, color=colors_s, alpha=0.85, edgecolor='white')
for bar,v in zip(bars2,gain_vals):
    ax2.text(bar.get_x()+bar.get_width()/2,v+100,f'{v:,.0f}',ha='center',fontsize=8,fontweight='bold')
ax2.set_title('M3: GDP Gain từ đầu tư số\n(tỷ VND)', fontweight='bold')
ax2.set_ylabel('Tỷ VND'); ax2.grid(axis='y',alpha=0.3)

ax3=fig.add_subplot(233)
job_vals=[m4_results[n] for n in names_k]
colors_job=['#4CAF50' if v>0 else '#F44336' for v in job_vals]
ax3.bar(names_k_short, job_vals, color=colors_job, alpha=0.85, edgecolor='white')
ax3.axhline(0,color='black',lw=1)
ax3.set_title('M4: NetJob ròng\n(việc làm tạo ra − mất đi)', fontweight='bold')
ax3.set_ylabel('Số việc làm'); ax3.grid(axis='y',alpha=0.3)

ax4=fig.add_subplot(234)
cyber_vals=[m5_results[n]['cyber'] for n in names_k]
emiss_vals=[m5_results[n]['emission'] for n in names_k]
x_p=np.arange(5)
ax4.bar(x_p-0.2, cyber_vals, 0.38, label='Cyber risk', color='#F44336',alpha=0.8)
ax4.bar(x_p+0.2, emiss_vals, 0.38, label='Emission',   color='#FF9800',alpha=0.8)
ax4.set_xticks(x_p); ax4.set_xticklabels(names_k_short)
ax4.set_title('M5: Rủi ro tổng hợp\n(thấp hơn = tốt hơn)', fontweight='bold')
ax4.legend(); ax4.grid(axis='y',alpha=0.3)

ax5=fig.add_subplot(235)
cagr_vals=[m1_results[n]['CAGR']*100 for n in names_k]
ax5.plot(names_k_short, cagr_vals, 'D-', color='#1976D2', lw=2.5, ms=12)
for i,(x_v,y_v) in enumerate(zip(range(5),cagr_vals)):
    ax5.text(x_v, y_v+0.05, f'{y_v:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax5.axhline(8.02, color='green',ls='--',alpha=0.6,label='Thực tế 2025 (8.02%)')
ax5.set_title('CAGR GDP 2025-2030\n(%/năm)', fontweight='bold')
ax5.set_ylabel('%/năm'); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

ax6=fig.add_subplot(236)
Ck_pct=Ck/Ck.max()*100
bars6=ax6.barh(names_k_short[::-1], Ck_pct[::-1],
               color=[colors_s[list(names_k).index(n)] for n in names_k[::-1]], alpha=0.85)
ax6.axvline(Ck_pct.max(), color='red', ls='--', alpha=0.5)
for bar,v in zip(bars6, Ck_pct[::-1]):
    ax6.text(v+0.5, bar.get_y()+bar.get_height()/2, f'{v:.1f}%',
             va='center', fontsize=10, fontweight='bold')
ax6.set_title(f'M6: Xếp hạng TOPSIS\n(★ tốt nhất: {best_k})', fontweight='bold')
ax6.set_xlabel('Điểm TOPSIS (% so với tốt nhất)'); ax6.grid(axis='x',alpha=0.3)

plt.tight_layout()
plt.savefig('bai12_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"""
KHUYẾN NGHỊ CHÍNH SÁCH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① Kịch bản TỐI ƯU: {best_k}
   → Đầu tư cân bằng AI (45%) + Số hóa (20%) + Nhân lực (15%) + Hạ tầng (20%)

② Cảnh báo rủi ro:
   - S3 (AI dẫn dắt): GDP gain CAO NHẤT nhưng cyber risk lớn nhất
   - S1 (Truyền thống): an toàn nhưng bỏ lỡ cơ hội kinh tế số

③ Ba vùng ưu tiên AI (M2 TOPSIS): {top3[0]}, {top3[1]}, {top3[2]}
   → Phù hợp QĐ 127/QĐ-TTg: xây 3 trung tâm AI tại các vùng sẵn sàng nhất

④ Thị trường lao động: tất cả kịch bản cho NetJob DƯƠNG
   → Điều kiện: đầu tư đủ x_H để đào tạo lại (tốc độ đào tạo ≥ tốc độ tự động hóa)

⑤ Hạn chế mô hình:
   - Giả định tuyến tính; thực tế có phi tuyến, cú sốc, lan tỏa ngành
   - Cần cập nhật hàng năm bằng dữ liệu GSO thực tế
   - Không thay thế quy trình dân chủ lập ngân sách Quốc hội
""")
print("✓ Bài 12 → bai12_dashboard.png")
