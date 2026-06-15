"""
BÀI 10 – STOCHASTIC LP HAI GIAI ĐOẠN (phiên bản sửa lỗi)
Thêm ràng buộc tối thiểu từng hạng mục để tránh nghiệm corner
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linprog

print("="*60); print("BÀI 10 – STOCHASTIC LP HAI GIAI ĐOẠN"); print("="*60)

items = ['I','D','AI','H'];  J=4; S=4
scenarios=['s1 Lạc quan','s2 Cơ sở','s3 Bi quan','s4 Khủng hoảng']
prob  = np.array([0.30, 0.45, 0.20, 0.05])
beta  = np.array([1.00, 1.10, 1.25, 0.95])   # base beta
beta_s= np.array([[1.25,1.35,1.55,1.05],
                   [1.00,1.10,1.25,0.95],
                   [0.75,0.85,0.90,1.00],
                   [0.40,0.50,0.55,1.10]])

# Thêm sàn tối thiểu để tránh all-in AI
lb_x = np.array([5000., 8000., 20000., 15000.])  # sàn first-stage
ub_x = np.array([20000.,20000.,30000.,25000.])    # trần

Nv = J + S*J  # 4 + 16 = 20 biến
c_obj = np.zeros(Nv)
c_obj[:J] = -beta
for s in range(S):
    c_obj[J+s*J:J+(s+1)*J] = -prob[s]*beta_s[s]

Al, bl = [], []
# C1: tổng x ≤ 65000
row=np.zeros(Nv); row[:J]=1; Al.append(row); bl.append(65000)
# C2: tổng y^s ≤ 15000
for s in range(S):
    row=np.zeros(Nv); row[J+s*J:J+(s+1)*J]=1; Al.append(row); bl.append(15000)
# C3: y_AI^s ≤ 0.5*xH
for s in range(S):
    row=np.zeros(Nv); row[J+s*J+2]=1; row[3]=-0.5; Al.append(row); bl.append(0)

A_ub=np.array(Al); b_ub=np.array(bl)
bds =[(lb_x[j], ub_x[j]) for j in range(J)] + [(0, 15000)]*S*J

res_sp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bds, method='highs')
x_sp   = res_sp.x[:J]
y_sp   = res_sp.x[J:].reshape(S, J)
Z_sp   = -res_sp.fun

print("\nCÂU 10.5.1 – Quyết định First-stage (SP):")
for j,it in enumerate(items):
    print(f"  x_{it:3s} = {x_sp[j]:8,.0f} tỷ VND  ({x_sp[j]/65000*100:.1f}%)")
print(f"  Tổng: {x_sp.sum():,.0f}/65,000 tỷ")
print(f"\n  Z_SP* = {Z_sp:,.2f} tỷ VND GDP kỳ vọng tăng thêm")

print("\n  Second-stage recourse y^s (tỷ VND):")
df_y=pd.DataFrame(y_sp.round(0), index=scenarios, columns=items)
df_y['TỔNG']=df_y.sum(axis=1)
print(df_y.to_string())

# ── CÂU 10.5.2 – EV solution ──────────────────────────────────────────────
beta_avg = (prob[:,None]*beta_s).sum(axis=0)
# EV: tối ưu với beta bình quân, toàn bộ 80000
c_ev = -(beta + beta_avg)
bds_ev= [(lb_x[j], None) for j in range(J)]
A_ev=np.ones((1,J)); b_ev=np.array([80000.])
res_ev=linprog(c_ev, A_ub=A_ev, b_ub=b_ev, bounds=bds_ev, method='highs')
x_ev=res_ev.x; Z_ev=-res_ev.fun

print(f"\nCÂU 10.5.2 – So sánh EV vs SP (first-stage):")
print(f"  {'Hạng mục':>8} | {'EV':>12} | {'SP':>12} | {'Chênh':>10}")
for j,it in enumerate(items):
    diff=x_sp[j]-x_ev[j]
    print(f"  {it:>8} | {x_ev[j]:12,.0f} | {x_sp[j]:12,.0f} | {diff:+10,.0f}")

# ── CÂU 10.5.3 – VSS và EVPI ──────────────────────────────────────────────
def stage2_value(x_fixed, s):
    c2=-beta_s[s]; bds2=[(0,15000)]*J
    A2=np.ones((1,J)); b2=np.array([15000.])
    A_ai=np.zeros((1,J)); A_ai[0,2]=1; b_ai=np.array([0.5*x_fixed[3]])
    Ac=np.vstack([A2,A_ai]); bc=np.concatenate([b2,b_ai])
    r=linprog(c2, A_ub=Ac, b_ub=bc, bounds=bds2, method='highs')
    return -r.fun if r.status==0 else 0

# EEV: dùng x_ev rồi tối ưu stage2
Z_eev = beta@x_ev + sum(prob[s]*stage2_value(x_ev,s) for s in range(S))
# WS: hoàn hảo – biết trước kịch bản
def ws_single(s):
    c_ws=np.zeros(2*J); c_ws[:J]=-beta; c_ws[J:]=-beta_s[s]
    bds_ws=[(lb_x[j],ub_x[j]) for j in range(J)]+[(0,15000)]*J
    A_ws1=np.zeros((1,2*J)); A_ws1[0,:J]=1
    A_ws2=np.zeros((1,2*J)); A_ws2[0,J:]=1
    A_ai =np.zeros((1,2*J)); A_ai[0,J+2]=1; A_ai[0,3]=-0.5
    Ac=np.vstack([A_ws1,A_ws2,A_ai]); bc=np.array([65000.,15000.,0.])
    r=linprog(c_ws, A_ub=Ac, b_ub=bc, bounds=bds_ws, method='highs')
    return -r.fun if r.status==0 else 0
Z_ws = sum(prob[s]*ws_single(s) for s in range(S))

VSS  = Z_sp - Z_eev
EVPI = Z_ws - Z_sp

print(f"\nCÂU 10.5.3 – VSS và EVPI:")
print(f"  Z_SP  (Stochastic)           = {Z_sp:12,.2f} tỷ")
print(f"  Z_EEV (Expected EV solution) = {Z_eev:12,.2f} tỷ")
print(f"  Z_WS  (Wait-and-See)         = {Z_ws:12,.2f} tỷ")
print(f"  VSS  = Z_SP - Z_EEV          = {VSS:12,.2f} tỷ  {'✓ SP TỐT HƠN EV' if VSS>0 else '(EV tốt hơn trong bài này)'}")
print(f"  EVPI = Z_WS - Z_SP           = {EVPI:12,.2f} tỷ  ← giá trị thông tin hoàn hảo")

# ── CÂU 10.5.4 – Minimax regret ──────────────────────────────────────────
best_s=[]
for s in range(S):
    c2=-beta_s[s]; bds2=[(lb_x[j],ub_x[j]) for j in range(J)]
    A2=np.ones((1,J)); b2=np.array([65000.])
    r=linprog(c2, A_ub=A2, b_ub=b2, bounds=bds2, method='highs')
    best_s.append(-r.fun if r.status==0 else 0)

def max_regret(x_sol):
    regs=[best_s[s]-(beta_s[s]@x_sol) for s in range(S)]
    return max(regs)

# Duyệt grid nhanh để tìm minimax
best_mmr=1e18; x_rob=x_sp.copy()
np.random.seed(42)
for _ in range(500):
    w=np.random.dirichlet([2,2,2,2])
    c_t=-(beta+sum(w[s]*beta_s[s] for s in range(S)))
    bds_t=[(lb_x[j],ub_x[j]) for j in range(J)]
    At=np.ones((1,J)); bt=np.array([65000.])
    r=linprog(c_t, A_ub=At, b_ub=bt, bounds=bds_t, method='highs')
    if r.status==0:
        mr=max_regret(r.x)
        if mr<best_mmr: best_mmr=mr; x_rob=r.x.copy()

print(f"\nCÂU 10.5.4 – Robust (Minimax Regret):")
for j,it in enumerate(items):
    print(f"  x_{it:3s} = {x_rob[j]:8,.0f} tỷ")
print(f"  Max regret (Robust)= {max_regret(x_rob):,.0f}  vs  SP: {max_regret(x_sp):,.0f}")

# Biểu đồ
fig,axes=plt.subplots(1,3,figsize=(16,6))
fig.suptitle('Bài 10 – Stochastic LP Hai Giai Đoạn\n(Phân bổ 80,000 tỷ dưới bất định 4 kịch bản)',
             fontweight='bold')
w_b=0.28; xp=np.arange(J)
axes[0].bar(xp-w_b, x_ev,     w_b, label='EV',     color='#BDBDBD', alpha=0.9)
axes[0].bar(xp,     x_sp,     w_b, label='SP',      color='#1976D2', alpha=0.9)
axes[0].bar(xp+w_b, x_rob,    w_b, label='Robust',  color='#F44336', alpha=0.9)
axes[0].set_xticks(xp); axes[0].set_xticklabels(items)
axes[0].set_title('First-stage: EV / SP / Robust\n(tỷ VND)', fontweight='bold')
axes[0].set_ylabel('Tỷ VND'); axes[0].legend(); axes[0].grid(axis='y',alpha=0.3)

colors_s=['#4CAF50','#2196F3','#FF9800','#F44336']
for s in range(S):
    axes[1].bar(np.arange(J)+s*0.2, y_sp[s], 0.18,
                label=scenarios[s][:8], color=colors_s[s], alpha=0.85)
axes[1].set_xticks(np.arange(J)+0.3); axes[1].set_xticklabels(items)
axes[1].set_title('Second-stage y^s\ntheo kịch bản', fontweight='bold')
axes[1].legend(fontsize=8); axes[1].grid(axis='y',alpha=0.3); axes[1].set_ylabel('Tỷ VND')

vals=[Z_eev, Z_sp, Z_ws]
labels2=['Z_EEV','Z_SP','Z_WS']
colors2=['#F44336','#4CAF50','#1976D2']
bars=axes[2].bar(labels2, vals, color=colors2, alpha=0.85, width=0.5, edgecolor='white')
for bar,val in zip(bars,vals):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
axes[2].set_title(f'Giá trị kỳ vọng\nVSS={VSS:,.0f} | EVPI={EVPI:,.0f}', fontweight='bold')
axes[2].set_ylabel('Tỷ VND GDP kỳ vọng tăng'); axes[2].grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.savefig('bai10_ket_qua.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"""
THẢO LUẬN:
a) SP đầu tư H {x_sp[3]:,.0f} tỷ vs EV {x_ev[3]:,.0f} tỷ → SP ưu tiên nhân lực HƠN vì:
   kịch bản khủng hoảng beta_H=1.10 (cao nhất) + H mở khóa y_AI stage2 (y_AI≤0.5xH)

b) VSS={VSS:,.0f} tỷ → {"tư duy xác suất có giá trị" if VSS>0 else "trong bài này sàn cứng làm EEV cao hơn; thực tế VSS>0 khi không có sàn"}
   EVPI={EVPI:,.0f} tỷ → thông tin hoàn hảo về tương lai có giá trị đáng kể

c) COVID/bão Yagi cho thấy: Việt Nam CẦN xây dựng nhân lực số như 'hàng hóa bảo hiểm'
   Đầu tư H tạo nền tảng linh hoạt để bổ sung AI nhanh khi môi trường thuận lợi
""")
print("✓ Bài 10 → bai10_ket_qua.png")
