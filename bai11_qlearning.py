"""
BÀI 11 – Q-LEARNING (phiên bản hoàn chỉnh)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib

print("="*60); print("BÀI 11 – Q-LEARNING CHÍNH SÁCH KINH TẾ THÍCH NGHI"); print("="*60)

alpha_p,beta_p,gamma_p,delta_p,theta_p=0.33,0.42,0.10,0.08,0.07
alloc=np.array([[0.70,0.10,0.10,0.10],[0.40,0.25,0.15,0.20],
                [0.25,0.45,0.15,0.15],[0.20,0.20,0.45,0.15],[0.30,0.20,0.10,0.40]])
anames=['Truyền thống','Cân bằng','Số hóa nhanh','AI dẫn dắt','Bao trùm']
w_r=np.array([0.40,0.25,0.20,0.15]); BUDGET=1000.; T_ep=10
K0,D0,AI0,H0,A0=27500.,20.3,86.,30.,34.9

def disc(g,D,AI,U):
    s0=0 if g<0.05 else(1 if g<0.09 else 2)
    s1=0 if D<25 else(1 if D<35 else 2)
    s2=0 if AI<60 else(1 if AI<90 else 2)
    s3=2 if U>0.50 else(1 if U>0.25 else 0)
    return (s0,s1,s2,s3)

def step(K,D,AI,H,A,action):
    a=alloc[action]; L0=54.0
    Y0=A*K**alpha_p*L0**beta_p*D**gamma_p*AI**delta_p*H**theta_p
    K2=0.95*K+a[0]*BUDGET; D2=0.88*D+a[1]*BUDGET/100
    AI2=0.85*AI+a[2]*BUDGET/20; H2=0.98*H+a[3]*BUDGET/200
    A2=A*(1+0.003*D+0.002*AI+0.004*H)
    Y2=A2*K2**alpha_p*L0**beta_p*D2**gamma_p*AI2**delta_p*H2**theta_p
    gdp_g=(Y2-Y0)/Y0; U=max(0,0.40-0.005*H2+0.003*AI2/10)
    cyber=0.1+0.002*AI2/10-0.001*H2; emiss=a[0]*0.5+a[2]*0.3
    r=w_r[0]*gdp_g*10-w_r[1]*max(0,U-0.3)-w_r[2]*cyber-w_r[3]*emiss
    return K2,D2,AI2,H2,A2,disc(gdp_g,D2,AI2,U),r

# ── CÂU 11.3.2 – Training ──────────────────────────────────────────────────
Q=np.zeros((3,3,3,3,5)); lr=0.1; gq=0.95; n_ep=20000
ep_rew=[]; np.random.seed(0)
print(f"Training {n_ep} episodes (T={T_ep} năm/ep)...")

for ep in range(n_ep):
    eps=max(0.05, 1.0-ep/8000)
    K,D,AI,H,A=K0,D0,AI0,H0,A0
    s=disc(0.08,D,AI,0.30); tot=0.
    for _ in range(T_ep):
        a=np.random.randint(5) if np.random.rand()<eps else int(np.argmax(Q[s]))
        K,D,AI,H,A,s2,r=step(K,D,AI,H,A,a)
        Q[s+(a,)]+=lr*(r+gq*Q[s2].max()-Q[s+(a,)])
        s=s2; tot+=r
    ep_rew.append(tot)
    if (ep+1)%5000==0:
        print(f"  ep={ep+1:5d} eps={eps:.3f} avg-500={np.mean(ep_rew[-500:]):.4f}")

print("Hoàn thành!")

# ── CÂU 11.3.3 ─────────────────────────────────────────────────────────────
print("\nCÂU 11.3.3 – Chính sách π*(s):")
states={
    "VN 2026 thực tế (GDP-mid,D-low,AI-high,U-mid)":(1,0,1,1),
    "Khủng hoảng (GDP-low,D-low,AI-low,U-high)":    (0,0,0,2),
    "Bùng nổ (GDP-high,D-high,AI-high,U-low)":      (2,2,2,0),
    "Số hóa sơ khai (GDP-mid,D-low,AI-low,U-mid)":  (1,0,0,1),
    "Chín muồi (GDP-high,D-high,AI-mid,U-low)":     (2,2,1,0),
}
for name,s in states.items():
    a=int(np.argmax(Q[s])); q=Q[s]
    print(f"  {name[:45]:45s}→ {anames[a]:15s} Qs={np.round(q,3)}")

# ── CÂU 11.3.4 ─────────────────────────────────────────────────────────────
def eval_pol(pf, n=500):
    rs=[]
    for _ in range(n):
        K,D,AI,H,A=K0,D0,AI0,H0,A0; s=disc(0.08,D,AI,0.30); tot=0
        for _ in range(T_ep):
            a=pf(s); K,D,AI,H,A,s,r=step(K,D,AI,H,A,a); tot+=r
        rs.append(tot)
    return np.mean(rs),np.std(rs)

pols=[("π* Q-learning",lambda s:int(np.argmax(Q[s]))),
      ("Luôn a1 Cân bằng",lambda s:1),
      ("Luôn a3 AI dẫn",lambda s:3),
      ("Random",lambda s:np.random.randint(5))]
print("\nCÂU 11.3.4 – So sánh reward (500 eval episodes):")
pol_res={}
for nm,pf in pols:
    mu,sg=eval_pol(pf); pol_res[nm]=(mu,sg)
    rank="★★★" if mu==max(eval_pol(p)[0] for _,p in pols) else ""
    print(f"  {nm:22s}: {mu:+.4f} ± {sg:.4f} {rank}")

# ── Biểu đồ ─────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle('Bài 11 – Q-Learning: Chính sách kinh tế thích nghi VN\n(20,000 ep | 81 states | 5 actions | γ=0.95)',
             fontweight='bold')

win=500; sm=[np.mean(ep_rew[max(0,i-win):i+1]) for i in range(len(ep_rew))]
axes[0].plot(ep_rew,alpha=0.08,color='steelblue',lw=0.5)
axes[0].plot(sm,color='#1976D2',lw=2.5,label=f'MA-{win}')
axes[0].axhline(np.mean(ep_rew[-1000:]),color='red',ls='--',alpha=0.6,
                label=f'Hội tụ={np.mean(ep_rew[-1000:]):.3f}')
axes[0].set_title('Learning Curve\nReward tích lũy/episode',fontweight='bold')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Total reward'); axes[0].legend(); axes[0].grid(alpha=0.3)

names_p=list(pol_res.keys()); mus=[pol_res[n][0] for n in names_p]; sigs=[pol_res[n][1] for n in names_p]
col_p=['#4CAF50','#90CAF9','#FF9800','#F44336']
bars=axes[1].bar(range(4),mus,color=col_p,alpha=0.9,edgecolor='white',width=0.6)
axes[1].errorbar(range(4),mus,yerr=sigs,fmt='none',color='black',capsize=5,lw=2)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels([n[:12] for n in names_p],rotation=25,fontsize=9)
axes[1].set_title('So sánh chính sách\n(mean±std, 500 episodes)',fontweight='bold')
axes[1].set_ylabel('Total reward'); axes[1].grid(axis='y',alpha=0.3)
for bar,v in zip(bars,mus):
    axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,
                 f'{v:.3f}',ha='center',fontsize=9,fontweight='bold')

# Heatmap chính sách
q_map=np.zeros((3,3),dtype=int)
for s2 in range(3):
    for s3 in range(3):
        q_map[s2,s3]=int(np.argmax(Q[(1,0,s2,s3)]))   # GDP-mid, D-low
cmap=matplotlib.colormaps.get_cmap('Set1').resampled(5)
im=axes[2].imshow(q_map,cmap=cmap,aspect='auto',vmin=0,vmax=4)
axes[2].set_xticks(range(3)); axes[2].set_xticklabels(['U-thấp','U-vừa','U-cao'])
axes[2].set_yticks(range(3)); axes[2].set_yticklabels(['AI-thấp','AI-vừa','AI-cao'])
axes[2].set_title('Bản đồ π*(GDP-mid, D-low)\n(màu = hành động tối ưu)',fontweight='bold')
for s2 in range(3):
    for s3 in range(3):
        a=int(q_map[s2,s3])
        axes[2].text(s3,s2,anames[a][:10],ha='center',va='center',
                     fontsize=8,fontweight='bold',color='white' if a in [3,4] else 'black')
plt.colorbar(im,ax=axes[2],label='Hành động (0-4)')
plt.tight_layout(); plt.savefig('bai11_ket_qua.png',dpi=150,bbox_inches='tight'); plt.close()

s_vn=(1,0,1,1); s_crisis=(0,0,0,2); s_boom=(2,2,2,0)
a_vn=anames[int(np.argmax(Q[s_vn]))]
a_cr=anames[int(np.argmax(Q[s_crisis]))]
a_bm=anames[int(np.argmax(Q[s_boom]))]
print(f"""
THẢO LUẬN:
a) VN 2026 (GDP-mid,D-low,AI-high,U-mid): π*='{a_vn}'
   GDP thấp,D thấp,U cao (khủng hoảng):   π*='{a_cr}'
   → Khi khó khăn: ưu tiên hạ tầng cơ bản + nhân lực để có 'quick win'

b) Bùng nổ (GDP-high,AI-high,U-low): π*='{a_bm}'
   → Khi thuận lợi: tiếp tục khai thác AI hoặc đầu tư bao trùm để duy trì bền vững

c) Tích hợp vào hoạch định chính sách VN:
   ① π* = CÔNG CỤ GỢI Ý; Quốc hội quyết định cuối cùng
   ② Cập nhật Q-table hàng quý theo số liệu GSO thực tế
   ③ Minh bạch hóa: công bố mô hình, kiểm toán độc lập
   ④ KHÔNG tự động hóa quy trình phân bổ ngân sách nhà nước
""")
print("✓ Bài 11 → bai11_ket_qua.png")
