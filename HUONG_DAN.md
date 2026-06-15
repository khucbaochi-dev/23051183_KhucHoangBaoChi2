# HƯỚNG DẪN LÀM BÀI TẬP AIDEOM-VN
## Môn: Mô hình Ra quyết định – Phát triển Kinh tế Việt Nam trong Kỷ nguyên AI

---

## 1. CẤU TRÚC THƯ MỤC

```
aideom_vn/
├── bai01_cobb_douglas.py       # Bài 1 – Hàm sản xuất Cobb-Douglas
├── bai02_lp_budget.py          # Bài 2 – LP phân bổ ngân sách
├── bai03_priority.py           # Bài 3 – Chỉ số ưu tiên ngành
├── bai04_lp_region.py          # Bài 4 – LP phân bổ theo vùng
├── bai05_mip_projects.py       # Bài 5 – MIP lựa chọn dự án
├── bai06_topsis.py             # Bài 6 – TOPSIS xếp hạng vùng AI
├── bai07_pareto.py             # Bài 7 – Pareto đa mục tiêu
├── bai08_dynamic.py            # Bài 8 – Tối ưu động 2026-2035
├── bai09_labor.py              # Bài 9 – Mô phỏng lao động
├── bai10_stochastic.py         # Bài 10 – Stochastic LP 2 giai đoạn
├── bai11_qlearning.py          # Bài 11 – Q-Learning chính sách
├── bai12_aideom_vn.py          # Bài 12 – Đồ án tích hợp AIDEOM-VN
├── vietnam_macro_2020_2025.csv # Dữ liệu vĩ mô (thầy cung cấp)
├── vietnam_sectors_2024.csv    # Dữ liệu 10 ngành (thầy cung cấp)
├── vietnam_regions_2024.csv    # Dữ liệu 6 vùng (thầy cung cấp)
└── bai01_ket_qua.png ... bai12_dashboard.png  (biểu đồ tự sinh)
```

---

## 2. CÀI ĐẶT MÔI TRƯỜNG (làm 1 lần)

### Bước 1 – Cài Python (nếu chưa có)
Tải Python 3.10 hoặc 3.11 tại https://python.org  
Khi cài trên Windows: **đánh dấu "Add Python to PATH"**

### Bước 2 – Tạo môi trường ảo
Mở Terminal (Mac/Linux) hoặc PowerShell (Windows):

```bash
# Di chuyển vào thư mục dự án
cd đường_dẫn_tới/aideom_vn

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Mac/Linux)
source venv/bin/activate
```

### Bước 3 – Cài thư viện
```bash
pip install numpy pandas scipy matplotlib seaborn
```

> **Lưu ý:** Bộ bài tập này được viết **CHỈ dùng numpy, pandas, scipy, matplotlib, seaborn** – không cần pulp/cvxpy/pyomo/pymoo/gymnasium. Tất cả 12 bài đều chạy được ngay.

### Bước 4 – Kiểm tra
```bash
python -c "import numpy, pandas, scipy, matplotlib; print('OK!')"
```

---

## 3. CÁCH CHẠY TỪNG BÀI

### Chạy một bài cụ thể
```bash
cd aideom_vn
python bai01_cobb_douglas.py
```

### Chạy tất cả 12 bài liên tiếp
```bash
# Windows
for %f in (bai01 bai02 bai03 bai04 bai05 bai06 bai07 bai08 bai09 bai10 bai11 bai12) do python %f_*.py

# Mac/Linux
for f in bai01 bai02 bai03 bai04 bai05 bai06 bai07 bai08 bai09 bai10 bai11 bai12; do python ${f}_*.py; done
```

### Dùng Jupyter Notebook (khuyến nghị)
```bash
pip install jupyterlab
jupyter lab
```
Sau đó tạo notebook mới, copy code từng file .py vào từng cell.

### Dùng Google Colab (không cần cài đặt)
1. Vào https://colab.research.google.com
2. Upload 3 file CSV lên (Files → Upload)
3. Copy code từng bài vào cell và chạy

---

## 4. TÓM TẮT TỪNG BÀI

### BÀI 1 – Hàm sản xuất Cobb-Douglas (`bai01_cobb_douglas.py`)
**Mô hình:** `Y = A · K^0.33 · L^0.42 · D^0.10 · AI^0.08 · H^0.07`  
**Kỹ thuật:** numpy, pandas, growth accounting  
**Làm gì:**
- Đọc `vietnam_macro_2020_2025.csv`
- Ước lượng TFP A_t = Y / (K^α · L^β · D^γ · AI^δ · H^θ)
- Tính MAPE dự báo GDP
- Phân rã tăng trưởng (growth accounting)
- Dự báo GDP 2030  

**Kết quả quan trọng:** TFP tăng từ ~34.8 (2020) → ~35.4 (2025), GDP 2030 ≈ 16,600 nghìn tỷ VND

---

### BÀI 2 – LP Phân bổ ngân sách (`bai02_lp_budget.py`)
**Mô hình:** max Z = 0.85x₁ + 1.20x₂ + 0.95x₃ + 1.35x₄, tổng ngân sách 100 nghìn tỷ  
**Kỹ thuật:** `scipy.optimize.linprog` (HiGHS solver)  
**Làm gì:**
- Định nghĩa hàm mục tiêu và 6 ràng buộc
- Tính shadow price bằng perturbation (+1 tỷ mỗi ràng buộc)
- Phân tích độ nhạy ngân sách 70→160 nghìn tỷ
- Kiểm tra kịch bản x₃ ≥ 30  

**Kết quả:** Phân bổ tối ưu tập trung vào R&D (hệ số 1.35 cao nhất)

---

### BÀI 3 – Chỉ số ưu tiên ngành (`bai03_priority.py`)
**Mô hình:** Priority_i = Σ a_j · x̃_j(good) − a_risk · x̃_risk  
**Kỹ thuật:** Chuẩn hóa min-max, numpy  
**Làm gì:**
- Đọc `vietnam_sectors_2024.csv`
- Chuẩn hóa 7 chỉ số (đảo dấu Risk)
- Tính Priority với trọng số mặc định
- Heatmap độ nhạy khi thay đổi trọng số AI (a₆)
- So sánh 2 bộ trọng số: Tăng trưởng vs Bao trùm  

**Kết quả:** CNTT-TT > Logistics > CN Chế biến (top 3 ưu tiên chuyển đổi số)

---

### BÀI 4 – LP Phân bổ theo vùng (`bai04_lp_region.py`)
**Mô hình:** max Z = Σ β_jr · x_jr, 24 biến, 5 nhóm ràng buộc  
**Kỹ thuật:** `scipy.optimize.linprog`, 24 biến  
**Làm gì:**
- Đọc `vietnam_regions_2024.csv` lấy Digital Index
- Phân bổ 50,000 tỷ cho 6 vùng × 4 hạng mục
- So sánh có/không ràng buộc công bằng C5
- Vẽ heatmap phân bổ tối ưu  

**Kết quả:** Tây Nguyên ưu tiên H+I; ĐNB/ĐBSH ưu tiên AI

---

### BÀI 5 – MIP Lựa chọn dự án (`bai05_mip_projects.py`)
**Mô hình:** max Σ B_i · y_i, y_i ∈ {0,1}, 15 dự án, ngân sách 80,000 tỷ  
**Kỹ thuật:** `scipy.optimize.milp` (Mixed Integer LP)  
**Làm gì:**
- Định nghĩa 15 dự án với chi phí, lợi ích NPV
- 7 nhóm ràng buộc: ngân sách, loại trừ, tiên quyết, cân đối
- Phân tích khi nới B=100,000 tỷ
- Tối đa E[Z] có trọng số rủi ro p_i  

**Kết quả:** Chọn 9–10 dự án, ưu tiên 5G, AI quốc gia, bán dẫn, đào tạo kỹ sư

---

### BÀI 6 – TOPSIS Xếp hạng vùng AI (`bai06_topsis.py`)
**Mô hình:** TOPSIS 5 bước, 6 vùng × 8 tiêu chí  
**Kỹ thuật:** numpy thuần (không dùng thư viện ngoài)  
**Làm gì:**
- Đọc `vietnam_regions_2024.csv`
- TOPSIS với trọng số chuyên gia
- So sánh với trọng số Entropy
- Heatmap độ nhạy w_AI (0.10→0.40)  

**Kết quả:** ĐNB > ĐBSH > BTB-DHMT (top 3 ưu tiên đầu tư AI)

---

### BÀI 7 – Pareto đa mục tiêu (`bai07_pareto.py`)
**Mô hình:** max f₁ (GDP), min f₂ (bất bình đẳng), min f₃ (phát thải), min f₄ (rủi ro an ninh)  
**Kỹ thuật:** Weighted-sum scalarization + scipy SLSQP, 200 nghiệm Pareto  
**Làm gì:**
- Quét Pareto front bằng cách random weights + SLSQP
- Vẽ scatter 3D và parallel coordinates
- TOPSIS chọn nghiệm thỏa hiệp (w=[0.40,0.25,0.20,0.15])
- Phân tích chi phí cơ hội  

**Kết quả:** Đánh đổi GDP–bao trùm rõ ràng; nghiệm thỏa hiệp cân bằng 4 mục tiêu

---

### BÀI 8 – Tối ưu động 2026-2035 (`bai08_dynamic.py`)
**Mô hình:** max Σ ρ^t · ln(C_t), T=10, 5 loại vốn tích lũy  
**Kỹ thuật:** `scipy.optimize.minimize` SLSQP, 50 biến  
**Làm gì:**
- Xây phương trình động học K, D, AI, H, A
- Tối ưu phân bổ vốn từng năm
- Vẽ quỹ đạo tối ưu K, D, AI, H, GDP
- So sánh chiến lược: trải đều vs front-load  

**Kết quả:** H nên đi TRƯỚC AI; front-load vốn K trong 3 năm đầu

---

### BÀI 9 – Mô phỏng lao động (`bai09_labor.py`)
**Mô hình:** max Σ NetJob_i = NewJob + UpgradeJob − DisplacedJob  
**Kỹ thuật:** `scipy.optimize.linprog`, 16 biến (x_AI và x_H × 8 ngành)  
**Làm gì:**
- Phân bổ 30,000 tỷ cho AI và đào tạo lại
- Đảm bảo NetJob ≥ 0 mọi ngành
- Tìm ngưỡng x_H tối thiểu CN Chế biến
- Thêm ràng buộc Displaced ≤ 5% L  

**Kết quả:** CN Chế biến và Bán buôn-lẻ cần đầu tư đào tạo lại nhiều nhất

---

### BÀI 10 – Stochastic LP (`bai10_stochastic.py`)
**Mô hình:** max E[Z] = β·x + Σ p_s·β_s·y_s, 4 kịch bản, 20 biến  
**Kỹ thuật:** `scipy.optimize.linprog` deterministic equivalent  
**Làm gì:**
- First-stage: phân bổ x ≤ 65,000 tỷ
- Second-stage: recourse y^s ≤ 15,000 tỷ mỗi kịch bản
- So sánh EV vs SP vs WS
- Tính VSS và EVPI
- Robust optimization (minimax regret)  

**Kết quả:** SP đầu tư H nhiều hơn EV vì nhân lực là "bảo hiểm" chống khủng hoảng

---

### BÀI 11 – Q-Learning (`bai11_qlearning.py`)
**Mô hình:** MDP: 81 trạng thái (3⁴), 5 hành động, T=10 năm/episode  
**Kỹ thuật:** Tabular Q-learning thuần numpy, 20,000 episodes  
**Làm gì:**
- Xây môi trường kinh tế (không dùng gymnasium)
- Training với epsilon-greedy giảm dần
- Trích xuất chính sách π*(s)
- So sánh vs rule-based: a1, a3, random
- Vẽ learning curve và bản đồ chính sách  

**Kết quả:** π* hội tụ trong ~10,000 ep; ở trạng thái VN 2026 → chọn AI dẫn dắt

---

### BÀI 12 – AIDEOM-VN Tích hợp (`bai12_aideom_vn.py`)
**Mô hình:** 6 module liên kết, 5 kịch bản chính sách  
**Kỹ thuật:** Tổng hợp M1-M6 + TOPSIS xếp hạng kịch bản  
**Làm gì:**
- M1: Dự báo GDP 2030 theo 5 kịch bản
- M2: TOPSIS xếp hạng 6 vùng
- M3: LP phân bổ ngân sách mỗi kịch bản
- M4: NetJob ròng mỗi kịch bản
- M5: Rủi ro cyber + phát thải
- M6: Dashboard bảng tổng hợp KPI + xếp hạng TOPSIS  

**Kết quả:** S4 (Bao trùm số) tốt nhất theo TOPSIS đa tiêu chí; S5 (Tối ưu) cao nhất về GDP

---

## 5. CÁC LỖI THƯỜNG GẶP VÀ CÁCH SỬA

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|---------|
| `FileNotFoundError: vietnam_macro...csv` | Chưa chạy đúng thư mục | `cd aideom_vn` rồi mới chạy python |
| `ModuleNotFoundError: No module named 'pulp'` | Không cần cài – code đã dùng scipy | Bỏ qua, code đã được viết lại |
| `AttributeError: 'NoneType'...` | linprog không tìm được nghiệm | Kiểm tra ràng buộc có mâu thuẫn không |
| `numpy.linalg.LinAlgError` | Ma trận suy biến | Thêm `+ 1e-10` vào mẫu số |
| Biểu đồ không hiển thị | Backend matplotlib | Thêm `plt.savefig(...)` thay vì `plt.show()` |
| Bài 11 Q luôn = 0 | Sai index tuple khi update | Dùng `Q[s+(a,)] += ...` đúng cú pháp |

---

## 6. CÁCH VIẾT BÁO CÁO

### Cấu trúc mỗi bài (theo rubric thầy):

**1. Mô hình toán học (30%)**
- Phát biểu bài toán: biến quyết định, hàm mục tiêu, ràng buộc
- Giải thích ý nghĩa từng tham số

**2. Kết quả số (25%)**
- In bảng kết quả rõ ràng
- Kiểm tra sanity: tổng ngân sách, dấu kết quả

**3. Biểu đồ (15%)**
- Mỗi bài ít nhất 2 biểu đồ có nhãn trục + đơn vị
- File .png tự động lưu khi chạy code

**4. Thảo luận chính sách (30%)**
- Liên hệ với Nghị quyết 57-NQ/TW, QĐ 749, QĐ 127, QĐ 411
- Nêu hạn chế mô hình và hướng cải thiện

### Mẫu trích dẫn AI:
> *"Báo cáo này sử dụng Claude (Anthropic) để hỗ trợ viết code Python. Mọi kết quả số, giải thích kinh tế và thảo luận chính sách do nhóm tác giả thực hiện và chịu trách nhiệm."*

---

## 7. CHẠY NHANH TRÊN GOOGLE COLAB

```python
# Cell 1: Upload dữ liệu
from google.colab import files
uploaded = files.upload()  # chọn 3 file CSV

# Cell 2: Cài thư viện (đã có sẵn trên Colab)
# numpy, pandas, scipy, matplotlib đều có sẵn

# Cell 3: Copy code từ baiXX_*.py vào và chạy
```

---

## 8. BẢNG TỔNG HỢP KẾT QUẢ CHÍNH

| Bài | Kỹ thuật | Kết quả nổi bật |
|-----|----------|-----------------|
| 1 | Cobb-Douglas | GDP 2030 ≈ 16,600 nghìn tỷ, TFP tăng ổn định |
| 2 | LP (linprog) | R&D tối đa hóa Z*, shadow price ≈ 1.35 |
| 3 | Priority Index | Top 3: CNTT-TT, Logistics, CN Chế biến |
| 4 | LP 24 biến | ĐNB nhận nhiều nhất, chi phí công bằng <2% |
| 5 | MIP (milp) | Chọn 9 dự án, NPV = 183,200 tỷ |
| 6 | TOPSIS | ĐNB > ĐBSH > BTB-DHMT |
| 7 | Pareto SLSQP | Đánh đổi GDP–bao trùm rõ ràng |
| 8 | Tối ưu động | H trước AI; front-load K hiệu quả hơn |
| 9 | LP lao động | NetJob > 0 mọi ngành khi x_H đủ lớn |
| 10 | Stochastic LP | SP đầu tư H nhiều hơn EV (bảo hiểm rủi ro) |
| 11 | Q-Learning | π* hội tụ; VN 2026 → chọn AI dẫn dắt |
| 12 | AIDEOM-VN | S4 Bao trùm số tốt nhất theo đa tiêu chí |

---

*Tài liệu được tạo tự động. Phiên bản: tháng 6/2026.*
