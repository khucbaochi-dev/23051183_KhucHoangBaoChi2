# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 0 – DÁN VÀO COLAB ĐẦU TIÊN (không cần upload file)  ║
# ║  Chạy cell này 1 lần, các bài sau dùng được luôn           ║
# ╚══════════════════════════════════════════════════════════════╝

import pandas as pd
import io

# ── DỮ LIỆU MACRO 2020-2025 ─────────────────────────────────────
macro_csv = """year,GDP_trillion_VND,GDP_billion_USD,GDP_growth_pct,GDP_per_capita_USD,population_million,agriculture_share_pct,industry_share_pct,services_share_pct,taxes_share_pct,agri_growth_pct,industry_growth_pct,services_growth_pct,inflation_CPI_pct,FDI_disbursed_billion_USD,exports_billion_USD,imports_billion_USD,digital_economy_share_GDP_pct,labor_productivity_million_VND
2020,8044.4,346.6,2.91,3521,97.58,12.66,36.74,41.63,8.97,2.68,3.98,2.34,3.23,19.98,282.6,262.7,12.0,151.2
2021,8487.5,366.1,2.58,3717,98.51,12.36,37.86,40.95,8.83,2.90,4.05,1.22,1.84,19.74,336.3,332.2,12.7,171.3
2022,9513.3,408.8,8.02,4163,99.46,11.88,38.26,41.33,8.53,3.36,7.78,9.99,3.15,22.40,371.3,358.9,14.3,188.1
2023,10221.8,430.0,5.05,4347,100.30,11.96,37.12,42.54,8.38,3.83,3.74,6.82,3.25,23.18,355.5,327.5,16.5,199.3
2024,11511.9,476.3,7.09,4700,101.30,11.86,37.64,42.36,8.14,3.27,8.24,7.38,3.63,25.35,405.5,380.8,18.3,221.9
2025,12847.6,514.0,8.02,5026,102.30,11.64,37.65,42.75,7.96,3.74,8.95,8.50,3.31,27.60,475.0,455.0,19.5,245.0"""

# ── DỮ LIỆU 10 NGÀNH 2024 ───────────────────────────────────────
sectors_csv = """sector_id,sector_name_en,gdp_share_2024_pct,growth_rate_2024_pct,labor_million,labor_share_pct,export_billion_USD,digital_index_0_100,ai_readiness_0_100,fdi_attraction_billion_USD,spillover_coef_0_1,automation_risk_pct,rd_intensity_pct
1,Agriculture-Forestry-Fishery,11.86,3.27,13.2,26.5,40.5,28,15,2.1,0.35,18,0.15
2,Manufacturing,24.1,9.64,11.5,23.1,290.9,68,55,18.6,0.78,42,0.62
3,Construction,7.04,7.45,4.8,9.6,2.5,35,20,0.8,0.42,25,0.18
4,Mining,3.36,-1.2,0.3,0.6,8.2,50,30,0.5,0.3,55,0.22
5,Wholesale-Retail,9.85,7.1,7.8,15.7,5.5,72,48,3.2,0.55,38,0.1
6,Finance-Banking-Insurance,5.12,7.36,0.55,1.1,1.2,82,72,1.8,0.85,52,0.45
7,Logistics-Transport-Warehousing,5.45,9.93,1.95,3.9,3.1,65,42,2.4,0.72,35,0.2
8,Information-Communication-IT,3.85,7.85,0.62,1.2,178,92,88,4.6,0.92,28,1.2
9,Education-Training,3.85,6.42,2.15,4.3,0,55,38,0.4,0.65,22,0.3
10,Healthcare,2.85,6.85,0.75,1.5,0,58,45,1.2,0.6,18,0.55"""

# ── DỮ LIỆU 6 VÙNG 2024 ─────────────────────────────────────────
regions_csv = """region_id,region_name_en,population_million,grdp_trillion_VND,grdp_growth_pct,grdp_per_capita_million_VND,fdi_registered_billion_USD,exports_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,gini_coef,rd_intensity_pct,internet_penetration_pct
1,Northern Midlands and Mountains,14.2,810,8.5,57,3.5,42.5,38,22,21.5,0.405,0.18,72
2,Red River Delta,23.5,3580,7.9,152.3,20,132,78,68,36.8,0.358,0.85,92
3,North Central and South Central Coast,20.8,1820,6.85,87.5,8.2,68.5,55,40,27.5,0.372,0.32,84
4,Central Highlands,6.1,420,7.2,68.9,0.8,2.8,32,18,18.2,0.412,0.15,68
5,Southeast,19.2,3050,7.5,158.9,18.5,128.5,82,75,42.5,0.385,0.78,94
6,Mekong Delta,17.5,1409,7.3,80.5,2.1,25.7,48,30,16.8,0.392,0.22,78"""

# ── NẠP VÀO DATAFRAME ───────────────────────────────────────────
df_macro   = pd.read_csv(io.StringIO(macro_csv))
df_sectors = pd.read_csv(io.StringIO(sectors_csv))
df_regions = pd.read_csv(io.StringIO(regions_csv))

# Lưu ra file CSV để các bài đọc bằng pd.read_csv() bình thường
df_macro.to_csv('vietnam_macro_2020_2025.csv', index=False)
df_sectors.to_csv('vietnam_sectors_2024.csv', index=False)
df_regions.to_csv('vietnam_regions_2024.csv', index=False)

print("✓ Đã tạo 3 file CSV trong Colab!")
print(f"  Macro:   {df_macro.shape}   (6 năm × {df_macro.shape[1]} cột)")
print(f"  Sectors: {df_sectors.shape}  (10 ngành × {df_sectors.shape[1]} cột)")
print(f"  Regions: {df_regions.shape}   (6 vùng × {df_regions.shape[1]} cột)")
print("\nSẵn sàng chạy bài 1-12!")
