# 🛒 Brazilian E-Commerce Analytics

## 📋 Thông tin nhóm
- **Môn học:** Machine Learning / Python for Data Analytics
- **GVHD:** Hồ Nhựt Minh
- **Trường:** Đại học Công nghệ Kỹ thuật TP.HCM (HCMUTE)
- **Nhóm:** N12
- **Thành viên:**
  1. Dương Đặng Hoài An - 23126002
  2. Trần Ngọc Lan Anh - 23126005
  3. Nguyễn Thị Ngọc Hà - 23126012
  4. Huỳnh Thị Trà My - 23126027

## Cách chạy ứng dụng

### Bước 1: Cài đặt Python và thư viện
```bash
# Yêu cầu: Python 3.8 trở lên
pip install -r requirements.txt

1. Download file data_with_clusters.csv từ Google Drive:
   https://drive.google.com/drive/folders/1rPho9bIkuaEGDI37GloLLSmuFz2roACo?usp=drive_link
2. Đặt file vào folder project (cùng cấp với Appthu.py)
3. Chạy: streamlit run Appthu.py

ML_Brazilian_Ecommerce_N12_HCMUTE_2026/
├── Appthu.py              # Code chính Streamlit
├── requirements.txt       # Thư viện yêu cầu
├── README.md              # File hướng dẫn
├── .gitignore             # Loại bỏ file không cần
├── data/                  # 9 file CSV gốc (download:https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
├── master_clean.csv       # Data đã xử lý (không upload - download từ Drive)
└── master_dashboard.csv
├── master_dataframe.csv
├── classification_results.csv
├── data_summary.csv
├── regression_results.csv
├── rfm_analysis.csv
├── rfm_statistics.csv
├── rfm_with_clusters.csv
├── X_transformed.npy
└── y_target.joblib
