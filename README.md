# 🛒 PHÂN TÍCH THƯƠNG MẠI ĐIỆN TỬ BRAZIL

**Big Data Analytics - Machine Learning Project**  
**Nhóm 12 - HUTECH BIGDATA2026**
---
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

## 📖 MÔ TẢ DỰ ÁN

Ứng dụng Streamlit phân tích dữ liệu thương mại điện tử Brazil (Olist Dataset) với các chức năng:
- 📊 Dashboard tổng quan
- 👥 Phân khúc khách hàng (RFM + Clustering)
- ⭐ Hệ thống khuyến nghị sản phẩm
- 🛒 Phân tích giỏ hàng (Market Basket Analysis)
- 🎯 Dự đoán review score bằng Machine Learning

---

## 🚀 CÀI ĐẶT & CHẠY ỨNG DỤNG

### **Yêu cầu hệ thống**
- Python 3.8 trở lên
- pip (Python package manager)

### **Bước 1: Cài đặt các packages cần thiết**

```bash
pip install -r requirements.txt / cài thủ công: pip install streamlit pandas numpy plotly scikit-learn joblib

## **Bước 2: chạy ứng dụng: streamlit run N12_App.py
HOẶC: python -m streamlit run N12_App.py
---

```
## Link lưu data: https://drive.google.com/drive/folders/1pceMuoiJjko6TTZmL2BQYn9LkcBM0Jsx
## Cấu trúc thư mục:

N12_CKBIGDATA/
│
├── 📄 N12_App.py                      # File chính Streamlit App
├── 📄 requirements.txt                # Danh sách packages Python
├── 📄 .gitignore                      # Git ignore file
│
├── 📊 DATA FILES (CSV)
│   ├── olist_orders_dataset.csv       # 99,441 đơn hàng
│   ├── olist_customers_dataset.csv    # 99,441 khách hàng
│   ├── olist_order_items_dataset.csv  # 112,650 sản phẩm
│   ├── olist_order_payments_dataset.csv  # 103,886 thanh toán
│   ├── olist_order_reviews_dataset.csv   # 99,224 đánh giá
│   ├── olist_products_dataset.csv     # 32,951 sản phẩm
│   ├── olist_sellers_dataset.csv      # 3,095 người bán
│   ├── olist_geolocation_dataset.csv  # 1,000,163 vị trí
│   ├── product_category_name_translation.csv  # 71 danh mục
│   ├── rfm_analysis.csv               # RFM metrics (96,096 customers)
│   ├── rfm_with_clusters.csv          # RFM + Clusters (KMeans + GMM)
│   ├── association_rules.csv          # FP-Growth rules (14 columns)
│   └── file_structure_summary.csv     # Summary của tất cả files
│
└── 🤖 ML MODELS (PKL)
    ├── full_pipeline.pkl              # Pipeline dự đoán review score
    ├── best_review_classifier.pkl     # Classifier review (backup)
    ├── best_payment_regressor.pkl     # Regressor payment value
    └── tfidf_vectorizer.pkl           # TF-IDF Vectorizer
---
Nguồn dữ liệu gốc:
Dataset: Brazilian E-Commerce Public Dataset by Olist
Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Thời gian: 2016-2018
Vị trí: Brazil
---
🙏 LỜI CẢM ƠN
Cảm ơn giảng viên và các bạn đã hỗ trợ trong quá trình thực hiện project!
© 2026 - NHÓM 12 - BIGDATA2026
Built with ❤️ using Streamlit
