# Olist Analytics - Big Data & ML Project

> **Big Data Analytics & Machine Learning Project**  
> **Nhóm 12 - BIGDATA2026 - HCMUTE**


## 👥 THÔNG TIN NHÓM
NHÓM 12
1. Dương Đặng Hoài An - 23126002 
2. Trần Ngọc Lan Anh - 23126005
3. Nguyễn Thị Ngọc Hà - 23126012
4. Huỳnh Thị Trà My - 23126027  

- **Môn học:** Machine Learning / Python for Data Analytics
- **Giảng viên hướng dẫn:** Hồ Nhựt Minh
- **Trường:** Đại học Công nghệ Kỹ thuật TP.HCM (HCMUTE)
- **Học kỳ:** 2025-2026

---

## 📖 MÔ TẢ DỰ ÁN

Ứng dụng **Streamlit Dashboard** phân tích toàn diện dữ liệu thương mại điện tử Brazil (Olist Dataset) với 6 module chức năng:

| Module | Chức năng | Công nghệ |
|:---|:---|:---|
| 📊 **Dashboard** | Tổng quan KPIs, biểu đồ EDA | Plotly, Pandas |
| 👥 **Phân khúc KH** | RFM Analysis + K-Means Clustering | Scikit-learn |
| ⭐ **Khuyến nghị SP** | Product Recommendation (SVD/Matrix Factorization) | Implicit, Surprise |
| 🛒 **Xu hướng** | Market Basket Analysis (FP-Growth) | MLxtend |
| 🎯 **Dự đoán** | Review Score Prediction | Random Forest Regressor |
| ⚙️ **Admin** | Model Management, Data Upload | Joblib, Streamlit |

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG

### ✅ Yêu cầu hệ thống
- Python **3.8** trở lên
- pip (Python package manager)
- RAM tối thiểu: 8GB (khuyến nghị 16GB để train model)

### 📋 Quy trình cài đặt nhanh (3 Bước)

> **Lưu ý quan trọng:** Dữ liệu và model đã được xử lý sẵn trên Google Colab. Bạn **không cần** chạy lại script train hay xử lý data thô. Chỉ cần tải file về, đặt đúng chỗ và chạy.

```bash
# ==============================================================================
# BƯỚC 1: Clone repository từ GitHub về máy
# ==============================================================================
git clone https://github.com/23126012/ML_Brazilian_Ecommerce_N12_HCMUTE_2026.git
cd CK BIGDATA final

# ==============================================================================
# BƯỚC 2: Tải dữ liệu & model đã xử lý sẵn (Pre-processed & Pre-trained)
# ==============================================================================
# 🔗 Link Google Drive: https://drive.google.com/drive/folders/1pceMuoiJjko6TTZmL2BQYn9LkcBM0Jsx
#
# 📥 Hướng dẫn:
# 1. Download toàn bộ file .csv và .pkl từ link trên.
# 2. GIẢI NÉN (nếu có) và COPY TOÀN BỘ file vào THƯ MỤC GỐC của project.
#    ⚠️ QUAN TRỌNG: Các file phải nằm CÙNG CẤP với file Appfinal.py.
#    Sau bước này, bạn sẽ có:
#    - 9 file dataset gốc (olist_*.csv)
#    - 8 file data đã xử lý (rfm_*.csv, tfidf_vectorizer.pkl, etc.)
#    - Chưa có: Các file model (.pkl) → Sẽ tạo ở Bước 4

# ==============================================================================
# BƯỚC 3: Cài đặt dependencies & Khởi chạy ứng dụng
# ==============================================================================
pip install -r requirements.txt
streamlit run Appfinal.py

# > Ứng dụng sẽ tự động mở tại: http://localhost:8501
# > Dữ liệu & model sẽ được load tự động (< 5 giây). Sẵn sàng demo!
```
---
📁 CẤU TRÚC DỰ ÁN
```
CK BIGDATA FINAL/
│
├── 📄 Appfinal.py                               ← Streamlit App chính (6 trang)
├── 📄 train_model.py                            ← Script train Random Forest
├── 📄 train_svd_model.py                        ← Script train SVD Recommendation
├──  final.ipynb                               ← Google Colab Notebook
├── 📄 README.md                                 ← File hướng dẫn này
├── 📄 requirements.txt                          ← Python dependencies
├── 📄 .gitignore                                ← Git ignore rules
│
├──  DATA TẢI VỀ TỪ GOOGLE DRIVE (Bắt buộc)
│   │   🔗 Link: https://drive.google.com/drive/folders/1pceMuoiJjko6TTZmL2BQYn9LkcBM0Jsx
│   │
│   ├── 📦 Dataset gốc (Olist E-commerce)
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_order_payments_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   ├── olist_products_dataset.csv
│   │   ├── olist_sellers_dataset.csv
│   │   ├── olist_geolocation_dataset.csv
│   │   └── product_category_name_translation.csv
│   │
│   └── 📦 Dữ liệu đã xử lý & Output từ Colab
│       ├── rfm_customer_ids.csv
│       ├── rfm_results.csv
│       ├── rfm_scaled.npy
│       ├── rfm_scored_final.csv
│       ├── rfm_scaler.pkl
│       ├── rfm_distribution.html
│       ├── top_10_association_rules.csv         ← FP-Growth output
│       └── tfidf_vectorizer.pkl
│
└──  MODEL & MAPPING FILES (Tự sinh sau khi chạy train_*.py)
    ├── full_pipeline.pkl                        ← Random Forest pipeline
    ├── label_encoders.pkl                       ← Label encoders
    ├── svd_model.pkl                            ← SVD recommendation model
    ├── customer_to_idx.pkl                      ← Customer ID mapping
    ├── product_to_idx.pkl                       ← Product ID mapping
    ├── all_products.pkl                         ← All products list
    └── final_model_regression.pkl               ← Regression model backup
```    
⚠️ Lưu ý bắt buộc: Để code chạy đúng, tất cả file dữ liệu và model phải nằm trực tiếp trong thư mục gốc (cùng cấp với Appfinal.py). Code đang sử dụng đường dẫn tương đối (pd.read_csv('file.csv')) nên sẽ báo lỗi nếu bạn đặt file vào thư mục con

---
## 🛠️ CÔNG NGHỆ SỬ DỤNG

| Category | Packages & Tools |
|:---|:---|
| **Frontend & App** | Streamlit, Plotly Express |
| **Data Processing** | Pandas, NumPy (Pre-processing trên Google Colab) |
| **Machine Learning** | Scikit-learn, Implicit (Matrix Factorization) |
| **Association Rules** | MLxtend (FP-Growth Algorithm) |
| **Model Persistence** | Joblib (Serialization) |
| **Development Env** | VS Code, Git/GitHub, Google Colab |

---
🙏 LỜI CẢM ƠN
Cảm ơn giảng viên và các bạn đã hỗ trợ trong quá trình thực hiện project!
© 2026 - NHÓM 12 - BIGDATA2026
Built with ❤️ using Streamlit
