"""
Script huấn luyện Model SVD (dùng implicit - Matrix Factorization)
run: python train_svd_model.py
"""
import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix
import joblib
from pathlib import Path

print("🚀 Bắt đầu huấn luyện SVD Model (dùng implicit)...")

# Kiểm tra file tồn tại
reviews_path = 'olist_order_reviews_dataset.csv'
items_path = 'olist_order_items_dataset.csv'
orders_path = 'olist_orders_dataset.csv'

if not all(Path(f).exists() for f in [reviews_path, items_path, orders_path]):
    print("❌ Không tìm thấy file dữ liệu!")
    exit()

print("📂 Đang load dữ liệu...")
reviews = pd.read_csv(reviews_path)
items = pd.read_csv(items_path)
orders = pd.read_csv(orders_path)

# Merge: reviews → orders → items
print("🔄 Đang merge dữ liệu...")
df = reviews.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')
df = df.merge(items[['order_id', 'product_id']], on='order_id', how='left')
df = df[['customer_id', 'product_id', 'review_score']].dropna()

print(f"✅ Dataset: {len(df)} ratings")
print(f"   - Customers: {df['customer_id'].nunique():,}")
print(f"   - Products: {df['product_id'].nunique():,}")

# Tạo mapping IDs
customer_ids = df['customer_id'].unique()
product_ids = df['product_id'].unique()

customer_to_idx = {cid: idx for idx, cid in enumerate(customer_ids)}
product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}

# Tạo sparse matrix
print("🔄 Đang tạo sparse matrix...")
df['customer_idx'] = df['customer_id'].map(customer_to_idx)
df['product_idx'] = df['product_id'].map(product_to_idx)

sparse_matrix = csr_matrix((df['review_score'], 
                            (df['customer_idx'], df['product_idx'])))

# Train Matrix Factorization
print("🧠 Đang huấn luyện Matrix Factorization (ALS)...")
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    iterations=15,
    regularization=0.01,
    random_state=42
)

model.fit(sparse_matrix)

print("✅ Model trained!")

# Lưu model
print("💾 Đang lưu model...")
joblib.dump(model, 'svd_model.pkl')
joblib.dump(customer_to_idx, 'customer_to_idx.pkl')
joblib.dump(product_to_idx, 'product_to_idx.pkl')
joblib.dump(product_ids, 'all_products.pkl')

print("✅ HOÀN TẤT!")
print("   Model saved: svd_model.pkl")
print("   Bây giờ chạy: streamlit run Appfinal.py")
