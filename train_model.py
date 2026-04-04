"""
Script huấn luyện Model dự đoán Review Score
run: python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Bắt đầu huấn luyện Model...")

# Load Data
orders = pd.read_csv('olist_orders_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')

print("✅ Đã load dữ liệu.")

# Merge Data
pay_agg = payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_type': 'first'
}).reset_index()

rev_agg = reviews[['order_id', 'review_score']].drop_duplicates()

df = orders.merge(items, on='order_id', how='left')
df = df.merge(pay_agg, on='order_id', how='left')
df = df.merge(rev_agg, on='order_id', how='left')
df = df.merge(products[['product_id', 'product_category_name', 'product_weight_g', 
                         'product_length_cm', 'product_height_cm', 'product_width_cm']], 
              on='product_id', how='left')
df = df.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
df = df.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left', 
              suffixes=('_seller', '_customer'))

# Feature Engineering
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_month'] = df['order_purchase_timestamp'].dt.month
df['order_hour'] = df['order_purchase_timestamp'].dt.hour
df['order_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['total_order_value'] = df['price'] + df['freight_value']

# Fill NaN Categorical
df['payment_type'] = df['payment_type'].fillna('credit_card')
df['seller_state'] = df['seller_state'].fillna('SP')
df['product_category_name'] = df['product_category_name'].fillna('unknown')
df['order_status'] = df['order_status'].fillna('delivered')

# Drop rows không có review_score (Target)
df_model = df.dropna(subset=['review_score']).copy()

# 🔧 SỬA LỖI QUAN TRỌNG: Điền NaN cho biến số (Numeric)
# Random Forest không chấp nhận NaN, phải fill bằng median
numeric_cols = ['price', 'freight_value', 'product_weight_g', 
                'product_length_cm', 'product_height_cm', 'product_width_cm', 'total_order_value']
for col in numeric_cols:
    df_model[col] = df_model[col].fillna(df_model[col].median())

features = ['price', 'freight_value', 'product_weight_g', 'product_length_cm', 
            'product_height_cm', 'product_width_cm', 'payment_type', 'customer_state', 
            'seller_state', 'product_category_name', 'order_status', 'order_month', 
            'order_hour', 'order_dayofweek', 'total_order_value']

target = 'review_score'

# Encode Categorical Variables - LƯU LẠI CÁC ENCoders
le_dict = {}
categorical_cols = ['payment_type', 'customer_state', 'seller_state', 
                    'product_category_name', 'order_status']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = df_model[col].astype(str)
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le
    print(f"✅ Encoded {col}: {len(le.classes_)} categories")

X = df_model[features]
y = df_model[target]

# Train Model
print("🧠 Đang huấn luyện Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"✅ Model R² Score: {score:.4f}")

# LƯU MODEL VÀ ENCoders
joblib.dump(model, 'full_pipeline.pkl')
joblib.dump(le_dict, 'label_encoders.pkl') 

print("💾 Đã lưu: full_pipeline.pkl và label_encoders.pkl")
print("✅ HOÀN TẤT! Chạy: streamlit run Appfinal.py")
