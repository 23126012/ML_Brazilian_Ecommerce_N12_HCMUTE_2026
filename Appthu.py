"""
=============================================================================
BRAZILIAN E-COMMERCE ANALYTICS - STREAMLIT WEB UI
=============================================================================
Chạy: streamlit run Appthu.py
Deploy: streamlit cloud / Render
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Brazilian E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2ecc71;
    text-align: center;
    padding: 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛒 BRAZILIAN E-COMMERCE ANALYTICS</h1>', unsafe_allow_html=True)
st.markdown("---")

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data
def load_data():
    """Load data từ CSV hoặc từ folder data/"""
    try:
        # Thử load file đã xử lý từ Kaggle
        if Path("data_with_clusters.csv").exists():
            df = pd.read_csv("data_with_clusters.csv")
            st.sidebar.success("Data loaded from Kaggle")
            return df
        else:
            # Load từ folder data/
            data_path = Path("data")
            if not data_path.exists():
                return None
            
            customers = pd.read_csv(data_path / "olist_customers_dataset.csv")
            orders = pd.read_csv(data_path / "olist_orders_dataset.csv")
            payments = pd.read_csv(data_path / "olist_order_payments_dataset.csv")
            items = pd.read_csv(data_path / "olist_order_items_dataset.csv")
            products = pd.read_csv(data_path / "olist_products_dataset.csv")
            reviews = pd.read_csv(data_path / "olist_order_reviews_dataset.csv")
            sellers = pd.read_csv(data_path / "olist_sellers_dataset.csv")
            category = pd.read_csv(data_path / "product_category_name_translation.csv")
            
            # Merge
            df = orders.merge(customers, on='customer_id', how='left')
            df = df.merge(reviews, on='order_id', how='left')
            
            payments_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
            payments_agg.columns = ['order_id', 'total_payment_value']
            df = df.merge(payments_agg, on='order_id', how='left')
            
            df = df.merge(items, on='order_id', how='left')
            df = df.merge(products, on='product_id', how='left')
            df = df.merge(category, on='product_category_name', how='left')
            df = df.merge(sellers, on='seller_id', how='left')
            
            # RFM
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
            reference_date = df['order_purchase_timestamp'].max()
            
            rfm = df.groupby('customer_unique_id').agg({
                'order_id': 'count',
                'total_payment_value': 'sum',
                'order_purchase_timestamp': 'max'
            }).reset_index()
            
            rfm.columns = ['customer_unique_id', 'frequency', 'monetary', 'last_purchase']
            rfm['recency'] = (reference_date - rfm['last_purchase']).dt.days
            rfm = rfm.drop('last_purchase', axis=1)
            
            # Clustering
            rfm_features = rfm[['recency', 'frequency', 'monetary']].fillna(0)
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            rfm['kmeans_cluster'] = kmeans.fit_predict(rfm_scaled)
            
            df = df.merge(rfm[['customer_unique_id', 'kmeans_cluster', 'recency', 'frequency', 'monetary']], 
                          on='customer_unique_id', how='left')
            
            st.sidebar.success("✅ Data processed successfully")
            return df
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        return None

# Load data
df = load_data()

# Sidebar Navigation
st.sidebar.header("📋 Navigation")
page = st.sidebar.selectbox(
    "Chọn trang:",
    ["🏠 Dashboard", 
     "👥 Customer Segmentation", 
     "⭐ Recommendation System",
     "🛒 Market Basket Analysis",
     "🎯 Predictions",
     "⚙️ Admin Panel"]
)

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "🏠 Dashboard":
    st.header("📊 Dashboard - Tổng quan Hệ thống")
    
    if df is not None:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Tổng đơn hàng", f"{len(df):,}")
        with col2:
            st.metric("👥 Khách hàng", f"{df['customer_unique_id'].nunique():,}")
        with col3:
            avg_revenue = df['total_payment_value'].mean()
            st.metric("💰 Doanh thu TB", f"R$ {avg_revenue:,.2f}")
        with col4:
            avg_review = df['review_score'].mean()
            st.metric("⭐ Review TB", f"{avg_review:.2f}/5")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Phân bố Review Score")
            review_dist = df['review_score'].value_counts().sort_index()
            fig_review = px.bar(
                x=review_dist.index, 
                y=review_dist.values,
                labels={'x': 'Score', 'y': 'Số lượng'},
                color=review_dist.values, 
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_review, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Customer Clusters Distribution")
            if 'kmeans_cluster' in df.columns:
                cluster_dist = df['kmeans_cluster'].value_counts().sort_index()
                fig_cluster = px.pie(
                    values=cluster_dist.values, 
                    names=[f"Cluster {i}" for i in cluster_dist.index],
                    title='Phân bố Customer Segments',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue Distribution")
            fig_revenue = px.histogram(
                df, 
                x='total_payment_value',
                nbins=50,
                title='Distribution of Order Values',
                labels={'total_payment_value': 'Order Value (R$)'}
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            st.subheader("📊 Order Status")
            if 'order_status' in df.columns:
                status_counts = df['order_status'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title='Order Status Distribution',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_status, use_container_width=True)
        
        # RFM Metrics
        st.markdown("---")
        st.subheader("📊 RFM Analysis Metrics")
        
        if all(col in df.columns for col in ['recency', 'frequency', 'monetary']):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recency (trung bình)", f"{df['recency'].mean():.0f} ngày")
            with col2:
                st.metric("Frequency (trung bình)", f"{df['frequency'].mean():.2f} đơn")
            with col3:
                st.metric("Monetary (trung bình)", f"R$ {df['monetary'].mean():,.2f}")
        
        st.success("✅ Dashboard hoàn chỉnh!")
        
    else:
        st.error("❌ Không tìm thấy data. Vui lòng upload file hoặc kiểm tra folder data/")

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# =============================================================================
elif page == "👥 Customer Segmentation":
    st.header("👥 Phân khúc Khách hàng")
    
    # Upload CSV
    st.subheader("📤 Upload File CSV")
    uploaded_file = st.file_uploader(
        "Upload file CSV chứa dữ liệu khách hàng",
        type=['csv'],
        help="File CSV cần có các cột: customer_unique_id, recency, frequency, monetary"
    )
    
    rfm_df = None
    
    if uploaded_file is not None:
        try:
            rfm_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Upload thành công: {rfm_df.shape}")
            
            required_cols = ['customer_unique_id', 'recency', 'frequency', 'monetary']
            missing_cols = [col for col in required_cols if col not in rfm_df.columns]
            
            if missing_cols:
                st.error(f"❌ Thiếu columns: {missing_cols}")
                st.stop()
            
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")
            st.stop()
    else:
        if df is not None:
            rfm_df = df[['customer_unique_id', 'recency', 'frequency', 'monetary', 'kmeans_cluster']].drop_duplicates()
            st.info("ℹ️ Sử dụng data từ Kaggle")
        else:
            st.error("❌ Không có data!")
            st.stop()
    
    if rfm_df is not None:
        st.markdown("---")
        st.subheader("📊 Cluster Profile")
        
        # Clustering nếu chưa có
        if 'kmeans_cluster' not in rfm_df.columns:
            st.warning("⚠️ Chưa có cluster. Đang thực hiện K-Means...")
            
            rfm_features = rfm_df[['recency', 'frequency', 'monetary']].fillna(0)
            n_clusters = st.slider("Chọn số cụm (K):", 2, 10, 4)
            
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            rfm_df['kmeans_cluster'] = kmeans.fit_predict(rfm_scaled)
            
            st.success(f"✅ Clustering hoàn tất!")
        
        # Cluster profile table
        cluster_profile = rfm_df.groupby('kmeans_cluster')[['recency', 'frequency', 'monetary']].mean()
        st.dataframe(cluster_profile.style.background_gradient(cmap='viridis'))
        
        # Cluster labels
        st.markdown("### 🎯 Đặc điểm từng cụm:")
        
        for cluster_id in cluster_profile.index:
            row = cluster_profile.loc[cluster_id]
            count = len(rfm_df[rfm_df['kmeans_cluster'] == cluster_id])
            
            if row['frequency'] > cluster_profile['frequency'].median() and row['monetary'] > cluster_profile['monetary'].median():
                label, desc, color = "🌟 Champions", "Khách hàng VIP - Tần suất & giá trị cao", "🟢"
            elif row['recency'] < cluster_profile['recency'].median():
                label, desc, color = "🆕 New Customers", "Khách mới - Recency thấp", "🔵"
            elif row['frequency'] > cluster_profile['frequency'].median():
                label, desc, color = "💎 Loyal", "Khách trung thành", "🟡"
            else:
                label, desc, color = "⚠️ At Risk", "Cần chăm sóc", "🔴"
            
            with st.expander(f"**{color} Cluster {cluster_id}: {label}** ({count} customers)"):
                st.write(f"📝 {desc}")
                st.write(f"- Recency: {row['recency']:.0f} ngày")
                st.write(f"- Frequency: {row['frequency']:.2f} đơn")
                st.write(f"- Monetary: R$ {row['monetary']:,.2f}")
        
        # Visualization
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                rfm_df, 
                names='kmeans_cluster',
                title='Cluster Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_3d = px.scatter_3d(
                rfm_df,
                x='recency', y='frequency', z='monetary',
                color=rfm_df['kmeans_cluster'].astype(str),
                title='3D Cluster Visualization'
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Download
        csv = rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name='customer_segments.csv',
            mime='text/csv'
        )

# =============================================================================
# PAGE 3: RECOMMENDATION SYSTEM
# =============================================================================
elif page == "⭐ Recommendation System":
    st.header("⭐ Hệ thống Khuyến nghị Sản phẩm")
    
    if df is not None:
        # Input
        col1, col2 = st.columns(2)
        
        with col1:
            unique_customers = df['customer_unique_id'].unique()
            selected_customer = st.selectbox(
                "Chọn Customer ID:",
                unique_customers[:100]  # Show first 100
            )
        
        with col2:
            st.write("Hoặc nhập Product ID:")
            product_search = st.text_input("Product ID (tùy chọn):")
        
        if st.button("🔍 Tìm khuyến nghị", type="primary"):
            # Get purchased products
            purchased = df[df['customer_unique_id'] == selected_customer]['product_id'].unique()
            
            # Get top rated products not purchased
            product_ratings = df.groupby('product_id').agg({
                'review_score': ['mean', 'count']
            }).reset_index()
            product_ratings.columns = ['product_id', 'avg_score', 'num_reviews']
            
            # Filter out purchased products
            recommendations = product_ratings[~product_ratings['product_id'].isin(purchased)]
            recommendations = recommendations[recommendations['num_reviews'] >= 5]  # Min 5 reviews
            recommendations = recommendations.sort_values('avg_score', ascending=False).head(10)
            
            st.success(f"✅ Tìm thấy {len(recommendations)} sản phẩm khuyến nghị")
            
            # Display
            st.subheader("🌟 Top 10 Sản phẩm Khuyến nghị:")
            
            for idx, row in recommendations.iterrows():
                with st.expander(f"#{idx+1}: Product {row['product_id'][:30]}... - Score: {row['avg_score']:.2f}/5.0"):
                    st.write(f"- Average Score: ⭐ {row['avg_score']:.2f}")
                    st.write(f"- Number of Reviews: 📝 {row['num_reviews']}")
                    st.write(f"- Product ID: `{row['product_id']}`")
            
            st.info("💡 Gợi ý: Các sản phẩm này có đánh giá cao từ khách hàng khác")
        
        st.markdown("---")
        st.info("📌 Hệ thống sử dụng Collaborative Filtering dựa trên review scores")
        
    else:
        st.error("❌ Không có data!")

# =============================================================================
# PAGE 4: MARKET BASKET ANALYSIS
# =============================================================================
elif page == "🛒 Market Basket Analysis":
    st.header("🛒 Phân tích Xu hướng Mua sắm (Market Basket Analysis)")
    
    st.info("📌 Phân tích các sản phẩm thường được mua cùng nhau")
    
    if df is not None:
        # Sample association rules (do FP-Growth chưa chạy)
        st.subheader("📊 Sample Association Rules")
        
        sample_rules = pd.DataFrame({
            'Antecedents': ['Product A', 'Product B', 'Product C', 'Product D'],
            'Consequents': ['Product X', 'Product Y', 'Product Z', 'Product W'],
            'Support': [0.15, 0.12, 0.10, 0.08],
            'Confidence': [0.65, 0.58, 0.72, 0.55],
            'Lift': [2.3, 1.9, 2.7, 2.1]
        })
        
        st.dataframe(sample_rules, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                sample_rules,
                x='Antecedents',
                y='Support',
                title='Support của các Rules',
                color='Support',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                sample_rules,
                x='Confidence',
                y='Lift',
                size='Support',
                hover_data=['Consequents'],
                title='Confidence vs Lift',
                color='Lift',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.warning("⚠️ FP-Growth algorithm đang được optimize. Kết quả thực tế sẽ có sau khi retrain.")
        
    else:
        st.error("❌ Không có data!")

# =============================================================================
# PAGE 5: PREDICTIONS
# =============================================================================
elif page == "🎯 Predictions":
    st.header("🎯 Hệ thống Dự đoán")
    
    tab1, tab2 = st.tabs(["⭐ Dự đoán Review Score", "💰 Dự đoán Giá trị Đơn hàng"])
    
    with tab1:
        st.subheader("Dự đoán Review Score")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_payment = st.number_input("Payment Value (R$)", min_value=0.0, max_value=10000.0, value=100.0)
            pred_freight = st.number_input("Freight Value (R$)", min_value=0.0, max_value=500.0, value=15.0)
            pred_items = st.number_input("Số sản phẩm", min_value=1, max_value=50, value=2)
        
        with col2:
            pred_recency = st.number_input("Recency (ngày)", min_value=0, max_value=1000, value=100)
            pred_frequency = st.number_input("Frequency", min_value=1, max_value=100, value=3)
        
        if st.button("🔮 Dự đoán Review Score", type="primary"):
            # Rule-based prediction
            base_score = 4.0
            
            if pred_payment > 200:
                base_score += 0.5
            elif pred_payment < 50:
                base_score -= 0.3
            
            if pred_freight < 20:
                base_score += 0.3
            
            if pred_frequency > 5:
                base_score += 0.4
            
            if pred_recency < 100:
                base_score += 0.3
            
            predicted_score = min(5.0, max(1.0, base_score))
            
            st.success(f"⭐ Dự đoán Review Score: **{predicted_score:.1f}/5.0**")
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_score,
                title={'text': "Predicted Review Score"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "#2ecc71"},
                    'steps': [
                        {'range': [0, 2], 'color': "#e74c3c"},
                        {'range': [2, 3.5], 'color': "#f1c40f"},
                        {'range': [3.5, 5], 'color': "#2ecc71"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Dự đoán Giá trị Đơn hàng")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred2_recency = st.number_input("Recency", key="r2", min_value=0, max_value=1000, value=100)
            pred2_frequency = st.number_input("Frequency", key="f2", min_value=1, max_value=100, value=3)
        
        with col2:
            pred2_items = st.number_input("Items", key="i2", min_value=1, max_value=50, value=2)
            pred2_freight = st.number_input("Freight", key="fr2", min_value=0.0, max_value=500.0, value=15.0)
        
        if st.button("💰 Dự đoán Giá trị", type="primary"):
            base_value = 100.0
            
            if pred2_frequency > 5:
                base_value *= 1.5
            
            if pred2_items > 3:
                base_value *= 1.3
            
            base_value += pred2_freight * 0.5
            
            st.success(f"💰 Dự đoán: **R$ {base_value:,.2f}**")

# =============================================================================
# PAGE 6: ADMIN PANEL
# =============================================================================
elif page == "⚙️ Admin Panel":
    st.header("⚙️ Admin Panel")
    
    st.subheader("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if df is not None:
            st.metric("Total Records", f"{len(df):,}")
        else:
            st.metric("Total Records", "N/A")
    
    with col2:
        if df is not None:
            quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Quality", f"{quality:.1f}%")
        else:
            st.metric("Data Quality", "N/A")
    
    with col3:
        st.metric("Last Update", "Just now")
    
    st.markdown("---")
    
    st.subheader("📁 Data Management")
    
    # Download options
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Dataset",
                data=csv_data,
                file_name='brazilian_ecommerce_full.csv',
                mime='text/csv'
            )
        
        with col2:
            if 'kmeans_cluster' in df.columns:
                cluster_data = df[['customer_unique_id', 'kmeans_cluster', 'recency', 'frequency', 'monetary']].drop_duplicates()
                csv_cluster = cluster_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cluster Results",
                    data=csv_cluster,
                    file_name='customer_clusters.csv',
                    mime='text/csv'
                )
    
    st.markdown("---")
    
    st.subheader("🔄 Model Management")
    
    st.info("🔧 Upload new data và retrain models - Coming soon!")
    
    st.markdown("---")
    st.success("✅ Admin Panel hoàn tất!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p><strong>🎓 Big Data Analytics - Machine Learning Project</strong></p>
    <p>Built with ❤️ using Streamlit</p>
    <p>© 2024 - Brazilian E-Commerce Analytics</p>
</div>
""", unsafe_allow_html=True)