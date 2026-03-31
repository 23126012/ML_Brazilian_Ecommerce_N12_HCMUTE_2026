"""
BRAZILIAN E-COMMERCE ANALYTICS - STREAMLIT WEB UI
Chạy: streamlit run Appthu.py
Deploy: streamlit cloud / Render
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
    """Load data từ các file đã xử lý"""
    try:
        # Ưu tiên 1: rfm_final.csv
        if Path("rfm_final.csv").exists():
            df = pd.read_csv("rfm_final.csv")
            st.sidebar.success("✅ Data loaded: rfm_final.csv")
            return df
        
        # Ưu tiên 2: data_with_clusters.csv
        elif Path("data_with_clusters.csv").exists():
            df = pd.read_csv("data_with_clusters.csv")
            st.sidebar.success("✅ Data loaded: data_with_clusters.csv")
            return df
        
        # Ưu tiên 3: Load từ folder data/ (9 files)
        else:
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
            
            df = orders.merge(customers, on='customer_id', how='left')
            df = df.merge(reviews, on='order_id', how='left')
            
            payments_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
            payments_agg.columns = ['order_id', 'total_payment_value']
            df = df.merge(payments_agg, on='order_id', how='left')
            
            df = df.merge(items, on='order_id', how='left')
            df = df.merge(products, on='product_id', how='left')
            df = df.merge(category, on='product_category_name', how='left')
            df = df.merge(sellers, on='seller_id', how='left')
            
            st.sidebar.success("✅ Data processed from 9 CSV files")
            return df
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        return None

# Load data
df = load_data()

# =============================================================================
# LOAD ML MODELS
# =============================================================================
@st.cache_resource
def load_models():
    """Load các model đã train từ file .pkl"""
    models = {}
    
    if Path("svd_model.pkl").exists():
        try:
            models['svd'] = joblib.load("svd_model.pkl")
            st.sidebar.success("✅ SVD Model loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Không thể load SVD Model: {e}")
    
    if Path("classification_model.pkl").exists():
        try:
            models['classification'] = joblib.load("classification_model.pkl")
            st.sidebar.success("✅ Classification Model loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Không thể load Classification Model: {e}")
    
    if Path("regression_model.pkl").exists():
        try:
            models['regression'] = joblib.load("regression_model.pkl")
            st.sidebar.success("✅ Regression Model loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Không thể load Regression Model: {e}")
    
    return models

models = load_models()

# =============================================================================
# LOAD ASSOCIATION RULES
# =============================================================================
@st.cache_data
def load_association_rules():
    """Load association rules từ file CSV"""
    try:
        if Path("association_rules.csv").exists():
            rules = pd.read_csv("association_rules.csv")
            if len(rules) > 0:
                st.sidebar.success(f"✅ Association Rules loaded ({len(rules)} rules)")
                return rules
            else:
                st.sidebar.warning("⚠️ File association_rules.csv rỗng")
                return None
        else:
            return None
    except Exception as e:
        st.sidebar.warning(f"⚠️ Không thể load Association Rules: {e}")
        return None

association_rules = load_association_rules()

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.header("📋 Navigation")
page = st.sidebar.selectbox(
    "Chọn trang: ",
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
        # Debug: Hiển thị columns
        with st.expander("🔍 Xem danh sách columns"):
            st.write(df.columns.tolist())
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Tổng đơn hàng", f"{len(df):,}")
        with col2:
            st.metric("👥 Khách hàng", f"{df['customer_unique_id'].nunique():,}")
        with col3:
            # Tìm column payment
            payment_cols = [col for col in df.columns if 'payment' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
            if payment_cols:
                avg_revenue = df[payment_cols[0]].mean()
                st.metric("💰 Doanh thu TB", f"R$ {avg_revenue:,.2f}")
            else:
                st.metric("💰 Doanh thu TB", "N/A")
        with col4:
            if 'review_score' in df.columns:
                avg_review = df['review_score'].mean()
                st.metric("⭐ Review TB", f"{avg_review:.2f}/5")
            else:
                st.metric("⭐ Review TB", "N/A")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Phân bố Review Score")
            if 'review_score' in df.columns:
                review_dist = df['review_score'].value_counts().sort_index()
                fig_review = px.bar(
                    x=review_dist.index, 
                    y=review_dist.values,
                    labels={'x': 'Score', 'y': 'Số lượng'},
                    color=review_dist.values, 
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_review, use_container_width=True)
            else:
                st.warning("⚠️ Không có column review_score")
        
        with col2:
            st.subheader("🎯 Customer Clusters Distribution")
            cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
            if cluster_cols:
                cluster_dist = df[cluster_cols[0]].value_counts().sort_index()
                fig_cluster = px.pie(
                    values=cluster_dist.values, 
                    names=[f"Cluster {i}" for i in cluster_dist.index],
                    title='Phân bố Customer Segments',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("⚠️ Không có column cluster")
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue Distribution")
            payment_cols = [col for col in df.columns if 'payment' in col.lower() or 'value' in col.lower()]
            if payment_cols:
                fig_revenue = px.histogram(
                    df, 
                    x=payment_cols[0],
                    nbins=50,
                    title='Distribution of Order Values',
                    labels={payment_cols[0]: 'Order Value (R$)'}
                )
                st.plotly_chart(fig_revenue, use_container_width=True)
            else:
                st.warning("⚠️ Không có column payment/value")
        
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
            else:
                st.warning("⚠️ Không có column order_status")
        
        # RFM Metrics
        st.markdown("---")
        st.subheader("📊 RFM Analysis Metrics")
        
        rfm_mapping = {
            'recency': [col for col in df.columns if 'recency' in col.lower()],
            'frequency': [col for col in df.columns if 'frequency' in col.lower()],
            'monetary': [col for col in df.columns if 'monetary' in col.lower()]
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if rfm_mapping['recency']:
                st.metric("Recency (trung bình)", f"{df[rfm_mapping['recency'][0]].mean():.0f} ngày")
            else:
                st.metric("Recency", "N/A")
        with col2:
            if rfm_mapping['frequency']:
                st.metric("Frequency (trung bình)", f"{df[rfm_mapping['frequency'][0]].mean():.2f} đơn")
            else:
                st.metric("Frequency", "N/A")
        with col3:
            if rfm_mapping['monetary']:
                st.metric("Monetary (trung bình)", f"R$ {df[rfm_mapping['monetary'][0]].mean():,.2f}")
            else:
                st.metric("Monetary", "N/A")
        
        st.success("✅ Dashboard hoàn chỉnh!")
        
    else:
        st.error("❌ Không tìm thấy data. Vui lòng upload file hoặc kiểm tra folder data/")

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# =============================================================================
elif page == "👥 Customer Segmentation":
    st.header("👥 Phân khúc Khách hàng")
    
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
            rfm_df = df[['customer_unique_id', 'recency', 'frequency', 'monetary']].drop_duplicates()
            if 'kmeans_cluster' in df.columns:
                rfm_df = df[['customer_unique_id', 'recency', 'frequency', 'monetary', 'kmeans_cluster']].drop_duplicates()
            st.info("ℹ️ Sử dụng data từ file đã load")
        else:
            st.error("❌ Không có data!")
            st.stop()
    
    if rfm_df is not None:
        st.markdown("---")
        st.subheader("📊 Cluster Profile")
        
        if 'kmeans_cluster' not in rfm_df.columns:
            st.warning("⚠️ Chưa có cluster. Đang thực hiện K-Means...")
            
            rfm_features = rfm_df[['recency', 'frequency', 'monetary']].fillna(0)
            n_clusters = st.slider("Chọn số cụm (K): ", 2, 10, 4)
            
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            rfm_df['kmeans_cluster'] = kmeans.fit_predict(rfm_scaled)
            
            st.success(f"✅ Clustering hoàn tất!")
        
        cluster_profile = rfm_df.groupby('kmeans_cluster')[['recency', 'frequency', 'monetary']].mean()
        st.dataframe(cluster_profile.style.background_gradient(cmap='viridis'))
        
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
        col1, col2 = st.columns(2)
        
        with col1:
            unique_customers = df['customer_unique_id'].unique()
            selected_customer = st.selectbox(
                "Chọn Customer ID: ",
                unique_customers[:100]
            )
        
        with col2:
            product_search = st.text_input("Product ID (tùy chọn): ")
        
        if st.button("🔍 Tìm khuyến nghị", type="primary"):
            if 'svd' in models:
                try:
                    user_predictions = []
                    all_products = df['product_id'].unique()
                    purchased = df[df['customer_unique_id'] == selected_customer]['product_id'].unique()
                    
                    for product_id in all_products[:500]:
                        if product_id not in purchased:
                            try:
                                pred = models['svd'].predict(str(selected_customer), str(product_id))
                                user_predictions.append({
                                    'product_id': product_id,
                                    'predicted_score': pred.est
                                })
                            except:
                                pass
                    
                    if user_predictions:
                        recommendations = pd.DataFrame(user_predictions)
                        recommendations = recommendations.sort_values('predicted_score', ascending=False).head(10)
                        
                        st.success(f"✅ Tìm thấy {len(recommendations)} sản phẩm khuyến nghị (SVD Model)")
                        
                        for idx, row in recommendations.iterrows():
                            with st.expander(f"#{idx+1}: Product {row['product_id'][:30]}... - Score: {row['predicted_score']:.2f}/5.0"):
                                st.write(f"- Predicted Score: ⭐ {row['predicted_score']:.2f}")
                                st.write(f"- Product ID: `{row['product_id']}`")
                    else:
                        st.warning("⚠️ Không có khuyến nghị cho khách hàng này")
                        
                except Exception as e:
                    st.error(f"❌ Lỗi SVD: {e}")
            else:
                st.warning("⚠️ Chưa có SVD Model. Sử dụng Rule-based recommendation.")
                
                purchased = df[df['customer_unique_id'] == selected_customer]['product_id'].unique()
                product_ratings = df.groupby('product_id').agg({
                    'review_score': ['mean', 'count']
                }).reset_index()
                product_ratings.columns = ['product_id', 'avg_score', 'num_reviews']
                
                recommendations = product_ratings[~product_ratings['product_id'].isin(purchased)]
                recommendations = recommendations[recommendations['num_reviews'] >= 5]
                recommendations = recommendations.sort_values('avg_score', ascending=False).head(10)
                
                st.success(f"✅ Tìm thấy {len(recommendations)} sản phẩm khuyến nghị (Rule-based)")
        
        st.markdown("---")
        if 'svd' in models:
            st.success("📌 Hệ thống sử dụng **SVD Model** (Surprise Library)")
        else:
            st.info("📌 Hệ thống sử dụng **Rule-based** recommendation")
    else:
        st.error("❌ Không có data!")

# =============================================================================
# PAGE 4: MARKET BASKET ANALYSIS
# =============================================================================
elif page == "🛒 Market Basket Analysis":
    st.header("🛒 Phân tích Xu hướng Mua sắm (Market Basket Analysis)")
    st.info("📌 Phân tích các sản phẩm thường được mua cùng nhau")
    
    if association_rules is not None and len(association_rules) > 0:
        st.subheader("📊 Association Rules (FP-Growth)")
        st.success(f"✅ Tìm thấy {len(association_rules)} luật kết hợp")
        
        st.dataframe(association_rules, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                association_rules.head(20),
                x='Antecedents',
                y='Support',
                title='Top 20 Rules - Support',
                color='Support',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                association_rules.head(20),
                x='Confidence',
                y='Lift',
                size='Support',
                hover_data=['Consequents'],
                title='Confidence vs Lift',
                color='Lift',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        csv = association_rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Association Rules",
            data=csv,
            file_name='association_rules.csv',
            mime='text/csv'
        )
        
        st.success("📌 Sử dụng **FP-Growth Algorithm** (mlxtend)")
        
    else:
        st.warning("⚠️ Chưa có file association_rules.csv. Hiển thị sample data.")
        
        sample_rules = pd.DataFrame({
            'Antecedents': ['Product A', 'Product B', 'Product C', 'Product D'],
            'Consequents': ['Product X', 'Product Y', 'Product Z', 'Product W'],
            'Support': [0.15, 0.12, 0.10, 0.08],
            'Confidence': [0.65, 0.58, 0.72, 0.55],
            'Lift': [2.3, 1.9, 2.7, 2.1]
        })
        
        st.dataframe(sample_rules, use_container_width=True)
        st.info("📌 FP-Growth algorithm sẽ được tích hợp đầy đủ trong Tuần 2")
    
    if df is None:
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
            if 'classification' in models:
                try:
                    input_data = pd.DataFrame({
                        'payment_value': [pred_payment],
                        'freight_value': [pred_freight],
                        'order_items': [pred_items],
                        'recency': [pred_recency],
                        'frequency': [pred_frequency]
                    })
                    
                    predicted_score = models['classification'].predict(input_data)[0]
                    predicted_score = min(5.0, max(1.0, predicted_score))
                    
                    st.success(f"⭐ Dự đoán Review Score: **{predicted_score:.1f}/5.0** (ML Model)")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi model: {e}")
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
                    st.success(f"⭐ Dự đoán Review Score: **{predicted_score:.1f}/5.0** (Rule-based)")
            else:
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
                
                st.success(f"⭐ Dự đoán Review Score: **{predicted_score:.1f}/5.0** (Rule-based)")
            
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
            if 'regression' in models:
                try:
                    input_data = pd.DataFrame({
                        'recency': [pred2_recency],
                        'frequency': [pred2_frequency],
                        'items': [pred2_items],
                        'freight': [pred2_freight]
                    })
                    
                    predicted_value = models['regression'].predict(input_data)[0]
                    
                    st.success(f"💰 Dự đoán: **R$ {predicted_value:,.2f}** (ML Model)")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi model: {e}")
                    base_value = 100.0
                    if pred2_frequency > 5:
                        base_value *= 1.5
                    if pred2_items > 3:
                        base_value *= 1.3
                    base_value += pred2_freight * 0.5
                    st.success(f"💰 Dự đoán: **R$ {base_value:,.2f}** (Rule-based)")
            else:
                base_value = 100.0
                
                if pred2_frequency > 5:
                    base_value *= 1.5
                
                if pred2_items > 3:
                    base_value *= 1.3
                
                base_value += pred2_freight * 0.5
                
                st.success(f"💰 Dự đoán: **R$ {base_value:,.2f}** (Rule-based)")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'classification' in models:
            st.success("✅ Classification Model: Sẵn sàng")
        else:
            st.warning("⚠️ Classification Model: Chưa có")
    with col2:
        if 'regression' in models:
            st.success("✅ Regression Model: Sẵn sàng")
        else:
            st.warning("⚠️ Regression Model: Chưa có")
    with col3:
        if 'svd' in models:
            st.success("✅ SVD Model: Sẵn sàng")
        else:
            st.warning("⚠️ SVD Model: Chưa có")

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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if Path("svd_model.pkl").exists():
            st.success("✅ SVD Model")
        else:
            st.warning("⚠️ SVD Model")
    with col2:
        if Path("classification_model.pkl").exists():
            st.success("✅ Classification Model")
        else:
            st.warning("⚠️ Classification Model")
    with col3:
        if Path("regression_model.pkl").exists():
            st.success("✅ Regression Model")
        else:
            st.warning("⚠️ Regression Model")
    
    st.markdown("---")
    
    st.subheader("📤 Upload New Data")
    uploaded = st.file_uploader("Upload data mới (CSV)", type=['csv'])
    if uploaded:
        st.success("✅ Upload thành công!")
    
    st.markdown("---")
    st.success("Admin xin chào!")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    🎓 Big Data Analytics - Machine Learning Project<br>
    Built with ❤️ using Streamlit<br>
    © 2026 - Brazilian E-Commerce Analytics - HCMUTE N12
</div>
""", unsafe_allow_html=True)
