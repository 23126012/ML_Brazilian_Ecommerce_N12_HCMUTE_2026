"""
🛒 PHÂN TÍCH THƯƠNG MẠI ĐIỆN TỬ BRAZIL
RUN: streamlit run N12_App.py
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
    page_title="PHÂN TÍCH THƯƠNG MẠI ĐIỆN TỬ BRAZIL",
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛒 PHÂN TÍCH THƯƠNG MẠI ĐIỆN TỬ BRAZIL</h1>', unsafe_allow_html=True)
st.markdown("---")

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data
def load_data():
    try:
        data_path = Path(".")
        
        orders = pd.read_csv(data_path / "olist_orders_dataset.csv")
        customers = pd.read_csv(data_path / "olist_customers_dataset.csv")
        items = pd.read_csv(data_path / "olist_order_items_dataset.csv")
        payments = pd.read_csv(data_path / "olist_order_payments_dataset.csv")
        reviews = pd.read_csv(data_path / "olist_order_reviews_dataset.csv")
        products = pd.read_csv(data_path / "olist_products_dataset.csv")
        
        df = orders.merge(customers, on='customer_id', how='left')
        df = df.merge(items, on='order_id', how='left')
        
        payment_agg = payments.groupby('order_id')['payment_value'].sum().reset_index()
        payment_agg.columns = ['order_id', 'payment_value']
        df = df.merge(payment_agg, on='order_id', how='left')
        
        df = df.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')
        df = df.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
        
        st.sidebar.success("✅ Data loaded from 9 CSV files")
        st.sidebar.info(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        return None

df = load_data()

# =============================================================================
# LOAD RFM DATA
# =============================================================================
@st.cache_data
def load_rfm_data():
    try:
        if Path("rfm_with_clusters.csv").exists():
            rfm_df = pd.read_csv("rfm_with_clusters.csv")
            st.sidebar.success("✅ RFM loaded: rfm_with_clusters.csv")
            return rfm_df
        elif Path("rfm_analysis.csv").exists():
            rfm_df = pd.read_csv("rfm_analysis.csv")
            st.sidebar.success("✅ RFM loaded: rfm_analysis.csv")
            return rfm_df
        else:
            return None
    except:
        return None

rfm_df = load_rfm_data()

# =============================================================================
# LOAD ML MODELS
# =============================================================================
@st.cache_resource
def load_models():
    models = {}
    
    # Load Full Pipeline (Review Score) - ƯU TIÊN FILE MỚI
    if Path("full_pipeline.pkl").exists():
        try:
            models['pipeline'] = joblib.load("full_pipeline.pkl")
            st.sidebar.success("✅ Full Pipeline loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Pipeline: {e}")
    
    # Load Old Classifier (Backup)
    elif Path("best_review_classifier.pkl").exists():
        try:
            models['classifier'] = joblib.load("best_review_classifier.pkl")
            st.sidebar.success("✅ Review Classifier loaded (old)")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Classifier: {e}")
    
    # Load Payment Regressor
    if Path("best_payment_regressor.pkl").exists():
        try:
            models['regressor'] = joblib.load("best_payment_regressor.pkl")
            st.sidebar.success("✅ Payment Regressor loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Regressor: {e}")
    
    # Load TFIDF Vectorizer
    if Path("tfidf_vectorizer.pkl").exists():
        try:
            models['tfidf'] = joblib.load("tfidf_vectorizer.pkl")
            st.sidebar.success("✅ TFIDF Vectorizer loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ TFIDF: {e}")
    
    return models

models = load_models()

# =============================================================================
# LOAD ASSOCIATION RULES
# =============================================================================
@st.cache_data
def load_association_rules():
    try:
        if Path("association_rules.csv").exists():
            rules = pd.read_csv("association_rules.csv")
            if len(rules) > 0:
                st.sidebar.success(f"✅ Association Rules ({len(rules)})")
                return rules
            else:
                st.sidebar.warning("⚠️ association_rules.csv rỗng (0 rows)")
                return None
        return None
    except:
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
        with st.expander("🔍 Xem danh sách columns"):
            st.write(df.columns.tolist())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Tổng đơn hàng", f"{len(df):,}")
        with col2:
            if 'customer_unique_id' in df.columns:
                st.metric("👥 Khách hàng", f"{df['customer_unique_id'].nunique():,}")
            else:
                st.metric("👥 Khách hàng", "N/A")
        with col3:
            if 'payment_value' in df.columns:
                avg_revenue = df['payment_value'].mean()
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Phân bố Review Score")
            if 'review_score' in df.columns:
                review_dist = df['review_score'].value_counts().sort_index()
                fig_review = px.bar(x=review_dist.index, y=review_dist.values, color=review_dist.values, color_continuous_scale='Viridis')
                st.plotly_chart(fig_review, use_container_width=True)
            else:
                st.warning("⚠️ Không có column review_score")
        
        with col2:
            st.subheader("📊 Order Status")
            if 'order_status' in df.columns:
                status_counts = df['order_status'].value_counts()
                fig_status = px.pie(values=status_counts.values, names=status_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.warning("⚠️ Không có column order_status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue Distribution")
            if 'payment_value' in df.columns:
                fig_revenue = px.histogram(df, x='payment_value', nbins=50, color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig_revenue, use_container_width=True)
            else:
                st.warning("⚠️ Không có column payment_value")
        
        with col2:
            st.subheader("🏷️ Product Categories")
            if 'product_category_name' in df.columns:
                cat_counts = df['product_category_name'].value_counts().head(10)
                fig_cat = px.bar(x=cat_counts.values, y=cat_counts.index, orientation='h', color=cat_counts.values, color_continuous_scale='Blues')
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.warning("⚠️ Không có column product_category_name")
        
        if rfm_df is not None:
            st.markdown("---")
            st.subheader("📊 RFM Analysis Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Recency' in rfm_df.columns:
                    st.metric("Recency (trung bình)", f"{rfm_df['Recency'].mean():.0f} ngày")
            with col2:
                if 'Frequency' in rfm_df.columns:
                    st.metric("Frequency (trung bình)", f"{rfm_df['Frequency'].mean():.2f} đơn")
            with col3:
                if 'Monetary' in rfm_df.columns:
                    st.metric("Monetary (trung bình)", f"R$ {rfm_df['Monetary'].mean():,.2f}")
        
        st.success("✅ Dashboard hoàn chỉnh!")
    else:
        st.error("❌ Không tìm thấy data!")

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTATION (ĐÃ SỬA LỖI MATPLOTLIB)
# =============================================================================
elif page == "👥 Customer Segmentation":
    st.header("👥 Phân khúc Khách hàng")
    
    if rfm_df is not None:
        current_rfm = rfm_df
        st.info("ℹ️ Sử dụng data từ rfm_with_clusters.csv")
    else:
        st.error("❌ Không có RFM data!")
        st.stop()
    
    if current_rfm is not None:
        st.markdown("---")
        st.subheader("📊 Cluster Profile")
        
        cluster_col = None
        if 'Cluster_KMeans' in current_rfm.columns:
            cluster_col = 'Cluster_KMeans'
            st.info("📌 Sử dụng KMeans Clustering")
        elif 'Cluster_GMM' in current_rfm.columns:
            cluster_col = 'Cluster_GMM'
            st.info("📌 Sử dụng GMM Clustering")
        
        if cluster_col:
            cluster_profile = current_rfm.groupby(cluster_col)[['Recency', 'Frequency', 'Monetary']].mean()
            
            # ✅ SỬA LỖI: Thay background_gradient bằng highlight_max (không cần matplotlib)
            st.dataframe(cluster_profile.style.highlight_max(color='lightgreen'))
            
            st.markdown("### 🎯 Đặc điểm từng cụm:")
            for cluster_id in cluster_profile.index:
                row = cluster_profile.loc[cluster_id]
                count = len(current_rfm[current_rfm[cluster_col] == cluster_id])
                
                if row['Frequency'] > cluster_profile['Frequency'].median() and row['Monetary'] > cluster_profile['Monetary'].median():
                    label, desc, color = "🌟 Champions", "Khách hàng VIP", "🟢"
                elif row['Recency'] < cluster_profile['Recency'].median():
                    label, desc, color = "🆕 New Customers", "Khách mới", "🔵"
                elif row['Frequency'] > cluster_profile['Frequency'].median():
                    label, desc, color = "💎 Loyal", "Khách trung thành", "🟡"
                else:
                    label, desc, color = "⚠️ At Risk", "Cần chăm sóc", "🔴"
                
                with st.expander(f"**{color} Cluster {cluster_id}: {label}** ({count} customers)"):
                    st.write(f"📝 {desc}")
                    st.write(f"- Recency: {row['Recency']:.0f} ngày")
                    st.write(f"- Frequency: {row['Frequency']:.2f} đơn")
                    st.write(f"- Monetary: R$ {row['Monetary']:,.2f}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(current_rfm, names=cluster_col, color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if all(col in current_rfm.columns for col in ['Recency', 'Frequency', 'Monetary']):
                    fig_3d = px.scatter_3d(current_rfm, x='Recency', y='Frequency', z='Monetary', color=current_rfm[cluster_col].astype(str))
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            csv = current_rfm.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name='customer_segments.csv',
                mime='text/csv'
            )
        else:
            st.warning("⚠️ Không có column cluster")

# =============================================================================
# PAGE 3: RECOMMENDATION SYSTEM
# =============================================================================
elif page == "⭐ Recommendation System":
    st.header("⭐ Hệ thống Khuyến nghị Sản phẩm")
    
    if df is not None:
        has_customer = 'customer_unique_id' in df.columns
        has_product = 'product_id' in df.columns
        has_review = 'review_score' in df.columns
        
        st.info(f"📌 product_id: {has_product} | review_score: {has_review}")
        
        if not has_customer:
            st.error("❌ Không có column customer_unique_id")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            unique_customers = df['customer_unique_id'].unique()
            selected_customer = st.selectbox("Chọn Customer ID: ", unique_customers[:100])
        
        with col2:
            if has_product:
                all_products = df['product_id'].unique()
                selected_product = st.selectbox("Chọn sản phẩm: ", all_products[:100])
        
        if st.button("🔍 Tìm khuyến nghị", type="primary"):
            if has_product and has_review:
                purchased = df[df['customer_unique_id'] == selected_customer]['product_id'].unique()
                product_ratings = df.groupby('product_id').agg({'review_score': ['mean', 'count']}).reset_index()
                product_ratings.columns = ['product_id', 'avg_score', 'num_reviews']
                recommendations = product_ratings[~product_ratings['product_id'].isin(purchased)]
                recommendations = recommendations[recommendations['num_reviews'] >= 5].sort_values('avg_score', ascending=False).head(10)
                
                if len(recommendations) > 0:
                    st.success(f"✅ Tìm thấy {len(recommendations)} khuyến nghị")
                    for idx, row in recommendations.iterrows():
                        with st.expander(f"#{idx+1}: Product {str(row['product_id'])[:50]}... - Score: {row['avg_score']:.2f}/5.0"):
                            st.write(f"- Average Score: ⭐ {row['avg_score']:.2f}")
                            st.write(f"- Number of Reviews: 📝 {row['num_reviews']}")
                else:
                    st.warning("⚠️ Không có khuyến nghị")
            else:
                st.error(f"❌ Thiếu columns: product={has_product}, review={has_review}")
        
        st.markdown("---")
        st.info("📌 Recommendation dựa trên review_score trung bình của sản phẩm")
    else:
        st.error("❌ Không có data!")

# =============================================================================
# PAGE 4: MARKET BASKET ANALYSIS
# =============================================================================
elif page == "🛒 Market Basket Analysis":
    st.header("🛒 Phân tích Xu hướng Mua sắm")
    
    if association_rules is not None and len(association_rules) > 0:
        st.subheader("📊 Association Rules (FP-Growth)")
        st.success(f"✅ Tìm thấy {len(association_rules)} luật kết hợp")
        st.dataframe(association_rules, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'antecedents' in association_rules.columns and 'support' in association_rules.columns:
                fig = px.bar(association_rules.head(20), x='antecedents', y='support', color='support', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'confidence' in association_rules.columns and 'lift' in association_rules.columns:
                fig = px.scatter(association_rules.head(20), x='confidence', y='lift', size='support', hover_data=['consequents'], color='lift', color_continuous_scale='Plasma')
                st.plotly_chart(fig, use_container_width=True)
        
        csv = association_rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Association Rules",
            data=csv,
            file_name='association_rules.csv',
            mime='text/csv'
        )
        st.success("📌 Sử dụng **FP-Growth Algorithm**")
    else:
        st.warning("⚠️ File association_rules.csv rỗng (0 rows)")
        
        sample_rules = pd.DataFrame({
            'antecedents': ['product_A', 'product_B'],
            'consequents': ['product_X', 'product_Y'],
            'support': [0.15, 0.12],
            'confidence': [0.65, 0.58],
            'lift': [2.3, 1.9]
        })
        st.dataframe(sample_rules, use_container_width=True)

# =============================================================================
# PAGE 5: PREDICTIONS (DÙNG FULL_PIPELINE.PKL)
# =============================================================================
elif page == "🎯 Predictions":
    st.header("🎯 Hệ thống Dự đoán Review Score")
    
    # Load model
    if Path("full_pipeline.pkl").exists():
        try:
            pipeline_model = joblib.load("full_pipeline.pkl")
            st.success("✅ Review Pipeline Model loaded")
        except Exception as e:
            st.warning(f"⚠️ Model load error: {e}")
            pipeline_model = None
    else:
        st.error("❌ Không tìm thấy full_pipeline.pkl")
        pipeline_model = None
    
    st.markdown("---")
    
    # Form nhập liệu
    st.subheader("📝 Nhập thông tin đơn hàng")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_price = st.number_input("💰 Price (R$)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
        pred_freight = st.number_input("🚚 Freight Value (R$)", min_value=0.0, max_value=500.0, value=15.0, step=1.0)
        pred_payment = st.number_input("💳 Payment Value (R$)", min_value=0.0, max_value=10000.0, value=120.0, step=1.0)
    
    with col2:
        pred_weight = st.number_input("⚖️ Product Weight (g)", min_value=0, max_value=50000, value=500, step=10)
        pred_length = st.number_input("📏 Product Length (cm)", min_value=0, max_value=200, value=30, step=1)
        pred_height = st.number_input("📐 Product Height (cm)", min_value=0, max_value=200, value=10, step=1)
    
    with col3:
        pred_customer_state = st.selectbox("📍 Customer State", ['SP', 'RJ', 'MG', 'ES', 'PR', 'SC', 'RS', 'BA', 'PE', 'CE'])
        pred_seller_state = st.selectbox("🏪 Seller State", ['SP', 'RJ', 'MG', 'ES', 'PR', 'SC', 'RS', 'BA', 'PE', 'CE'])
        pred_payment_type = st.selectbox("💵 Payment Type", ['credit_card', 'boleto', 'voucher', 'debit_card'])
    
    col4, col5 = st.columns(2)
    
    with col4:
        pred_order_status = st.selectbox("📦 Order Status", ['delivered', 'shipped', 'processing', 'cancelled'])
        pred_category = st.selectbox("🏷️ Product Category", ['electronics', 'home', 'fashion', 'sports', 'beauty', 'food', 'books', 'toys'])
    
    with col5:
        pred_date = st.date_input("📅 Order Date", value=pd.Timestamp.now().date())
        pred_hour = st.slider("🕐 Order Hour", 0, 23, 10)
    
    st.markdown("---")
    
    # Nút predict
    if st.button("🔮 DỰ ĐOÁN REVIEW SCORE", type="primary", use_container_width=True):
        if pipeline_model is not None:
            try:
                # Tạo input data
                input_data = pd.DataFrame([{
                    'price': pred_price,
                    'freight_value': pred_freight,
                    'payment_value': pred_payment,
                    'product_weight_g': pred_weight,
                    'product_length_cm': pred_length,
                    'product_height_cm': pred_height,
                    'order_status': pred_order_status,
                    'payment_type': pred_payment_type,
                    'customer_state': pred_customer_state,
                    'seller_state': pred_seller_state,
                    'product_category_name_english': pred_category,
                    'order_purchase_timestamp': f"{pred_date} {pred_hour:02d}:00:00",
                    'order_month': pd.Timestamp(pred_date).month,
                    'order_hour': pred_hour,
                    'order_dayofweek': pd.Timestamp(pred_date).dayofweek,
                    'total_order_value': pred_price + pred_freight
                }])
                
                # Convert timestamp
                input_data['order_purchase_timestamp'] = pd.to_datetime(input_data['order_purchase_timestamp'])
                
                # Predict
                prediction = pipeline_model.predict(input_data)[0]
                predicted_score = min(5.0, max(1.0, float(prediction)))
                
                # Hiển thị kết quả
                st.markdown("### 🎯 Kết quả dự đoán:")
                
                # Metric
                st.metric("Review Score", f"{predicted_score:.1f}/5.0")
                
                st.markdown("---")
                
                # Gauge chart - HIỂN THỊ CHẮC CHẮN
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_score,
                    title={'text': "Review Score"},
                    gauge={
                        'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#2ecc71", 'thickness': 0.75},
                        'bgcolor': "rgba(0,0,0,0.1)",
                        'borderwidth': 2,
                        'bordercolor': "#2ecc71",
                        'steps': [
                            {'range': [0, 2], 'color': "rgba(231,76,60,0.3)"},
                            {'range': [2, 3.5], 'color': "rgba(241,196,15,0.3)"},
                            {'range': [3.5, 5], 'color': "rgba(46,204,113,0.3)"}
                        ]
                    }
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'size': 24}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Dự đoán hoàn tất: **{predicted_score:.1f}/5.0**")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)[:300]}")
                st.info("💡 Kiểm tra lại input data")
        else:
            st.warning("⚠️ Model chưa được load")
    
    st.markdown("---")
    st.subheader("🔄 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pipeline_model is not None:
            st.success("✅ Review Pipeline (full_pipeline.pkl)")
        else:
            st.warning("⚠️ Review Pipeline")
    
    with col2:
        if Path("best_payment_regressor.pkl").exists():
            st.success("✅ Payment Regressor")
        else:
            st.warning("⚠️ Payment Regressor (Chưa có)")

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
    
    st.subheader("📂 Available Files")
    csv_files = [f for f in Path(".").glob("*.csv")]
    pkl_files = [f for f in Path(".").glob("*.pkl")]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**CSV Files:**")
        for f in csv_files:
            size = f.stat().st_size / 1024
            st.write(f"- {f.name} ({size:.1f} KB)")
    with col2:
        st.write("**PKL Files:**")
        for f in pkl_files:
            size = f.stat().st_size / 1024
            st.write(f"- {f.name} ({size:.1f} KB)")
    
    st.markdown("---")
    if df is not None:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Merged Dataset",
            data=csv_data,
            file_name='brazilian_ecommerce_merged.csv',
            mime='text/csv'
        )
    
    if rfm_df is not None:
        csv_rfm = rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download RFM Data",
            data=csv_rfm,
            file_name='rfm_with_clusters.csv',
            mime='text/csv'
        )
    
    st.markdown("---")
    st.subheader("🔄 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if Path("full_pipeline.pkl").exists():
            if 'pipeline' in models:
                st.success("✅ Review Pipeline (Loaded)")
            else:
                st.warning("⚠️ Review Pipeline (Version incompatible)")
        else:
            st.error("❌ Review Pipeline (Not found)")
    
    with col2:
        if Path("best_payment_regressor.pkl").exists():
            if 'regressor' in models:
                st.success("✅ Payment Regressor (Loaded)")
            else:
                st.warning("⚠️ Payment Regressor (Version incompatible)")
        else:
            st.warning("⚠️ Payment Regressor (Not found)")
    
    st.markdown("---")
    st.success("✅ Admin Panel hoàn tất!")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    🎓 Big Data Analytics - Machine Learning Project<br>
    Built with ❤️ using Streamlit<br>
    © 2026 - NHOM 12 - BIGDATA2026
</div>
""", unsafe_allow_html=True)
