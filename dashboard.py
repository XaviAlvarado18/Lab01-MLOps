import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Walmart Sales Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess all datasets"""
    try:
        # Load datasets
        train_df = pd.read_csv('./data/train.csv')
        stores_df = pd.read_csv('./data/stores.csv')
        features_df = pd.read_csv('./data/features.csv')
        test_df = pd.read_csv('./data/test.csv')
        
        # Convert date columns
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        
        # Merge datasets for comprehensive analysis
        # Merge train with stores
        train_merged = train_df.merge(stores_df, on='Store', how='left')
        
        # Merge with features
        train_complete = train_merged.merge(features_df, on=['Store', 'Date'], how='left')
        
        return train_df, stores_df, features_df, test_df, train_complete
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def create_sales_overview(df):
    """Create sales overview metrics and charts"""
    st.subheader("📊 Sales Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df['Weekly_Sales'].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        avg_weekly_sales = df['Weekly_Sales'].mean()
        st.metric("Avg Weekly Sales", f"${avg_weekly_sales:,.0f}")
    
    with col3:
        total_stores = df['Store'].nunique()
        st.metric("Total Stores", total_stores)
    
    with col4:
        total_departments = df['Dept'].nunique()
        st.metric("Total Departments", total_departments)
    
    # Sales trend over time
    st.subheader("📈 Sales Trend Over Time")
    
    # Aggregate sales by date
    daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    fig_trend = px.line(
        daily_sales, 
        x='Date', 
        y='Weekly_Sales',
        title="Total Weekly Sales Over Time",
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Date': 'Date'}
    )
    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

def create_store_analysis(df, stores_df):
    """Create store performance analysis"""
    st.subheader("🏪 Store Performance Analysis")
    
    # Store performance metrics
    store_performance = df.groupby('Store').agg({
        'Weekly_Sales': ['sum', 'mean', 'count']
    }).round(2)
    store_performance.columns = ['Total_Sales', 'Avg_Sales', 'Weeks_Count']
    store_performance = store_performance.reset_index()
    
    # Merge with store info
    store_performance = store_performance.merge(stores_df, on='Store', how='left')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performing stores
        top_stores = store_performance.nlargest(10, 'Total_Sales')
        fig_top = px.bar(
            top_stores,
            x='Store',
            y='Total_Sales',
            color='Type',
            title="Top 10 Stores by Total Sales",
            labels={'Total_Sales': 'Total Sales ($)'}
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Sales by store type
        type_sales = store_performance.groupby('Type')['Total_Sales'].sum().reset_index()
        fig_type = px.pie(
            type_sales,
            values='Total_Sales',
            names='Type',
            title="Sales Distribution by Store Type"
        )
        st.plotly_chart(fig_type, use_container_width=True)
    
    # Store size vs performance
    st.subheader("Store Size vs Performance")
    fig_size = px.scatter(
        store_performance,
        x='Size',
        y='Total_Sales',
        color='Type',
        size='Avg_Sales',
        hover_data=['Store'],
        title="Store Size vs Total Sales",
        labels={'Size': 'Store Size (sq ft)', 'Total_Sales': 'Total Sales ($)'}
    )
    st.plotly_chart(fig_size, use_container_width=True)

def create_department_analysis(df):
    """Create department performance analysis"""
    st.subheader("🏬 Department Analysis")
    
    # Department performance
    dept_performance = df.groupby('Dept').agg({
        'Weekly_Sales': ['sum', 'mean', 'count']
    }).round(2)
    dept_performance.columns = ['Total_Sales', 'Avg_Sales', 'Weeks_Count']
    dept_performance = dept_performance.reset_index()
    dept_performance = dept_performance.sort_values('Total_Sales', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top departments
        top_depts = dept_performance.head(15)
        fig_dept = px.bar(
            top_depts,
            x='Dept',
            y='Total_Sales',
            title="Top 15 Departments by Sales",
            labels={'Total_Sales': 'Total Sales ($)', 'Dept': 'Department'}
        )
        fig_dept.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        # Department consistency (CV)
        dept_performance['CV'] = (df.groupby('Dept')['Weekly_Sales'].std() / 
                                 df.groupby('Dept')['Weekly_Sales'].mean()) * 100
        consistency_plot = dept_performance.head(15)
        
        fig_cv = px.scatter(
            consistency_plot,
            x='Avg_Sales',
            y='CV',
            size='Total_Sales',
            hover_data=['Dept'],
            title="Department Consistency (Lower CV = More Consistent)",
            labels={'CV': 'Coefficient of Variation (%)', 'Avg_Sales': 'Average Sales ($)'}
        )
        st.plotly_chart(fig_cv, use_container_width=True)

def create_seasonal_analysis(df):
    """Create seasonal and holiday analysis"""
    st.subheader("🎄 Seasonal & Holiday Analysis")
    
    # Add time features
    df_temp = df.copy()
    df_temp['Year'] = df_temp['Date'].dt.year
    df_temp['Month'] = df_temp['Date'].dt.month
    df_temp['Week'] = df_temp['Date'].dt.isocalendar().week
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Holiday vs Non-holiday sales
        holiday_sales = df_temp.groupby('IsHoliday')['Weekly_Sales'].agg(['sum', 'mean']).reset_index()
        holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})
        
        fig_holiday = px.bar(
            holiday_sales,
            x='IsHoliday',
            y='mean',
            title="Average Sales: Holiday vs Non-Holiday",
            labels={'mean': 'Average Weekly Sales ($)', 'IsHoliday': 'Period Type'}
        )
        st.plotly_chart(fig_holiday, use_container_width=True)
    
    with col2:
        # Monthly sales pattern
        monthly_sales = df_temp.groupby('Month')['Weekly_Sales'].mean().reset_index()
        monthly_sales['Month_Name'] = pd.to_datetime(monthly_sales['Month'], format='%m').dt.strftime('%B')
        
        fig_monthly = px.line(
            monthly_sales,
            x='Month',
            y='Weekly_Sales',
            title="Average Monthly Sales Pattern",
            labels={'Weekly_Sales': 'Average Weekly Sales ($)', 'Month': 'Month'}
        )
        fig_monthly.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

def create_economic_factors_analysis(df_complete):
    """Analyze economic factors impact on sales"""
    if df_complete is None or df_complete.empty:
        st.warning("Economic factors data not available")
        return
    
    st.subheader("💰 Economic Factors Impact")
    
    # Remove rows with missing economic data
    econ_df = df_complete.dropna(subset=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
    
    if econ_df.empty:
        st.warning("No complete economic data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature vs Sales
        fig_temp = px.scatter(
            econ_df.sample(n=min(1000, len(econ_df))),  # Sample for performance
            x='Temperature',
            y='Weekly_Sales',
            title="Temperature vs Weekly Sales",
            labels={'Temperature': 'Temperature (°F)', 'Weekly_Sales': 'Weekly Sales ($)'},
            opacity=0.6
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Fuel Price vs Sales
        fig_fuel = px.scatter(
            econ_df.sample(n=min(1000, len(econ_df))),
            x='Fuel_Price',
            y='Weekly_Sales',
            title="Fuel Price vs Weekly Sales",
            labels={'Fuel_Price': 'Fuel Price ($)', 'Weekly_Sales': 'Weekly Sales ($)'},
            opacity=0.6
        )
        st.plotly_chart(fig_fuel, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Economic Factors Correlation")
    corr_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    corr_data = econ_df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_data,
        title="Correlation Matrix: Sales vs Economic Factors",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">🛒 Walmart Sales Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        train_df, stores_df, features_df, test_df, train_complete = load_data()
    
    if train_df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(train_df['Date'].min(), train_df['Date'].max()),
        min_value=train_df['Date'].min(),
        max_value=train_df['Date'].max()
    )
    
    # Store filter
    selected_stores = st.sidebar.multiselect(
        "Select Stores",
        options=sorted(train_df['Store'].unique()),
        default=sorted(train_df['Store'].unique())[:10]  # Default to first 10 stores
    )
    
    # Department filter
    selected_depts = st.sidebar.multiselect(
        "Select Departments",
        options=sorted(train_df['Dept'].unique()),
        default=sorted(train_df['Dept'].unique())[:20]  # Default to first 20 departments
    )
    
    # Apply filters
    filtered_df = train_df[
        (train_df['Date'] >= pd.to_datetime(date_range[0])) &
        (train_df['Date'] <= pd.to_datetime(date_range[1])) &
        (train_df['Store'].isin(selected_stores)) &
        (train_df['Dept'].isin(selected_depts))
    ]
    
    if train_complete is not None:
        filtered_complete = train_complete[
            (train_complete['Date'] >= pd.to_datetime(date_range[0])) &
            (train_complete['Date'] <= pd.to_datetime(date_range[1])) &
            (train_complete['Store'].isin(selected_stores)) &
            (train_complete['Dept'].isin(selected_depts))
        ]
    else:
        filtered_complete = None
    
    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Data Summary**")
    st.sidebar.write(f"Records: {len(filtered_df):,}")
    st.sidebar.write(f"Stores: {filtered_df['Store'].nunique()}")
    st.sidebar.write(f"Departments: {filtered_df['Dept'].nunique()}")
    st.sidebar.write(f"Date Range: {filtered_df['Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🏪 Stores", 
        "🏬 Departments", 
        "🎄 Seasonal", 
        "💰 Economic Factors"
    ])
    
    with tab1:
        create_sales_overview(filtered_df)
    
    with tab2:
        create_store_analysis(filtered_df, stores_df)
    
    with tab3:
        create_department_analysis(filtered_df)
    
    with tab4:
        create_seasonal_analysis(filtered_df)
    
    with tab5:
        create_economic_factors_analysis(filtered_complete)
    
    # Raw data view
    with st.expander("🔍 View Raw Data"):
        st.dataframe(filtered_df.head(1000))  # Show first 1000 rows for performance

if __name__ == "__main__":
    main()