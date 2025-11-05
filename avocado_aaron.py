# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Avocado Analysis Dashboard",
    page_icon="🥑",
    layout="wide"
)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\Aaron Tan\OneDrive\文档\Aaron(DSTB)\avo.csv")
    return data

data = load_data()

# Features and target variable
X = data[['average_price', 'total_volume', 'total_bags']]
y = data['type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
@st.cache_resource
def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

rf_classifier = train_model(X_train, y_train)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))

# Streamlit app layout
st.title("🥑 Avocado Analysis Dashboard")

# Sidebar navigation
st.sidebar.header("📍 Navigation")
page = st.sidebar.selectbox("Choose a page", ["Introduction", "EDA", "Prediction"])

# ==================== INTRODUCTION PAGE ====================
if page == "Introduction":
    # Enhanced Introduction with Key Metrics
    st.header("Aaron Tan Wen Zhuan (0137612) Welcome to the Avocado Type Predictor!")
    
    # Key Metrics at a Glance
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        avg_price = data['average_price'].mean()
        st.metric("Avg Price", f"${avg_price:.2f}")
    with col3:
        total_vol = data['total_volume'].sum() / 1e9
        st.metric("Total Volume", f"{total_vol:.1f}B")
    with col4:
        st.metric("Avocado Types", data['type'].nunique())
    
    st.divider()
    
    # Context and Information
    st.subheader("🥑 About Avocados")
    st.markdown("""
    Avocados are nutrient-rich fruits known for their creamy texture and mild, buttery flavor. 
    Native to Central and South America, they are cultivated worldwide, with Mexico being the largest producer. 
    Avocados are packed with healthy monounsaturated fats, fiber, vitamins (such as B, C, E, and K), and minerals like potassium. 
    They are widely used in salads, sandwiches, smoothies, and the popular Mexican dip, guacamole. 
    Due to their health benefits, avocados are a favorite in balanced diets, promoting heart health, digestion, and skin nourishment.
    """)
    
    # How to Use This App
    st.subheader("💡 How to Use This Dashboard")
    st.info("""
    **📌 Navigation Guide:**
    - **Introduction** - Overview and key statistics (you are here!)
    - **EDA** - Explore data visualizations and trends
    - **Prediction** - Use AI to predict avocado type based on features
    
    **🎯 Quick Tips:**
    - Use the sidebar to navigate between pages
    - Interact with charts to see detailed information
    - Try the prediction tool with different values
    """)
    
    # Display avocado image
    st.image('avocado.jpg', caption="Fresh Avocados", use_column_width=True)
    
    # Key Insights
    st.subheader("🔍 Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        organic_avg = data[data['type'] == 'organic']['average_price'].mean()
        conventional_avg = data[data['type'] == 'conventional']['average_price'].mean()
        price_diff = organic_avg - conventional_avg
        st.success(f"💰 Organic avocados cost **${price_diff:.2f} more** on average than conventional ones")
    
    with col2:
        max_price = data['average_price'].max()
        min_price = data['average_price'].min()
        st.warning(f"📈 Price range: **${min_price:.2f}** to **${max_price:.2f}**")

# ==================== EDA PAGE ====================
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")
    
    # Original EDA: Average Price by Type
    st.subheader("💵 Average Price by Type")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=data, x='type', y='average_price', estimator='mean', palette='viridis', ax=ax1)
        ax1.set_title("Average Price by Type", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Type", fontsize=12)
        ax1.set_ylabel("Average Price ($)", fontsize=12)
        st.pyplot(fig1)
    
    with col2:
        st.markdown("### 📌 Key Findings")
        organic_price = data[data['type'] == 'organic']['average_price'].mean()
        conventional_price = data[data['type'] == 'conventional']['average_price'].mean()
        
        st.metric("Organic Avg", f"${organic_price:.2f}")
        st.metric("Conventional Avg", f"${conventional_price:.2f}")
        
        percentage_diff = ((organic_price - conventional_price) / conventional_price) * 100
        st.info(f"Organic avocados are **{percentage_diff:.1f}%** more expensive")
    
    st.divider()
    
    # NEW EDA COMPONENT: Price Distribution Comparison
    st.subheader("📈 Additional Analysis: Price Distribution Comparison")
    
    tab1, tab2 = st.tabs(["📊 Distribution Plot", "📉 Box Plot"])
    
    with tab1:
        st.markdown("**Compare the distribution of prices between organic and conventional avocados**")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Create histogram for both types
        organic_data = data[data['type'] == 'organic']['average_price']
        conventional_data = data[data['type'] == 'conventional']['average_price']
        
        ax2.hist(organic_data, bins=30, alpha=0.6, label='Organic', color='green', edgecolor='black')
        ax2.hist(conventional_data, bins=30, alpha=0.6, label='Conventional', color='orange', edgecolor='black')
        
        ax2.set_xlabel('Average Price ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Price Distribution: Organic vs Conventional', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig2)
        
        st.success("""
        **💡 Insight:** The distribution shows that conventional avocados have a more concentrated 
        price range, while organic avocados show more price variability and tend to be in higher price ranges.
        """)
    
    with tab2:
        st.markdown("**Box plot comparison showing median, quartiles, and outliers**")
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        # Create box plot
        data.boxplot(column='average_price', by='type', ax=ax3, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'))
        
        ax3.set_xlabel('Type', fontsize=12)
        ax3.set_ylabel('Average Price ($)', fontsize=12)
        ax3.set_title('Price Distribution by Type (Box Plot)', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove the default title
        
        st.pyplot(fig3)
        
        # Statistical summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Organic Statistics:**")
            st.write(f"- Median: ${organic_data.median():.2f}")
            st.write(f"- Std Dev: ${organic_data.std():.2f}")
            st.write(f"- Range: ${organic_data.min():.2f} - ${organic_data.max():.2f}")
        
        with col2:
            st.markdown("**Conventional Statistics:**")
            st.write(f"- Median: ${conventional_data.median():.2f}")
            st.write(f"- Std Dev: ${conventional_data.std():.2f}")
            st.write(f"- Range: ${conventional_data.min():.2f} - ${conventional_data.max():.2f}")

# ==================== PREDICTION PAGE ====================
elif page == "Prediction":
    st.header("🤖 Random Forest Classifier Prediction")
    
    # Model Performance
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    with col2:
        st.metric("Training Samples", f"{len(X_train):,}")
    with col3:
        st.metric("Test Samples", f"{len(X_test):,}")
    
    st.divider()
    
    # Prediction Interface
    st.subheader("🎯 Predict Avocado Type")
    st.markdown("Enter the features below to predict whether the avocado is **organic** or **conventional**:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input fields
        avg_price = st.number_input(
            "💵 Average Price ($)",
            value=1.5,
            step=0.1,
            min_value=0.0,
            max_value=5.0,
            help="Enter the average price of the avocado"
        )
        
        total_vol = st.number_input(
            "📦 Total Volume",
            value=100000,
            step=10000,
            min_value=0,
            help="Enter the total volume sold"
        )
        
        total_bags = st.number_input(
            "🛍️ Total Bags",
            value=5000,
            step=500,
            min_value=0,
            help="Enter the total number of bags"
        )
        
        # Prediction button
        predict_button = st.button("🔮 Predict Type", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📌 Input Guide")
        st.info("""
        **Typical Ranges:**
        - Price: $0.50 - $3.00
        - Volume: 10K - 500K
        - Bags: 1K - 50K
        
        **Tips:**
        - Higher prices often indicate organic
        - Try different combinations
        - Compare with dataset stats
        """)
    
    # Perform prediction
    if predict_button:
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'average_price': [avg_price],
            'total_volume': [total_vol],
            'total_bags': [total_bags]
        })
        
        # Predict using the trained model
        prediction = rf_classifier.predict(user_input)[0]
        prediction_proba = rf_classifier.predict_proba(user_input)[0]
        
        # Display the prediction result
        st.divider()
        st.subheader("🎉 Prediction Result")
        
        if prediction == 'organic':
            st.success(f"### 🥑 The predicted type is: **{prediction.upper()}**")
            st.balloons()
        else:
            st.info(f"### 🥑 The predicted type is: **{prediction.upper()}**")
        
        # Show confidence
        col1, col2 = st.columns(2)
        with col1:
            confidence = max(prediction_proba) * 100
            st.metric("Prediction Confidence", f"{confidence:.1f}%")
        
        with col2:
            if prediction == 'organic':
                st.metric("Organic Probability", f"{prediction_proba[1]*100:.1f}%")
            else:
                st.metric("Conventional Probability", f"{prediction_proba[0]*100:.1f}%")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>🥑 Avocado Analysis Dashboard | Created by Aaron | Data Analysis & Machine Learning</p>
</div>
""", unsafe_allow_html=True)