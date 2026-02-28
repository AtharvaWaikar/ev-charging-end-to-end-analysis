"""
EV Charging Station Demand & Battery Health Prediction System
A Streamlit application for electric vehicle charging optimization

Features:
- Real-time charging station occupancy prediction
- EV battery health monitoring and degradation analysis
- Smart charging cost optimization
- Station performance analytics
- AI-powered recommendations for drivers and operators

Author: Atharva_W
Purpose: Demonstrate advanced ML in emerging EV infrastructure technology
Dataset: ev_charging_data.csv (realistic EV charging records)
"""

# ============== IMPORTS ==============
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============== PAGE CONFIGURATION ==============
st.set_page_config(page_title="EV Charging Intelligence Hub", layout="wide")

# ============== CUSTOM STYLING ==============
st.markdown("""
    <style>
    .main-header { 
        font-size: 3em; 
        color: #1f77b4; 
        margin-bottom: 20px;
        font-weight: bold;
    }
    .section-header { 
        font-size: 1.8em; 
        color: #2ca02c; 
        margin-top: 30px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============== DATA LOADING FUNCTION ==============
@st.cache_data
def load_ev_charging_data():
    """
    Load EV charging dataset from CSV file.
    
    Returns:
        pandas.DataFrame: Complete EV charging dataset with calculated features
    """
    try:
        df = pd.read_csv(r"E:\Nits Python\Streamlit\EV Charging project\ev_charging_dataset_clean.csv")
        
        # Convert Timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Extract temporal features

        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['Month'] = df['Timestamp'].dt.month
        
        # Calculate derived features
        df['Charging_Cost'] = df['Energy_Consumed_kWh'] * df['Price_per_kWh']
        df['Charging_Efficiency'] = (df['Energy_Consumed_kWh'] / (df['Energy_Consumed_kWh'] / 0.9)) * 100
        df['Effective_Cost'] = df['Charging_Cost'] / df['Charging_Efficiency'] * 100
        df['Degradation_Rate'] = (1 - df['Battery_Health_Percent'] / 100) / (df['Charge_Cycles'] + 1)
        df['Remaining_Life_Months'] = (df['Battery_Health_Percent'] - 80) / (df['Degradation_Rate'] + 0.001) * 12
        df['Remaining_Life_Months'] = df['Remaining_Life_Months'].clip(lower=0)
        
        return df
    
    except FileNotFoundError:
        st.error("❌ Error: 'ev_charging_dataset_clean.csv' file not found!")
        st.stop()

# ============== STATION OCCUPANCY PREDICTION ==============
def train_occupancy_model(df):
    """
    Train a machine learning model to predict charging station occupancy.
    """
    feature_cols = ['Hour', 'DayOfWeek', 'Temperature_C', 'Humidity_Percent', 'Month']
    X = df[feature_cols].copy()
    y = (df['Station_Occupancy_Percent'] > 70).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, accuracy, feature_importance

# ============== BATTERY HEALTH PREDICTION ==============
def train_battery_health_model(df):
    """
    Train model to predict EV battery health degradation.
    """
    feature_cols = ['Vehicle_Age_Months', 'Charge_Cycles', 'Avg_Charging_Temperature_C', 'Fast_Charge_Count', 'Depth_of_Discharge_Percent']
    X = df[feature_cols].copy()
    y = df['Battery_Health_Percent'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, rmse, feature_importance, y_test, y_pred

# ============== CHARGING COST OPTIMIZATION ==============
def get_optimal_charging_strategy(df, battery_current_percent, battery_capacity_kwh, target_percent=80):
    """
    Recommend optimal charging strategy based on cost and time.
    """
    energy_needed = battery_capacity_kwh * (target_percent - battery_current_percent) / 100
    
    hourly_stats = df.groupby('Hour').agg({
        'Price_per_kWh': 'mean',
        'Station_Occupancy_Percent': 'mean',
        'Charger_Type': lambda x: (x == 'DC Fast').sum()
    }).rename(columns={'Charger_Type': 'FastCharger_Available'})
    
    best_time_hour = hourly_stats['Price_per_kWh'].idxmin()
    best_price = hourly_stats.loc[best_time_hour, 'Price_per_kWh']
    fastest_time_hour = hourly_stats['Station_Occupancy_Percent'].idxmin()
    
    cost_fast = energy_needed * best_price * 1.3
    cost_level2 = energy_needed * best_price * 0.9
    
    return {
        'energy_needed_kwh': energy_needed,
        'best_time_hour': best_time_hour,
        'cheapest_price': best_price,
        'fastest_available_hour': fastest_time_hour,
        'cost_fast_charging': cost_fast,
        'cost_level2_charging': cost_level2,
        'savings_level2': cost_fast - cost_level2,
        'fast_charge_health_impact': 0.5,
        'level2_health_impact': 0.1
    }

# ============== MAIN APPLICATION ==============
st.markdown('<h1 class="main-header">⚡ EV Charging Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown("AI-powered platform for charging optimization, battery health monitoring, and station intelligence")

# Load data
df = load_ev_charging_data()
df['Genre_Primary'] = df[['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Animation', 'Superhero']].fillna(0).idxmax(axis=1) if 'Action' in df.columns else 'General'

# ============== SIDEBAR NAVIGATION ==============
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Dashboard", "⚡ Station Occupancy", "🔋 Battery Health", "💰 Cost Optimizer", "📊 Analytics"])

# ============== PAGE 1: DASHBOARD ==============
if page == "🏠 Dashboard":
    st.markdown('<h2 class="section-header">📊 EV Charging Network Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Stations", df['Station_ID'].nunique(), "Active")
    with col2:
        st.metric("Avg Occupancy", f"{df['Station_Occupancy_Percent'].mean():.1f}%", f"±{df['Station_Occupancy_Percent'].std():.1f}%")
    with col3:
        st.metric("Avg Battery Health", f"{df['Battery_Health_Percent'].mean():.1f}%", "Network Avg")
    with col4:
        st.metric("Avg Charge Cost", f"₹{df['Charging_Cost'].mean():.2f}", "per session")
    with col5:
        st.metric("Total Sessions", len(df), "Recorded")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Station Occupancy Pattern")
        hourly_occupancy = df.groupby('Hour')['Station_Occupancy_Percent'].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hourly_occupancy.index, hourly_occupancy.values, marker='o', linewidth=3, markersize=8, color='#2ca02c')
        ax.fill_between(hourly_occupancy.index, hourly_occupancy.values, alpha=0.3, color='#2ca02c')
        ax.set_xlabel("Hour of Day", fontsize=11, fontweight='bold')
        ax.set_ylabel("Avg Occupancy (%)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Battery Health Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Battery_Health_Percent'], bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax.axvline(df['Battery_Health_Percent'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['Battery_Health_Percent'].mean():.1f}%")
        ax.set_xlabel("Battery Health (%)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Number of Vehicles", fontsize=11, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
    
    st.subheader("🏆 Top 5 Charging Stations by Usage")
    top_stations = df.groupby('Station_ID').agg({
        'User_ID': 'count',
        'Station_Occupancy_Percent': 'mean',
        'Charging_Cost': 'mean'
    }).rename(columns={'User_ID': 'Sessions', 'Station_Occupancy_Percent': 'Avg_Occupancy_%', 'Charging_Cost': 'Avg_Cost_₹'}).sort_values('Sessions', ascending=False).head(5)
    st.dataframe(top_stations.round(2), use_container_width=True)

# ============== PAGE 2: STATION OCCUPANCY ==============
elif page == "⚡ Station Occupancy":
    st.markdown('<h2 class="section-header">⚡ Charging Station Occupancy Prediction</h2>', unsafe_allow_html=True)
    
    occupancy_model, occupancy_accuracy, occupancy_importance = train_occupancy_model(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Predict Station Occupancy")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            predicted_hour = st.slider("Select Hour of Day", 0, 23, 12)
        with col_b:
            predicted_temp = st.slider("Temperature (°C)", -10, 40, 20)
        with col_c:
            predicted_humidity = st.slider("Humidity (%)", 20, 100, 60)
        
        col_d, col_e = st.columns(2)
        with col_d:
            predicted_day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            predicted_day_num = day_mapping[predicted_day]
        with col_e:
            predicted_month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            month_mapping = {month: i+1 for i, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])}
            predicted_month_num = month_mapping[predicted_month]
        
        if st.button("🔮 Predict Occupancy"):
            input_features = np.array([[predicted_hour, predicted_day_num, predicted_temp, predicted_humidity, predicted_month_num]])
            occupancy_pred = occupancy_model.predict(input_features)[0]
            occupancy_prob = occupancy_model.predict_proba(input_features)[0][1]
            
            st.success(f"**Prediction:** Station will be {'BUSY ✓' if occupancy_pred == 1 else 'FREE ✓'}")
            st.metric("Occupancy Probability", f"{occupancy_prob*100:.1f}%", "Confidence Level")
            
            if occupancy_prob > 0.7:
                st.warning("⚠️ **High occupancy expected!** Consider charging at a different time.")
            elif occupancy_prob < 0.3:
                st.info("✅ **Low occupancy expected!** Good time to charge.")
            else:
                st.info("ℹ️ **Moderate occupancy expected.** Plan accordingly.")
    
    with col2:
        st.subheader("📈 Model Performance")
        st.metric("Model Accuracy", f"{occupancy_accuracy*100:.1f}%", "On Test Data")
        
        st.subheader("🎯 Feature Importance")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(occupancy_importance['Feature'], occupancy_importance['Importance'], color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b'][:len(occupancy_importance)])
        ax.set_xlabel("Importance", fontweight='bold')
        st.pyplot(fig)

# ============== PAGE 3: BATTERY HEALTH ==============
elif page == "🔋 Battery Health":
    st.markdown('<h2 class="section-header">🔋 EV Battery Health Monitoring</h2>', unsafe_allow_html=True)
    
    battery_model, battery_rmse, battery_importance, y_test, y_pred = train_battery_health_model(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Predict Battery Health")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            vehicle_age = st.slider("Vehicle Age (months)", 0, 120, 36)
        with col_b:
            charge_cycles = st.slider("Charge Cycles Completed", 0, 5000, 1000)
        with col_c:
            avg_temp = st.slider("Avg Charging Temp (°C)", 10, 50, 25)
        
        col_d, col_e = st.columns(2)
        with col_d:
            fast_charges = st.slider("DC Fast Charges", 0, 500, 50)
        with col_e:
            depth_discharge = st.slider("Depth of Discharge (%)", 10, 100, 80)
        
        if st.button("🔮 Predict Battery Health"):
            input_features = np.array([[vehicle_age, charge_cycles, avg_temp, fast_charges, depth_discharge]])
            health_pred = battery_model.predict(input_features)[0]
            
            st.success(f"**Predicted Battery Health:** {health_pred:.1f}%")
            
            if health_pred >= 95:
                status = "🟢 Excellent - Like new"
            elif health_pred >= 85:
                status = "🟡 Good - Healthy battery"
            elif health_pred >= 80:
                status = "🟠 Fair - Monitor closely"
            else:
                status = "🔴 Poor - Consider replacement"
            
            st.markdown(f"**Status:** {status}")
            
            degradation_rate = (100 - health_pred) / (charge_cycles + 1)
            remaining_months = (health_pred - 80) / (degradation_rate + 0.001) * 12
            remaining_months = max(0, remaining_months)
            
            st.metric("Estimated Remaining Life", f"{remaining_months:.0f} months", "Until 80% health (EOL)")
            
            st.subheader("💡 Health Recommendations")
            recommendations = []
            if avg_temp > 35:
                recommendations.append("🌡️ Reduce charging temperature - Use cooler environments")
            if fast_charges > 100:
                recommendations.append("⚡ Reduce DC fast charging - Use Level 2 more often")
            if depth_discharge > 90:
                recommendations.append("🔋 Avoid deep discharges - Charge when 20% remaining")
            if charge_cycles > 3000:
                recommendations.append("⏳ High cycle count - Consider replacement soon")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ All health factors are optimal!")
    
    with col2:
        st.subheader("📈 Model Performance")
        st.metric("Prediction RMSE", f"{battery_rmse:.2f}%", "Average Error")
        
        st.subheader("🎯 Feature Importance")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(battery_importance['Feature'], battery_importance['Importance'], color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'][:len(battery_importance)])
        ax.set_xlabel("Importance", fontweight='bold')
        st.pyplot(fig)

# ============== PAGE 4: COST OPTIMIZER ==============
elif page == "💰 Cost Optimizer":
    st.markdown('<h2 class="section-header">💰 Smart Charging Cost Optimizer</h2>', unsafe_allow_html=True)
    
    st.info("⚡ Get personalized recommendations to save money on charging while protecting your battery")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔋 Your Vehicle Information")
        battery_capacity = st.slider("Battery Capacity (kWh)", 30, 100, 60)
        current_battery = st.slider("Current Battery Level (%)", 5, 95, 30)
        target_battery = st.slider("Target Charge Level (%)", current_battery + 5, 100, 80)
    
    with col2:
        st.subheader("🎯 Optimization Strategy")
        strategy_type = st.radio("Select Priority", ["💰 Save Money (Cheapest)", "⚡ Save Time (Fastest)", "🔋 Protect Battery (Healthiest)"])
    
    if st.button("📊 Optimize My Charging"):
        strategy = get_optimal_charging_strategy(df, current_battery, battery_capacity, target_battery)
        
        st.success("✅ Optimization Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Energy Needed", f"{strategy['energy_needed_kwh']:.1f} kWh", "To reach target")
        with col2:
            st.metric("Best Time to Charge", f"{strategy['best_time_hour']:02d}:00", f"₹{strategy['cheapest_price']:.2f}/kWh")
        with col3:
            st.metric("Cost Savings (Level 2)", f"₹{strategy['savings_level2']:.2f}", f"vs Fast Charging")
        
        st.subheader("📋 Recommended Strategy")
        if strategy_type == "💰 Save Money (Cheapest)":
            st.info(f"""**Cost-Optimized Charging Strategy:**
✅ **Recommended:** Level 2 Charging at {strategy['best_time_hour']:02d}:00
💰 **Estimated Cost:** ₹{strategy['cost_level2']:.2f}
⏱️ **Time Required:** 4-6 hours (Level 2)
🔋 **Battery Impact:** Minimal degradation ({strategy['level2_health_impact']:.1f}%)""")
        elif strategy_type == "⚡ Save Time (Fastest)":
            st.info(f"""**Time-Optimized Charging Strategy:**
✅ **Recommended:** DC Fast Charging at {strategy['fastest_available_hour']:02d}:00
⏱️ **Estimated Time:** 30-45 minutes
💰 **Cost:** ₹{strategy['cost_fast_charging']:.2f}
⚠️ **Battery Impact:** Higher degradation ({strategy['fast_charge_health_impact']:.1f}% per session)""")
        else:
            st.info(f"""**Battery-Health-Optimized Strategy:**
✅ **Recommended:** Level 2 Charging at {strategy['best_time_hour']:02d}:00
🔋 **Battery Protection:** Excellent - Degradation Rate: {strategy['level2_health_impact']:.1f}% per session
💰 **Cost:** ₹{strategy['cost_level2']:.2f}
✨ **Result:** Extended battery lifespan by months""")

# ============== PAGE 5: ANALYTICS ==============
elif page == "📊 Analytics":
    st.markdown('<h2 class="section-header">📊 Network Analytics & Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Charger Type Revenue Analysis")
        charger_revenue = df.groupby('Charger_Type').agg({'Charging_Cost': ['sum', 'mean', 'count']}).round(2)
        charger_revenue.columns = ['Total_Revenue', 'Avg_Cost', 'Sessions']
        
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(charger_revenue.index, charger_revenue['Total_Revenue'], color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel("Total Revenue (₹)", fontweight='bold')
        ax.set_title("Revenue by Charger Type", fontweight='bold')
        st.pyplot(fig)
        st.dataframe(charger_revenue, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Vehicle Models by Sessions")
        top_vehicles = df['Vehicle_Model'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(range(len(top_vehicles)), top_vehicles.values, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_vehicles)))
        ax.set_yticklabels(top_vehicles.index)
        ax.set_xlabel("Number of Sessions", fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
    
    st.subheader("📈 Monthly Trends")
    monthly_stats = df.groupby('Month').agg({'User_ID': 'count', 'Charging_Cost': 'sum', 'Battery_Health_Percent': 'mean'}).rename(columns={'User_ID': 'Sessions', 'Charging_Cost': 'Revenue', 'Battery_Health_Percent': 'Avg_Health'})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_stats.index, monthly_stats['Sessions'], marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
        ax.fill_between(monthly_stats.index, monthly_stats['Sessions'], alpha=0.3, color='#2ca02c')
        ax.set_xlabel("Month", fontweight='bold')
        ax.set_ylabel("Sessions", fontweight='bold')
        ax.set_title("Monthly Sessions", fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_stats.index, monthly_stats['Revenue'], marker='s', linewidth=2.5, markersize=8, color='#ff7f0e')
        ax.fill_between(monthly_stats.index, monthly_stats['Revenue'], alpha=0.3, color='#ff7f0e')
        ax.set_xlabel("Month", fontweight='bold')
        ax.set_ylabel("Revenue (₹)", fontweight='bold')
        ax.set_title("Monthly Revenue", fontweight='bold')
        st.pyplot(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_stats.index, monthly_stats['Avg_Health'], marker='^', linewidth=2.5, markersize=8, color='#d62728')
        ax.fill_between(monthly_stats.index, monthly_stats['Avg_Health'], alpha=0.3, color='#d62728')
        ax.axhline(80, color='red', linestyle='--', linewidth=1.5, label='EOL Threshold')
        ax.set_xlabel("Month", fontweight='bold')
        ax.set_ylabel("Battery Health (%)", fontweight='bold')
        ax.set_title("Avg Battery Health", fontweight='bold')
        ax.legend()
        st.pyplot(fig)