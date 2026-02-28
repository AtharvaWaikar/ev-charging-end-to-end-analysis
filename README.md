# ⚡ EV Charging End-to-End Analysis

---

## 📌 Project Overview

This is a **complete end-to-end Data Science project** on **EV charging stations**.  

The project workflow includes:  

1. **Raw & Unclean Dataset** – original data collected from charging stations  
2. **Data Cleaning & Processing** – generating a clean, structured dataset  
3. **Exploratory Data Analysis (EDA)** – understanding trends, patterns, and anomalies  
4. **Interactive Streamlit App** – enabling users to explore charging station usage, trends, and metrics dynamically  

This project simulates a **real-world Data Science workflow**, demonstrating skills in data cleaning, analysis, visualization, and interactive app development.

---

## 🏷 Project Badges

![Python](https://img.shields.io/badge/Language-Python-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Library-Streamlit-red?style=flat-square)
![Pandas](https://img.shields.io/badge/Library-Pandas-green?style=flat-square)
![Data](https://img.shields.io/badge/Data-CSV-yellow?style=flat-square)
![End-to-End](https://img.shields.io/badge/Workflow-End--to--End-orange?style=flat-square)

---

## 📂 Dataset Description

### Raw Dataset
- Original, unclean data with missing values and inconsistencies  
- Columns include:  
  - `Station_ID` → Unique ID for each charging station  
  - `Location` → City or area  
  - `Date` → Transaction date  
  - `Time` → Transaction time  
  - `Energy_Consumed` → kWh per session  
  - `Cost` → Cost of charging session  
  - `User_ID` → Identifier for the user  

### Clean Dataset
- Cleaned and processed dataset ready for analysis  
- Missing values handled, data types corrected, and unnecessary columns removed  

---

## 🧹 Data Cleaning & Processing

- Removed duplicates and inconsistencies  
- Handled missing or null values  
- Standardized column names and formats  
- Generated additional features for analysis (e.g., daily usage, peak hours)  
- Created a clean dataset for use in analysis and the Streamlit app  

---

## 📈 Key Analysis & Features

- **Usage Trends:** Peak hours, busiest stations, energy consumption patterns  
- **Cost Analysis:** Average cost per session, revenue by location  
- **User Insights:** Frequent users, session durations, energy patterns  
- **Interactive Exploration:** Filters for location, date, station, and energy consumption in Streamlit  

---

## 📊 Streamlit App Features

- Interactive dashboards with dynamic charts  
- Real-time filtering by:
  - Location  
  - Date range  
  - Station ID  
  - Energy consumed  

<p align="center">
  <img src="outputs/ev-charging-dashboard.png" width="900"/>
</p>

- Displays KPIs such as:  
  - Total energy consumed  
  - Total revenue  
  - Number of charging sessions  
  - Average cost per session  

---

## 🛠 Tools & Technologies Used

- **Python:** Core programming and data manipulation  
- **Pandas & Numpy:** Data cleaning, aggregation, and feature engineering  
- **Matplotlib & Seaborn:** Visualizations and EDA  
- **Streamlit:** Interactive web-based dashboard for user exploration  
- **Scikit-learn (Optional):** Any modeling if applied  

---

## 🔍 Key Insights

- Identified peak usage hours and busiest stations  
- Analyzed energy consumption trends across locations  
- Users with higher frequency contribute disproportionately to revenue  
- Cleaned dataset enables better, faster analysis and model building  

---

## 📖 Guide Document

Refer to `guide.txt` for:

- How to run the Streamlit app  
- Dataset explanations  
- Step-by-step instructions for using filters and exploring dashboards  

---

## 🧩 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/ev-charging-end-to-end-analysis.git
```

2. Navigate to the project folder:

```bash
cd ev-charging-end-to-end-analysis
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Atharva Waikar** - Data Scientist | Python | Streamlit | End-to-End Analytics  

---

⭐ If you find this project useful or interesting, please give it a star!
