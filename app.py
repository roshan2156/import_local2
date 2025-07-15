import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(page_title="Smart Procurement Cost Predictor", layout="wide")
st.title("🔍 Smart Procurement: Cost Forecasting for Local vs Imported Components")

# Expected features
expected_features = [
    'Component_ID', 'Component_Name', 'Category', 'Source_Type',
    'Base_Cost_USD', 'Quantity', 'Lead_Time_Days',
    'Transport_Cost_USD', 'Customs_Duty_Percent',
    'Currency_Rate', 'Total_Cost_USD'
]

if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded CSV Columns")
    st.write(user_df.columns.tolist())

    st.info("Please map your CSV columns to the expected features below:")

    column_mapping = {}
    for feature in expected_features:
        column_mapping[feature] = st.selectbox(f"Select column for '{feature}'", options=[None] + list(user_df.columns), index=0)

    mandatory_features = ['Component_Name', 'Category', 'Source_Type', 'Base_Cost_USD', 'Quantity', 'Total_Cost_USD']
    missing_mandatory = [feat for feat in mandatory_features if column_mapping.get(feat) in (None, '')]

    if missing_mandatory:
        st.error(f"❌ Please map all mandatory features: {', '.join(mandatory_features)}")
        st.stop()

    df = pd.DataFrame()
    for feature in expected_features:
        mapped_col = column_mapping.get(feature)
        if mapped_col in (None, ''):
            if feature == 'Lead_Time_Days': df[feature] = 5
            elif feature == 'Transport_Cost_USD': df[feature] = 0.0
            elif feature == 'Customs_Duty_Percent': df[feature] = 0.0
            elif feature == 'Currency_Rate': df[feature] = 1.0
            else: df[feature] = np.nan
        else:
            df[feature] = user_df[mapped_col]

    numeric_fields = ['Base_Cost_USD', 'Quantity', 'Lead_Time_Days', 'Transport_Cost_USD', 'Customs_Duty_Percent', 'Currency_Rate', 'Total_Cost_USD']
    error_fields = []
    for field in numeric_fields:
        try:
            df[field] = pd.to_numeric(df[field])
        except:
            error_fields.append(field)

    if error_fields:
        st.error(f"❌ Invalid data in: {', '.join(error_fields)}")
        st.stop()

    st.success("✅ File loaded and columns mapped successfully.")

else:
    df = pd.read_csv("cost_forecasting_data_100.csv")
    st.info("Using default sample dataset.")

with open("cost_forecasting_data_100.csv", "rb") as f:
    st.download_button("📥 Download Sample CSV Template", f, "sample_cost_data.csv", "text/csv")

st.subheader("📖 Dataset Preview")
st.dataframe(df.head())

# Label Encoding
le_name, le_cat, le_src = LabelEncoder(), LabelEncoder(), LabelEncoder()
try:
    df['Component_Name'] = le_name.fit_transform(df['Component_Name'].astype(str))
    df['Category'] = le_cat.fit_transform(df['Category'].astype(str))
    df['Source_Type'] = le_src.fit_transform(df['Source_Type'].astype(str))
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Prepare data
X = df.drop(['Component_ID', 'Total_Cost_USD'], axis=1, errors='ignore')
y = df['Total_Cost_USD']
X = X.fillna(0)
y = y.fillna(y.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(random_state=42)
lr_model = LinearRegression()
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Evaluation
rf_r2 = r2_score(y_test, rf_pred)
lr_r2 = r2_score(y_test, lr_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_cv = cross_val_score(rf_model, X, y, cv=5).mean()
lr_cv = cross_val_score(lr_model, X, y, cv=5).mean()

# Model selection
model_choice = st.selectbox("🔧 Choose Regression Model for Prediction", ["Random Forest", "Linear Regression"])
reg_model = rf_model if model_choice == "Random Forest" else lr_model
y_pred = rf_pred if model_choice == "Random Forest" else lr_pred

# Comparison
st.subheader("📊 Regression Models Comparison")
col1, col2 = st.columns(2)
with col1:
    st.header("Random Forest")
    st.write(f"R² Score: {rf_r2:.4f}")
    st.write(f"RMSE: ${rf_rmse:.2f}")
    st.write(f"CV Mean Score: {rf_cv:.4f}")
with col2:
    st.header("Linear Regression")
    st.write(f"R² Score: {lr_r2:.4f}")
    st.write(f"RMSE: ${lr_rmse:.2f}")
    st.write(f"CV Mean Score: {lr_cv:.4f}")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Cost")
ax1.set_ylabel("Predicted Cost")
ax1.set_title(f"Actual vs Predicted Total Cost ({model_choice})")
st.pyplot(fig1)

# Feature Importance
st.subheader("📈 Feature Importance / Coefficients")
importance = (
    pd.Series(reg_model.feature_importances_, index=X.columns)
    if model_choice == "Random Forest"
    else pd.Series(reg_model.coef_, index=X.columns)
)
st.bar_chart(importance.sort_values())

# Classification
df['Cost_per_unit'] = df['Total_Cost_USD'] / df['Quantity'].replace(0, 1)
mean_cost_unit = df['Cost_per_unit'].mean()
df['Import_Better'] = ((df['Source_Type'] == 1) & (df['Cost_per_unit'] < mean_cost_unit)).astype(int)

Xc = df.drop(['Component_ID', 'Total_Cost_USD', 'Import_Better'], axis=1, errors='ignore')
yc = df['Import_Better']
clf = RandomForestClassifier(random_state=42).fit(Xc.fillna(0), yc)
acc = accuracy_score(yc, clf.predict(Xc.fillna(0)))

st.subheader("🧠 Classification: Is Import More Cost-Effective?")
st.write(f"**Accuracy:** {acc:.2%}")
st.text(classification_report(yc, clf.predict(Xc.fillna(0))))

fig2, ax2 = plt.subplots()
sns.boxplot(x='Source_Type', y='Cost_per_unit', data=df, ax=ax2)
ax2.set_xticklabels(['Local', 'Import'])
ax2.set_title("Cost Per Unit: Local vs Import")
st.pyplot(fig2)

# User Prediction
st.subheader("📥 Predict Your Own Component Cost")
col1, col2, col3 = st.columns(3)
with col1:
    component_name = st.selectbox("Component Name", le_name.classes_)
    category = st.selectbox("Category", le_cat.classes_)
with col2:
    source_type = st.selectbox("Source Type", le_src.classes_)
    quantity = st.number_input("Quantity", min_value=1, value=100)
with col3:
    base_cost = st.number_input("Base Cost (USD)", min_value=0.0)
    transport_cost = st.number_input("Transport Cost (USD)", min_value=0.0)
lead_time = st.slider("Lead Time (Days)", 1, 30, 5)
customs_duty = st.slider("Customs Duty (%)", 0, 30, 5)
currency_rate = st.number_input("Currency Rate", min_value=1.0, max_value=2.5, value=1.0)

if st.button("📊 Predict Total Cost"):

    def predict_cost(src_label):
        input_data = pd.DataFrame([[ 
            le_name.transform([component_name])[0],
            le_cat.transform([category])[0],
            le_src.transform([src_label])[0],
            base_cost, quantity, lead_time, transport_cost, customs_duty, currency_rate
        ]], columns=X.columns)
        return reg_model.predict(input_data)[0]

    predicted_import = predict_cost("Import")
    predicted_local = predict_cost("Local")
    cpu_import = predicted_import / quantity
    cpu_local = predicted_local / quantity
    selected_cost = predict_cost(source_type)
    selected_cpu = selected_cost / quantity

    recommended = "Import" if cpu_import < cpu_local else "Local"
    result_text = f"✅ {recommended} is more cost-effective."

    explanation = f"""🔍 **Cost per Unit**:
- Import: ${cpu_import:.2f}
- Local: ${cpu_local:.2f}

📌 **Recommendation**: {recommended} sourcing is more cost-effective."""

    if customs_duty > 0:
        explanation += f"\n- Customs Duty: {customs_duty}%"
    if transport_cost > 0:
        explanation += f"\n- Transport Cost: ${transport_cost:.2f}"
    explanation += f"\n- Lead Time Considered: {lead_time} days"

    st.success(f"💰 Predicted Total Cost (Selected - {source_type}): ${selected_cost:.2f}")
    st.info(f"📦 Cost per Unit (Selected): ${selected_cpu:.2f}")
    st.warning(f"📍 Recommendation: {result_text}")
    st.info(f"📝 Explanation:\n{explanation}")

    st.session_state.prediction_log.append({
        'Component_Name': component_name,
        'Category': category,
        'Source_Type_Selected': source_type,
        'Total_Cost_Selected': selected_cost,
        'Cost_per_unit_Import': cpu_import,
        'Cost_per_unit_Local': cpu_local,
        'Recommended_Source': recommended,
        'Explanation': explanation
    })

    fig3, ax3 = plt.subplots()
    ax3.bar(["Import", "Local"], [cpu_import, cpu_local], color=["green", "blue"])
    ax3.set_ylabel("Cost per Unit")
    ax3.set_title("Import vs Local Comparison")
    st.pyplot(fig3)

# Logs
if st.session_state.prediction_log:
    log_df = pd.DataFrame(st.session_state.prediction_log)
    st.subheader("📜 Prediction Log")
    st.dataframe(log_df)
    csv = log_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Prediction Log", csv, "prediction_log.csv", "text/csv")

st.download_button("⬇️ Download Processed CSV", df.to_csv(index=False).encode(), "processed_dataset.csv", "text/csv")

# ✅ Feedback Section
st.markdown("---")
st.subheader("💬 We Value Your Feedback!")

if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []

with st.form("feedback_form", clear_on_submit=True):
    name = st.text_input("👤 Your Name (Optional)")
    rating = st.slider("⭐ Rate the Model", 1, 5, 4)
    feedback = st.text_area("📝 Write your feedback here...")
    submit = st.form_submit_button("📨 Submit")

    if submit:
        st.session_state.feedback_log.append({
            'Name': name,
            'Rating': rating,
            'Feedback': feedback
        })
        st.success("✅ Thank you! Your feedback has been recorded.")

if st.session_state.feedback_log:
    st.markdown("### 🗂️ Recent Feedback")
    for entry in reversed(st.session_state.feedback_log[-5:]):
        st.markdown(f"**👤 {entry['Name'] or 'Anonymous'}** — ⭐ {entry['Rating']}/5")
        st.markdown(f"💬 {entry['Feedback']}")
        st.markdown("---")
    
    feedback_df = pd.DataFrame(st.session_state.feedback_log)
    feedback_csv = feedback_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Feedback Log", feedback_csv, "user_feedback.csv", "text/csv")