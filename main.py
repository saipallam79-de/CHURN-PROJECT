import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# -----------------------------
# 🎨 UI STYLE
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown("""
<style>

/* 🌌 Background */
.stApp {
    background: linear-gradient(-45deg, #141e30, #243b55, #1d2671, #c33764);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}

/* Animation */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Center container */
.center-box {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 80vh;
}

/* Title */
.title {
    color: white;
    font-size: 48px;
    margin-bottom: 40px;
}

/* Buttons */
div.stButton > button {
    width: 300px;
    height: 70px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
    color: white;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}

/* Hover */
div.stButton > button:hover {
    transform: scale(1.08);
    background: linear-gradient(135deg, #ff512f, #dd2476);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
df = pd.read_csv("telecom_churn.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------------
# NAVIGATION FUNCTIONS
# -----------------------------
def go_home():
    st.session_state.page = "home"

def go_single():
    st.session_state.page = "single"

def go_csv():
    st.session_state.page = "csv"

if "page" not in st.session_state:
    st.session_state.page = "home"

# =============================
# 🏠 HOME PAGE
# =============================
if st.session_state.page == "home":

    st.markdown('<div class="center-box">', unsafe_allow_html=True)

    st.markdown('<div class="title">🚀 Customer Churn Prediction</div>', unsafe_allow_html=True)

    st.button("🧍 Single Customer", on_click=go_single)
    st.write("")  # spacing
    st.button("📂 Upload CSV", on_click=go_csv)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🧍 SINGLE CUSTOMER
# =============================
elif st.session_state.page == "single":

    st.button("⬅ Back", on_click=go_home)

    st.header("Single Customer Prediction")

    AccountWeeks = st.number_input("AccountWeeks")
    ContractRenewal = st.number_input("ContractRenewal (0/1)")
    DataPlan = st.number_input("DataPlan (0/1)")
    DataUsage = st.number_input("DataUsage")
    CustServCalls = st.number_input("CustServCalls")
    DayMins = st.number_input("DayMins")
    DayCalls = st.number_input("DayCalls")
    MonthlyCharge = st.number_input("MonthlyCharge")
    OverageFee = st.number_input("OverageFee")
    RoamMins = st.number_input("RoamMins")

    if st.button("Predict"):

        data = [[
            AccountWeeks, ContractRenewal, DataPlan, DataUsage,
            CustServCalls, DayMins, DayCalls,
            MonthlyCharge, OverageFee, RoamMins
        ]]

        pred = model.predict(data)

        if pred[0] == 1:
            st.error("⚠️ Customer will LEAVE")
        else:
            st.success("✅ Customer will STAY")

# =============================
# 📂 CSV UPLOAD
# =============================
elif st.session_state.page == "csv":

    st.button("⬅ Back", on_click=go_home)

    st.header("Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    required_columns = [
        'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage',
        'CustServCalls', 'DayMins', 'DayCalls',
        'MonthlyCharge', 'OverageFee', 'RoamMins'
    ]

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.dataframe(df_upload.head())

        missing = [col for col in required_columns if col not in df_upload.columns]

        if len(missing) == 0:

            data = df_upload[required_columns]
            predictions = model.predict(data)

            df_upload['Prediction'] = pd.Series(predictions).map({
                0: "STAY",
                1: "LEAVE"
            })

            st.success("Prediction completed")

            st.dataframe(df_upload)

            # Charts
            st.subheader("Churn Analysis")

            counts = df_upload['Prediction'].value_counts()

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

            st.bar_chart(counts)

        else:
            st.error("Invalid CSV format")
            st.write("Missing columns:", missing)
