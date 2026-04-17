import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import boto3

# -----------------------------
# 🎥 VIDEO BACKGROUND + UI STYLE
# -----------------------------
st.markdown("""
<video autoplay muted loop id="bg-video">
  <source src="https://www.w3schools.com/howto/rain.mp4" type="video/mp4">
</video>

<style>
#bg-video {
  position: fixed;
  right: 0;
  bottom: 0;
  min-width: 100%;
  min-height: 100%;
  z-index: -1;
}

.stApp {
    color: white;
}

div.stButton > button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    font-size: 20px;
    padding: 15px 30px;
    border-radius: 15px;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.1);
    background: linear-gradient(45deg, #ff512f, #dd2476);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA + TRAIN MODEL
# -----------------------------
df = pd.read_csv("telecom_churn.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------------
# SESSION NAVIGATION
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# =============================
# 🏠 HOME PAGE
# =============================
if st.session_state.page == "home":

    st.title("🚀 AI Customer Churn System")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧍 Single Customer"):
            st.session_state.page = "single"

    with col2:
        if st.button("📂 CSV Upload"):
            st.session_state.page = "csv"

# =============================
# 🧍 SINGLE CUSTOMER PAGE
# =============================
elif st.session_state.page == "single":

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    st.header("🧍 Single Customer Prediction")

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
            result = "LEAVE"
            st.error("⚠️ Customer will LEAVE")
        else:
            result = "STAY"
            st.success("✅ Customer will STAY")

        # Save to S3
        try:
            s3 = boto3.client('s3')
            s3.put_object(
                Bucket='sai-first-bucket-76800',
                Key='results/single_output.txt',
                Body=str(data) + " -> " + result
            )
            st.info("Saved to AWS S3 ✅")
        except Exception as e:
            st.warning(f"S3 Error: {e}")

# =============================
# 📂 CSV UPLOAD PAGE
# =============================
elif st.session_state.page == "csv":

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    st.header("📂 Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    required_columns = [
        'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage',
        'CustServCalls', 'DayMins', 'DayCalls',
        'MonthlyCharge', 'OverageFee', 'RoamMins'
    ]

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.dataframe(df_upload.head())

        if all(col in df_upload.columns for col in required_columns):

            data = df_upload[required_columns]
            predictions = model.predict(data)

            df_upload['Prediction'] = pd.Series(predictions).map({
                0: "STAY",
                1: "LEAVE"
            })

            st.success("✅ Prediction completed")

            # Show all
            st.subheader("📊 Results")
            st.dataframe(df_upload)

            # Separate
            stay_df = df_upload[df_upload['Prediction'] == "STAY"]
            leave_df = df_upload[df_upload['Prediction'] == "LEAVE"]

            st.subheader("🟢 Staying Customers")
            st.success(f"{len(stay_df)} Customers")
            st.dataframe(stay_df)

            st.subheader("🔴 Leaving Customers")
            st.error(f"{len(leave_df)} Customers")
            st.dataframe(leave_df)

            # Download
            csv = df_upload.to_csv(index=False)
            st.download_button("Download Results", csv, "output.csv")

            # Save to S3
            try:
                s3 = boto3.client('s3')
                s3.put_object(
                    Bucket='sai-first-bucket-76800',
                    Key='results/bulk_output.csv',
                    Body=csv
                )
                st.info("Saved to AWS S3 ✅")
            except Exception as e:
                st.warning(f"S3 Error: {e}")

        else:
            st.error("❌ Invalid CSV format")
            st.write("Required columns:", required_columns)