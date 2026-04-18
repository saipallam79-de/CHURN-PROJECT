import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import boto3
import matplotlib.pyplot as plt

# -----------------------------
# 🎥 PREMIUM UI (VIDEO + GLASS)
# -----------------------------
st.markdown("""
<style>

/* VIDEO BACKGROUND */
#bg-video {
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%;
    min-height: 100%;
    z-index: -1;
    object-fit: cover;
}

/* DARK OVERLAY */
.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.6);
    z-index: -1;
}

/* GLASS CARD */
.card {
    backdrop-filter: blur(15px);
    background: rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 50px;
}

/* BUTTON STYLE */
div.stButton > button {
    width: 100%;
    height: 80px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 20px;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    transition: all 0.3s ease;
}

/* HOVER */
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #ff512f, #dd2476);
}

/* TITLE */
h1, h2, h3 {
    text-align: center;
    color: white;
}

</style>

<video autoplay muted loop id="bg-video">
  <source src="https://cdn.coverr.co/videos/coverr-clouds-over-mountains-1576/1080p.mp4" type="video/mp4">
</video>

<div class="overlay"></div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
df = pd.read_csv("telecom_churn.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------------
# NAVIGATION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# =============================
# 🏠 HOME PAGE
# =============================
if st.session_state.page == "home":

    st.markdown("<h1>🚀 AI Customer Churn System</h1>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧍 Single Customer"):
            st.session_state.page = "single"

    with col2:
        if st.button("📂 Upload Dataset"):
            st.session_state.page = "csv"

    st.markdown('</div>', unsafe_allow_html=True)

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
        except:
            pass

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

        missing = [col for col in required_columns if col not in df_upload.columns]

        if len(missing) == 0:

            data = df_upload[required_columns]
            predictions = model.predict(data)

            df_upload['Prediction'] = pd.Series(predictions).map({
                0: "STAY",
                1: "LEAVE"
            })

            st.success("✅ Prediction completed")

            st.subheader("📊 Results")
            st.dataframe(df_upload)

            stay_df = df_upload[df_upload['Prediction'] == "STAY"]
            leave_df = df_upload[df_upload['Prediction'] == "LEAVE"]

            st.subheader("🟢 Staying Customers")
            st.success(f"{len(stay_df)} Customers")
            st.dataframe(stay_df)

            st.subheader("🔴 Leaving Customers")
            st.error(f"{len(leave_df)} Customers")
            st.dataframe(leave_df)

            # PIE CHART
            st.subheader("📊 Churn Analysis")
            counts = df_upload['Prediction'].value_counts()

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

            # BAR CHART
            st.subheader("📈 Prediction Count")
            st.bar_chart(counts)

            # DOWNLOAD
            csv = df_upload.to_csv(index=False)
            st.download_button("Download Results", csv, "output.csv")

            # AWS SAVE
            try:
                s3 = boto3.client('s3')
                s3.put_object(
                    Bucket='sai-first-bucket-76800',
                    Key='results/bulk_output.csv',
                    Body=csv
                )
            except:
                pass

        else:
            st.error("❌ Invalid CSV format")
            st.write("Missing columns:", missing)
