import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧠",
    layout="centered"
)
st.markdown("""
<style>
body {
    background-color: #0d0d0d;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #111 0%, #000 60%);
}
.glass {
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(0,255,180,0.25);
    box-shadow: 0 0 25px rgba(0,255,180,0.15);
    margin-bottom: 25px;
}
h1, h2, h3 {
    color: #00ffcc;
}
label, p, span {
    color: #d6fff6 !important;
}
header {
    visibility: hidden;
    height: 0px;
}
</style>
""", unsafe_allow_html=True)

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

centers = scaler.inverse_transform(model.cluster_centers_)
cluster_names = {}

for i, (income_c, spending_c) in enumerate(centers):
    if income_c > 70 and spending_c > 60:
        label = "High Income • High Spending (Target)"
    elif income_c > 70 and spending_c <= 60:
        label = "High Income • Low Spending"
    elif income_c <= 70 and spending_c > 60:
        label = "Low Income • High Spending"
    else:
        label = "Low Income • Low Spending"

    cluster_names[i] = label


st.title("🧠 Customer Segmentation")
st.caption("K-Means clustering on mall customer data")

income = st.number_input(
    "💰 Annual Income (k$)",
    min_value=0,
    max_value=150,
    value=50,
    step=5
)

spending = st.number_input(
    "🛍️ Spending Score (1–100)",
    min_value=1,
    max_value=100,
    value=50,
    step=5
)

input_data = np.array([[income, spending]])
scaled_input = scaler.transform(input_data)
cluster = model.predict(scaled_input)[0]

st.markdown(f"""
### 🎯 Predicted Segment
**<span style='color:#00ffcc;font-size:24px'>{cluster_names[cluster]}</span>**
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.subheader("📊 Cluster Visualization")

df = pd.read_csv("Mall_Customers.csv")
x = df[['Annual Income (k$)', 'Spending Score (1-100)']]
df['Cluster'] = model.predict(scaler.transform(x))

fig, ax = plt.subplots(figsize=(6, 5))

for i in range(model.n_clusters):
    ax.scatter(
        df[df['Cluster'] == i]['Annual Income (k$)'],
        df[df['Cluster'] == i]['Spending Score (1-100)'],
        label=cluster_names[i],
        alpha=0.6
    )

ax.scatter(
    income,
    spending,
    s=180,
    c='#00ffcc',
    edgecolors='black',
    marker='X',
    label='Your Input'
)

ax.set_facecolor("#000000")
fig.patch.set_facecolor("#000000")

ax.set_xlabel("Annual Income (k$)", color="#00ffcc")
ax.set_ylabel("Spending Score (1-100)", color="#00ffcc")
ax.tick_params(colors="#d6fff6")

legend = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    facecolor="#000000",
    edgecolor="#00ffcc"
)

for text in legend.get_texts():
    text.set_color("#d6fff6")

st.pyplot(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
