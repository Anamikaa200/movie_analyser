import streamlit as st
import pandas as pd
from transformers import pipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Netflix Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load Sentiment Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    try:
        # Try comma-separated CSV
        df = pd.read_csv("netflix movie dhurandhar 2.csv")
    except Exception:
        try:
            # Try semicolon-separated CSV
            df = pd.read_csv("netflix movie dhurandhar 2.csv", sep=";")
        except Exception as e:
            st.error(f"Unable to load dataset: {e}")
            st.stop()

    return df

df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("🎬 Netflix Movie Review Analyzer")
st.write(
    "Analyze movie reviews using a Hugging Face Transformer model."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg",
    width=80
)

st.sidebar.title("📊 Sentiment Dashboard")
st.sidebar.markdown("---")

if "Class" in df.columns:

    positive = (df["Class"] == "POSITIVE").sum()
    negative = (df["Class"] == "NEGATIVE").sum()

    total = positive + negative

    pos_percent = (positive / total) * 100
    neg_percent = (negative / total) * 100

    st.sidebar.subheader("😊 Positive Reviews")
    st.sidebar.progress(pos_percent / 100)
    st.sidebar.success(f"{pos_percent:.1f}%")

    st.sidebar.subheader("😞 Negative Reviews")
    st.sidebar.progress(neg_percent / 100)
    st.sidebar.error(f"{neg_percent:.1f}%")

    st.sidebar.markdown("---")

    st.sidebar.subheader("📈 Sentiment Distribution")

    chart = pd.DataFrame(
        {
            "Percentage": [pos_percent, neg_percent]
        },
        index=["Positive", "Negative"]
    )

    st.sidebar.bar_chart(chart)

else:
    st.sidebar.warning("Column 'Class' not found.")

st.sidebar.markdown("---")

st.sidebar.info("""
### 🤖 Features

✔ AI Sentiment Analysis

✔ Live Prediction

✔ Dataset Visualization

✔ Interactive Dashboard

✔ Hugging Face Transformer
""")

st.sidebar.markdown("---")
st.sidebar.success("Made with ❤️ using Streamlit")
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Review Analyzer", "Dataset"])

# -----------------------------
# Review Analyzer
# -----------------------------
with tab1:

    st.subheader("Enter a Movie Review")

    review = st.text_area(
        "Review",
        height=180,
        placeholder="Type your review here..."
    )

    if st.button("Analyze Review"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            with st.spinner("Analyzing..."):

                result = classifier(review)[0]

            label = result["label"]
            confidence = result["score"]

            if label.upper() == "POSITIVE":
                st.success("😊 Positive Review")
            else:
                st.error("😞 Negative Review")

            st.metric(
                label="Confidence",
                value=f"{confidence*100:.2f}%"
            )

# -----------------------------
# Dataset
# -----------------------------
with tab2:

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)

    if "Class" in df.columns:

        st.subheader("Sentiment Distribution")

        chart = df["Class"].value_counts()

        st.bar_chart(chart)

    else:
        st.info("No 'Class' column found in dataset.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hugging Face Transformers")
