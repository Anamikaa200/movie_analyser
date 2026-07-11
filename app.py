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
st.sidebar.header("Dataset Information")

st.sidebar.write(f"Total Reviews: **{len(df)}**")

# Show class distribution only if column exists
if "Class" in df.columns:
    positive = (df["Class"] == "POSITIVE").sum()
    negative = (df["Class"] == "NEGATIVE").sum()

    st.sidebar.success(f"Positive Reviews: {positive}")
    st.sidebar.error(f"Negative Reviews: {negative}")
else:
    st.sidebar.warning("Column 'Class' not found.")

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
