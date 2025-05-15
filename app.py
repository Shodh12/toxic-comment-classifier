import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('./toxic_comment_bert', device_map='cpu')
    tokenizer = BertTokenizer.from_pretrained('./toxic_comment_bert')
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Title and intro
st.title("🛡️ Toxic Comment Classifier")
st.write("Detect toxic language using a fine-tuned BERT model.")

# Example selector
st.markdown("Try an example:")
example = st.selectbox("", [
    "You're amazing!",
    "I hate you.",
    "This is fine.",
    "Keep up the good work.",
    "Shut your mouth."
])
if st.button("Use Example"):
    st.session_state["user_input"] = example

# Text area
user_input = st.text_area("Your comment", value=st.session_state.get("user_input", ""))

# Single classification
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        inputs = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][prediction].item() * 100

        label_map = {0: "Non-Toxic ✅", 1: "Toxic ⚠️"}
        if prediction == 0:
            st.success(f"Prediction: {label_map[prediction]}")
        else:
            st.error(f"Prediction: {label_map[prediction]}")
        st.write(f"Confidence: **{confidence:.2f}%**")

        feedback = st.radio("Was this prediction correct?", ("Yes", "No"), horizontal=True)
        if st.button("Submit Feedback"):
            with open("feedback_log.csv", "a", encoding="utf-8") as f:
                f.write(f"{user_input.replace(',', ' ')},{prediction},{feedback}\n")
            st.success("Thanks for your feedback!")

# --- Bulk Classification ---
st.markdown("---")
st.header("📂 Bulk Comment Classification (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file with a column of comments", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Pick text column
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_columns:
        st.error("No text columns found in the uploaded CSV.")
    else:
        text_column = st.selectbox("Select the column with comments", text_columns)

        if st.button("Classify CSV Comments"):
            with st.spinner("Classifying comments..."):
                predictions = []
                confidences = []

                progress_bar = st.progress(0)
                total = len(df)

                for idx, comment in enumerate(df[text_column]):
                    if pd.isna(comment) or str(comment).strip() == "":
                        predictions.append("Empty")
                        confidences.append(0.0)
                    else:
                        inputs = tokenizer(comment, truncation=True, padding=True, max_length=128, return_tensors='pt')
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            probs = torch.softmax(logits, dim=1)
                            pred = torch.argmax(probs, dim=-1).item()
                            confidence = probs[0][pred].item() * 100

                            label = "Non-Toxic" if pred == 0 else "Toxic"
                            predictions.append(label)
                            confidences.append(confidence)

                    progress_bar.progress((idx + 1) / total)

                df["Prediction"] = predictions
                df["Confidence (%)"] = [f"{c:.2f}" for c in confidences]
                st.success("Bulk classification complete!")
                st.dataframe(df.head())

                # Bar chart summary
                st.markdown("### 📊 Prediction Summary")
                chart_data = df["Prediction"].value_counts().rename_axis("Label").reset_index(name="Count")
                st.bar_chart(data=chart_data.set_index("Label"))

                # Filter viewer
                st.markdown("### 🔍 Filter Results")
                filter_option = st.radio("Show:", ("All", "Only Toxic", "Only Non-Toxic"), horizontal=True)

                if filter_option == "Only Toxic":
                    filtered_df = df[df["Prediction"] == "Toxic"]
                elif filter_option == "Only Non-Toxic":
                    filtered_df = df[df["Prediction"] == "Non-Toxic"]
                else:
                    filtered_df = df

                st.dataframe(filtered_df)

                # Download filtered results
                csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Filtered Results",
                    data=csv_filtered,
                    file_name="filtered_comments.csv",
                    mime="text/csv",
                )
