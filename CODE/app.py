#imports 
import streamlit as st
import joblib
import json
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Streamlit page configuration
st.set_page_config(
    page_title="Fake News Detection System", # Title shown in the browser tab
    page_icon="",
    layout="centered" # Keeps the app content centered
)
# This function loads my CSS file
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call the function so the CSS is applied to the app
load_css()

# This loads the saved LinearSVC model
# @st.cache_resource means Streamlit only loads it once, which makes the app faster
@st.cache_resource
def load_svm():
    return joblib.load("fake_news_model.joblib")

# This loads the DistilBERT tokenizer and model
# The tokenizer converts text into numbers that BERT can understand
@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
    model     = DistilBertForSequenceClassification.from_pretrained("bert_model")
     # This puts the model into evaluation mode because we are only predicting, not training
    model.eval()
    return tokenizer, model

# This function loads model performance results from a JSON file
def load_metrics(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # If the file does not exist return to nothing
    return None

# Load the saved evaluation results for both models
svm_metrics  = load_metrics("model_metrics.json")
bert_metrics = load_metrics("bert_metrics.json")

# These are words that are often used in clickbait or fake news style headlines
SUSPICIOUS_WORDS = [
    "shocking", "breaking", "secret", "exposed",
    "click here", "you won't believe", "urgent",
    "government lies", "miracle", "banned"
]

# This function uses the LinearSVC model to predict if the text is real or fake
def predict_svm(text):
    model = load_svm()
    # The model expects a list so the text is placed inside square brackets
    return model.predict([text])[0]

# uses DistilBERT to predict if the text is real or fake
def predict_bert(text):
    tokenizer, model = load_bert()
    # Convert the text into tokens so DistilBERT can understand it
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return model.config.id2label[pred_id]

# displays suspicious words as small red labels on the page
def suspicious_pills_html(words):
    pills = "".join(
        f'<span style="display:inline-block;background:#7f1d1d;color:#fca5a5;'
        f'font-size:0.75rem;padding:3px 10px;border-radius:999px;margin:3px;">{w}</span>'
        for w in words
    )
    return f'<div style="margin-top:0.4rem;">{pills}</div>'

# creates a small styled box for showing a metric
def metric_tile(label, value, colour="#3b82f6"):
    return f"""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:12px;
                padding:14px 16px;text-align:center;">
        <div style="font-size:0.72rem;color:#6b7280;margin-bottom:4px;">{label}</div>
        <div style="font-size:1.5rem;font-weight:600;color:{colour};">{value}</div>
    </div>
    """


# displays the model evaluation results on the page
def metrics_section(m):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_tile("Overall Accuracy", f"{m['accuracy']}%", "#3b82f6"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_tile("Training Samples", f"{m['train_size']:,}", "#8b5cf6"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_tile("Test Samples", f"{m['test_size']:,}", "#06b6d4"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # This creates a custom HTML table to show precision, recall, and F1 score for both classes
    st.markdown("""
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <thead>
            <tr style="border-bottom:1px solid #374151;">
                <th style="text-align:left;padding:8px 10px;color:#9ca3af;font-weight:500;">Class</th>
                <th style="text-align:center;padding:8px 10px;color:#9ca3af;font-weight:500;">Precision</th>
                <th style="text-align:center;padding:8px 10px;color:#9ca3af;font-weight:500;">Recall</th>
                <th style="text-align:center;padding:8px 10px;color:#9ca3af;font-weight:500;">F1 Score</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:10px;color:#fca5a5;font-weight:500;"> Fake</td>
                <td style="text-align:center;padding:10px;color:#f9fafb;">{p_f}%</td>
                <td style="text-align:center;padding:10px;color:#f9fafb;">{r_f}%</td>
                <td style="text-align:center;padding:10px;color:#f9fafb;">{f1_f}%</td>
            </tr>
            <tr>
                <td style="padding:10px;color:#86efac;font-weight:500;"> Real</td>
                <td style="text-align:center;padding:10px;color:#f9fafb;">{p_r}%</td>
                <td style="text-align:center;padding:10px;color:#f9fafb;">{r_r}%</td>
                <td style="text-align:center;padding:10px;color:#f9fafb;">{f1_r}%</td>
            </tr>
        </tbody>
    </table>
    """.format(
        p_f=m["precision_fake"], r_f=m["recall_fake"],   f1_f=m["f1_fake"],
        p_r=m["precision_real"], r_r=m["recall_real"],   f1_r=m["f1_real"],
    ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Confusion Matrix**")
    cm = m["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;max-width:340px;
                border:1px solid #374151;border-radius:12px;overflow:hidden;margin-top:8px;">
        <div style="background:#14532d;padding:18px;text-align:center;">
            <div style="font-size:1.4rem;font-weight:600;color:#86efac;">{tn:,}</div>
            <div style="font-size:0.72rem;color:#6ee7b7;margin-top:4px;">True Fake<br>(Correct)</div>
        </div>
        <div style="background:#7f1d1d;padding:18px;text-align:center;">
            <div style="font-size:1.4rem;font-weight:600;color:#fca5a5;">{fp:,}</div>
            <div style="font-size:0.72rem;color:#fda4af;margin-top:4px;">False Real<br>(Missed fake)</div>
        </div>
        <div style="background:#7f1d1d;padding:18px;text-align:center;">
            <div style="font-size:1.4rem;font-weight:600;color:#fca5a5;">{fn:,}</div>
            <div style="font-size:0.72rem;color:#fda4af;margin-top:4px;">False Fake<br>(Missed real)</div>
        </div>
        <div style="background:#14532d;padding:18px;text-align:center;">
            <div style="font-size:1.4rem;font-weight:600;color:#86efac;">{tp:,}</div>
            <div style="font-size:0.72rem;color:#6ee7b7;margin-top:4px;">True Real<br>(Correct)</div>
        </div>
    </div>
    <p style="font-size:0.75rem;color:#4b5563;margin-top:8px;">
        Green = correct predictions &nbsp;|&nbsp; Red = incorrect predictions
    </p>
    """, unsafe_allow_html=True)




# header section with title and description of the system
st.markdown("""
<div class="hero-box">
    <h1> Fake News Detection System</h1>
    <p>
        Enter a news headline or article below. The machine learning model analyses
        the text and predicts whether it is likely <strong>real</strong> or <strong>fake</strong>.
    </p>
</div>
""", unsafe_allow_html=True)




# input section where users can paste news text and select which model to use for prediction
st.divider()
st.subheader("Enter News Text")
user_input = st.text_area("Paste a news headline or article below:", height=180)

word_count = len(user_input.split()) if user_input.strip() else 0
if 0 < word_count < 10:
    st.caption(f" Only {word_count} word(s) — longer text gives more reliable results.")

selected_model = st.radio(
    "Select model:",
    ["LinearSVC (TF-IDF)", "DistilBERT"],
    horizontal=True
)

check_news = st.button(" Check News", use_container_width=True)
st.divider()




# When the user clicks the "Check News" button, this code runs to get the prediction and display results
if check_news:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing..." if selected_model == "LinearSVC (TF-IDF)" else "Running DistilBERT — this may take a few seconds..."):
            if selected_model == "LinearSVC (TF-IDF)":
                prediction = predict_svm(user_input)
            else:
                prediction = predict_bert(user_input)

        char_count  = len(user_input)
        found_words = [w for w in SUSPICIOUS_WORDS if w in user_input.lower()]

        st.subheader(f" Prediction Result — {selected_model}")
        if prediction == "fake":
            st.error(" This news is likely **FAKE**")
        else:
            st.success(" This news appears to be **REAL**")

        st.subheader(" Article Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Word Count", word_count)
        with col2:
            st.metric("Character Count", char_count)

        if found_words:
            st.subheader(" Suspicious Words Detected")
            st.markdown(suspicious_pills_html(found_words), unsafe_allow_html=True)

        st.divider()




# displays the model evaluation results on the page 
st.subheader(" Model Evaluation Metrics")

tab1, tab2 = st.tabs(["LinearSVC (TF-IDF)", "DistilBERT"])

with tab1:
    if svm_metrics:
        metrics_section(svm_metrics)
    else:
        st.info("Run `py run_metrics.py` to generate LinearSVC metrics.", icon="ℹ")

with tab2:
    if bert_metrics:
        metrics_section(bert_metrics)
    else:
        st.info("Run the Colab notebook to generate BERT metrics.", icon="ℹ")

st.divider()




# About the system section with a step-by-step explanation of how the prediction works
st.subheader("How It Works")
steps = [
    ("1", "Input",     "You enter a news headline or article into the text box."),
    ("2", "Select",    "Choose between the LinearSVC baseline model or the DistilBERT transformer model."),
    ("3", "Analyse",   "The selected model processes the text and identifies patterns."),
    ("4", "Result",    "The model returns a prediction (REAL or FAKE)."),
]
for num, title, desc in steps:
    st.markdown(f"""
    <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:12px;">
        <div style="background:#1d4ed8;color:white;border-radius:50%;width:28px;height:28px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:0.78rem;font-weight:700;flex-shrink:0;">{num}</div>
        <div>
            <p style="margin:0;font-weight:600;color:#f9fafb;font-size:0.9rem;">{title}</p>
            <p style="margin:0;font-size:0.82rem;color:#9ca3af;">{desc}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()




# About the models section with an expander that shows more details when clicked
with st.expander("ℹ About the Models"):
    st.markdown("""
    **Model 1 — LinearSVC (Baseline)**  
    Algorithm: Linear Support Vector Classifier  
    Features: TF-IDF unigrams + bigrams  
    Accuracy: ~95.6%  
    Fast, lightweight, works well on political news headlines.

    ---

    **Model 2 — DistilBERT (Advanced)**  
    Algorithm: DistilBERT transformer (fine-tuned)  
    Features: Contextual word embeddings  
    Accuracy: See metrics tab  
    Better generalisation across domains and short text.

    **Training data:** Kaggle Fake and Real News Dataset (~44,000 headlines)  
    **Test split:** 33% held-out test set  
    **Limitations:** Trained primarily on US political news.
    """)

# Footer  
st.markdown("""
<p style="text-align:center;font-size:0.76rem;color:#4b5563;margin-top:1rem;">
    Dhavish Suneram · B00152921 · TU Dublin (TU860) · Final Year Project
</p>
""", unsafe_allow_html=True)