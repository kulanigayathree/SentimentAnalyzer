import streamlit as st
import joblib

# Load model + vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# ---- Custom Page Style ----
page_bg = """
<style>
body {
    background-color: #f7f3ff;
}
.main {
    background-color: #ffffff;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
textarea {
    border-radius: 12px !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---- App Title ----
st.markdown(
    "<h1 style='text-align:center; color:#6a0dad;'>💜 Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>Type something and I'll tell you if it's Positive or Negative 🌸</p>",
    unsafe_allow_html=True
)

st.write("")

# ---- Input Box ----
user_input = st.text_area(
    "✨ Enter your text below:",
    placeholder="Write something!",
    height=120
)

# ---- Prediction Button ----
if st.button("🔮 Predict Sentiment"):
    if user_input.strip():
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]

        if prediction == "positive":
            st.success("💛 **Positive!** Your text sounds happy & bright! ✨😄")
        else:
            st.error("💔 **Negative!** Your text sounds sad or upset 😔")

    else:
        st.warning("Please enter some text first 💜")

# ---- Footer ----
st.markdown(
    "<p style='text-align:center; margin-top:40px; color:#888;'>Your daily mood checker 🌟</p>",
    unsafe_allow_html=True
)
