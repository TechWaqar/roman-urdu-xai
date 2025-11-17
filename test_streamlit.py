import streamlit as st

st.title("🤖 Test App")
st.write("If you see this, Streamlit is working!")

text = st.text_input("Enter text:")
if st.button("Test"):
    st.success(f"You entered: {text}")