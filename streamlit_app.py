import streamlit as st

st.title('Making a button')
result = st.button("Click Here")

if result: 
    st.write("Why hello there")
else:
    st.write("Goodbye")