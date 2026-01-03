#################
# streamlit_demo.py
#################

"""
This is a simple demo of Streamlit
"""
import streamlit as st

st.title("Streamlit demo")
st.header("Hi Hello, welcome to the Streamlit demo")
st.subheader("Streamlit is a python framework for building web apps for machine learning and data science")

st.text("This is a simple demo of Streamlit")

st.success("This is a success message")
st.info("This is an info message")
st.warning("This is a warning message")
st.error("This is an error message")

if st.checkbox("Check me"):
    st.text("user checked the checkbox")
else:
    st.text("user did not check the checkbox")

state = st.radio("Select a state", ["California", "Texas", "New York"])
st.text(f"You selected {state}")

if state == "Texas":
    st.text("You selected Texas")
elif state == "California":
    st.text("You selected California")
elif state == "New York":
    st.text("You selected New York")
else:
    st.text("You selected something else")

occupation = st.selectbox("Select your occupation", ["Student", "Teacher", "Doctor", "Engineer", "Other"])
st.text(f"You selected {occupation}")

if st.button("Click me"):
    st.text("You clicked the button")


st.sidebar.header("Sidebar")
st.sidebar.text("This is a sidebar")

# Run the app
# streamlit run streamlit_demo.py

## Note:
# Use the streamlit community cloud to deploy the app 
# https://share.streamlit.io/
# Login using your github account
    

