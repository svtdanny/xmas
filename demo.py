from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import time

if 'someCustomText' not in st.session_state:
    st.session_state['someCustomText'] = "TEXT: "+datetime.now().strftime("%H:%M:%S")

def update_options():
    st.session_state['someCustomText'] = "BUTTON: "+ st.session_state['someCustomText']
    # print("button click: ", st.session_state['someCustomText'])

st.title('Image2Image demo')

numbers = st.empty()
iter=0
while True:
    iter+=1
    with numbers.container():
        # print("KEYS: ", st.session_state['someCustomText'])

        st.text(st.session_state['someCustomText'])

        st.button("update table", on_click=update_options, key="button_search_"+str(iter))

    time.sleep(10)


# import streamlit as st
# import time
# import random

# st.session_state['keys'] = ('Email', 'Home phone', 'Mobile phone'+datetime.now().strftime("%H:%M:%S"))

# def skeleton():
#     left, right = st.columns(2)
#     with right:
#         numbers = st.empty()
#     return left, right, numbers

# left, right, numbers = skeleton()
# while True:
#     with right:
#         with numbers.container():
#             st.selectbox('food',random.sample(range(10, 40), 4))
#     time.sleep(20)