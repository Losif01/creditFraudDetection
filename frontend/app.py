import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(page_title='My App',layout="wide")


selected = option_menu(
    menu_title=None,
    options=["Home", "EDA", "Model", "Software Engineering"],
    icons=["house-lock", "bar-chart-line-fill", "robot", "book"],
    orientation="horizontal",
    default_index=0,
    key='top_menu',
    styles={
        "container": {"padding": "0!important", "background-color": "#1f2937"},
        "icon": {"color": "#facc15", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#374151",
        },
        "nav-link-selected": {"background-color": "#111827"},
    }
)

if selected == "Home":
    HomePage = st.Page("pages/Home.py", title="Main Page", icon="🏠")
    pg = st.navigation([HomePage])
    pg.run()
    
    

elif selected == "EDA":
    Eda = st.Page("pages/EDA.py", title="EDA", icon="📊")
    pg = st.navigation([Eda])
    pg.run()
    

elif selected == "Model":
    Models = st.Page("pages/Model.py", title="Models", icon="🤖")
    pg = st.navigation([Models])
    pg.run()

elif selected == "Software Engineering":
    Swe = st.Page("pages/SWE.py", title="Models", icon="📘")
    pg = st.navigation([Swe])
    pg.run()


    


