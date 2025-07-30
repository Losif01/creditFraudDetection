import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(page_title='My App',layout="wide")


selected = option_menu(
    menu_title=None,
    options=["Home", "EDA", "Model"],
    icons=["house-lock", "bar-chart-line-fill", "robot"],
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
    TransformerModel = st.Page("pages/Model.py", title="Transformer Model", icon="🤖")
    pg = st.navigation([TransformerModel])
    pg.run()
    


