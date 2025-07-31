import streamlit as st 
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import logging
logger = logging.getLogger(__name__)
import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent  # frontend/pages/
frontend_dir = current_dir.parent    # frontend/
root_dir = frontend_dir.parent       # project root

# Path to raw data
data_path = root_dir / "src" / "data" / "raw" / "creditcard.csv"

# Read the dataset
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Dataset not found at {data_path.resolve()}")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Helper function to load image
def load_image(image_name):
    img_path = current_dir / image_name
    if not img_path.exists():
        st.warning(f"Image not found: {img_path.resolve()}")
        return None
    return str(img_path)

st.markdown("""
       ###  Steps of (EDA)""")

with st.expander("***Understanding the Data Structure***"):
    st.markdown("### subset from Data :")
    st.dataframe(df.head(5))
    st.markdown("""
                                            
        ```python
            data.info()
            
        ```
        Example Output :
        
        ```
        
        <class 'pandas.core.frame.DataFrame'>
                RangeIndex: 284807 entries, 0 to 284806
                Data columns (total 31 columns):
                #   Column  Non-Null Count   Dtype  
                ---  ------  --------------   -----  
                0   Time    284807 non-null  float64
                1   V1      284807 non-null  float64
                2   V2      284807 non-null  float64
                3   V3      284807 non-null  float64
                4   V4      284807 non-null  float64
                5   V5      284807 non-null  float64
                6   V6      284807 non-null  float64
                7   V7      284807 non-null  float64
                8   V8      284807 non-null  float64
                9   V9      284807 non-null  float64
                10  V10     284807 non-null  float64
                11  V11     284807 non-null  float64
                12  V12     284807 non-null  float64
                13  V13     284807 non-null  float64
                14  V14     284807 non-null  float64
                15  V15     284807 non-null  float64
                16  V16     284807 non-null  float64
                17  V17     284807 non-null  float64
                18  V18     284807 non-null  float64
                19  V19     284807 non-null  float64
                ...
                29  Amount  284807 non-null  float64
                30  Class   284807 non-null  int64  
        ```
        
        ```python
            print(df.shape) 
        ```
            shape Output :
            
        ```
            (284807, 31)
        ```
        
        
        ```python
            print(df.colums) 
        ```
        
        columns Output:
        
        ```
        Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
        'Class'],
        dtype='object')
        
        ```
        
        """)

with st.expander("**Summary of Statistics**"):
     st.markdown('''
                 ```python
                    data.descirbe()
                 ```
                 Output:
                 
                 ''')
     st.dataframe(df.describe())
     st.markdown("""
                
     **Insights** 
    - 🔸 The dataset contains **284,807** transactions, with only a **very small portion labeled as fraud**
    - 🔸 Features are already **PCA-transformed**, meaning most have **mean ≈ 0** and are uncorrelated.
    - 🔸 Some features (like `V5`, `V8`, `V20`) have **extreme values** which might be strong fraud indicators.
    - 🔸 The `Amount` column shows **high variance**, suggesting the need for normalization before modeling.
    - 🔸 Class imbalance is **significant**, with fraud (`Class=1`) being extremely rare. Special handling is required for modeling.
    
      """)
     
     
with st.expander("**Check and handle Duplicates**"):
    st.markdown("""
                                         
   -  ```python
        df.duplicated().sum()
         
      ```
      Output :
      
      ```
      1081
      ``` 
      
      Upon examining the dataset, **1,081** duplicated **rows** were identified out of **284,807** total **records**, accounting for approximately **0.38%** of the data.

      These duplicates may arise from repeated transaction entries or data collection errors. To ensure model performance and avoid data leakage or bias, we decided to **remove duplicated records** before further preprocessing. 
     """)
     
     
with st.expander("**Check and handle Missing values**"):
        st.markdown("""
        ### check missing value                                   
        ```python
        def getMissingValues(df):
            missing_values = df.isnull().sum().sort_values(ascending=False)
            missing_values = missing_values[missing_values > 0]
            missing_values = missing_values / len(df)
            return [missing_values], missing_values.__len__()
        print(getMissingValues(df)) 
        ```
        Output :
        
        ```
        zero
        ```
        there is no missing values in dataset 
        """)
        
with st.expander("**Data Visualization**"):
    st.markdown("""
    To gain deeper insights into the structure of the data and understand patterns related to fraud detection, we conducted a series of visualizations.
    """)
    st.markdown("""
                
             ##### Class Distribution (Fraud vs. Non-Fraud)       
                """)
    
    import plotly.express as px
    d = df['Class'].value_counts().reset_index(name='Count') ## show distrubution of target  varaible 

    fig = px.pie(d,values='Count',names=['not fraud',' fraud'],hole=0.4,opacity=0.6,
                color_discrete_sequence=["#0B0A09","#FF7676"],
                labels={'label':'Class','Class':'No. Of Samples'})

    fig.add_annotation(text='We can resample the data<br> to get a balanced dataset',
                    x=1.2,y=0.9,showarrow=False,font_size=12,opacity=0.7,font_family='monospace')
    fig.add_annotation(text='Class',
                    x=0.5,y=0.5,showarrow=False,font_size=14,opacity=0.7,font_family='monospace')

    fig.update_layout(
        font_family='monospace',
        title=dict(text='Q. How many samples of Credit Card  are  not fraud ?',x=0.47,y=0.98,
                font=dict(color='#000000',size=20)),
        legend=dict(x=0.37,y=-0.05,orientation='h',traceorder='reversed'),
        hoverlabel=dict(bgcolor='white'))

    fig.update_traces(textposition='outside', textinfo='percent+label')  
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.markdown("""
                       

Only 0.17% fraudulent transaction out all the transactions. The data is highly Unbalanced. Lets first apply our models without balancing it and if we don’t get a good accuracy then we can find a way to balance this dataset. But first, let’s implement the model without it and will balance the data only if needed.
            
            """)
    
    
    
    
    
    
    
    st.markdown("""##### Transaction Amount vs Time (with Class)
                
                """)
    st.image(load_image("output.png"))
    
    st.markdown("""
                
                ### Observations:

            - The `Amount` feature appears to be **right-skewed**, indicating the presence of many small transactions and few large ones.
            - The `Time` feature does not follow a clear pattern but may reflect **periodic trends** depending on time of day or other operational factors.
            - These insights are useful for later preprocessing steps, such as normalization or feature engineering.

                
                """)
    
    
    
    
    st.markdown("""##### Transaction Amount over Time 
                
                """)
    st.image(load_image('time and amount transaction.png'))
    st.markdown("""
            
            

        - **Legitimate transactions (Class = 0, shown in red)** dominate the dataset, especially for small amounts.
        - **Fraudulent transactions (Class = 1, shown in blue)** are fewer and more scattered, but may cluster around certain time periods or specific amount ranges.
        - There is no strong linear relationship between time and amount, but some separation is visible between the two classes, which might help in classification.

        Further analysis like **feature engineering based on time intervals** (e.g., time of day, peak hours) might improve model performance.

            
            
            """)
    
    

    
    
    
    st.markdown("""##### Correlation Heatmap
            
            """)
    st.image(load_image('correlation.png'))

    st.markdown("""
                
            - Most features have very low correlation with the target variable `Class`, indicating that fraudulent behavior is not linearly separable based on single features.

            - Some features (like `V14`, `V10`, `V17`) may show moderate negative or positive correlation with `Class` — these could be potentially informative for classification.

            - Features such as `Amount` and `Time` show weak correlation with `Class`, suggesting that additional preprocessing or feature engineering may be necessary.

            Note: Low correlation does not mean a feature is not useful — non-linear models (like tree-based algorithms) can capture complex relationships that correlation alone cannot reveal.
    
                """)
    
    st.markdown("""##### Positive Correlation Insights  """)
    st.image(load_image('postive correlation.png'))
    st.markdown("""
                
                
                
        #### Features with Positive Correlation:
        The following features tend to have higher values in fraudulent transactions:

        - `V11`
        - `V8`
        - `V4`
        - `V2`

        These features also show noticeable shifts in distribution and can contribute meaningfully to the predictive model.

        Visual inspection confirms the statistical correlation values and supports selecting these features for modeling.
  
                
                """)


    
    st.markdown("""##### Negative  Correlation Insights  """)
    st.image(load_image('negative correlation.png'))
    st.markdown("""
                
           
                
                
        #### Features with Negative Correlation:
        The following features tend to have lower values in fraudulent transactions:

        - `V17`
        - `V14`
        - `V12`
        - `V10`

        These features show a clear separation in distribution between fraud and non-fraud classes, indicating that they could be valuable for classification.
                
                
                """)
    
    st.markdown('#### Features Most Strongly Related to Fraud')
    st.image(load_image('Features Most Strongly Related to Fraud.png'))
    