import streamlit as st
import base64
from pathlib import Path

# Get the directory where this script (Home.py) is located
current_dir = Path(__file__).parent

# Build the path to the image relative to this script
image_path = current_dir / "3-Credit-Card-Fraud-Detection.png"

# Read and encode the image
with open(image_path, "rb") as img_file:
    encoded_image = base64.b64encode(img_file.read()).decode()

# Display the image centered
st.markdown("## 💳 Credit Card Fraud Detection")
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{}' width='60%' />
    </div>
    """.format(encoded_image),
    unsafe_allow_html=True
)
st.markdown("### Project Overview")
st.markdown('The use of online banking and credit card is increasing day by day. As the usage of credit/debit card or netbanking is increasing, the possibility of many fraud activities is also increasing. There are many incidents are happened in presently where because of lack of knowledge the credit card users are sharing their personal details, card details and one time password to a unknown fake call. And the result will be fraud happened with the account. Fraud is the problem that it is very difficult to trace the fraud person if he made call from a fake identity sim or call made by some internet services. So in this research some supervised methodologies and algorithms are used to detect fraud which gives approximate accurate results. The illegal or fraud activities put very negative impact on the business and customers loose trust on the company. It also affects the revenue and turnover of the company. In this research isolation forest algorithm is applied for classification to detect the fraud activities and the data sets are collected from the professional survey organizations.')
st.markdown("### About the Dataset")
st.markdown('The dataset contains transactions made by credit cards in September 2013 by European cardholders.This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are `Time` and `Amount`. Feature `Time` contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature `Amount` is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature `Class` is the response variable and it takes value 1 in case of fraud and 0 otherwise.Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.')
st.markdown("### Problem Statement")
st.markdown('With millions of credit card transactions happening daily, detecting fraudulent ones quickly and accurately is difficult. Manual review is not scalable, and delays in detection can lead to significant losses')
st.markdown('### Business Goal')
st.markdown('''To develop a machine learning model that can:
- Automatically flag suspicious credit card transactions,
- Minimize financial risk,
- Support fraud analysts in their daily investigations,
- Improve customer trust and satisfaction.

This solution will help reduce losses, improve response time, and enhance the overall security of digital transactions.
''')
st.markdown('### Contributors:')
st.markdown("- _Yousef Fawzi_ \n - _Hend Ramadan_")
st.sidebar.markdown("# Contact us! \n - ### [Hend Ramadan](mailto:hendtalba@gmail.com) \n - ### [Yousef Fawzi](mailto:losif.ai.2050@gmail.com) \n ### [github repo](https://github.com/Losif01/creditFraudDetection)")
