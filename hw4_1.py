import pandas as pd
from pycaret.classification import *
import streamlit as st

# 設置 Streamlit 頁面標題
st.set_page_config(page_title="Machine Learning Model Comparison", layout="wide")

# 標題
st.title("Titanic Survival Prediction: Model Comparison")

# 上傳資料區
uploaded_file = st.file_uploader("Upload Titanic Dataset (CSV)", type=["csv"])

if uploaded_file:
    # 讀取上傳的 CSV
    data = pd.read_csv(uploaded_file)

    # 預處理數據
    st.write("### Data Preview (First 5 Rows)")
    st.dataframe(data.head())

    # 移除不必要的欄位
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

    # 填補缺失值
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # 設置 PyCaret 環境
    st.write("### Setting up PyCaret environment...")
    with st.spinner("Setting up the environment. This might take a while..."):
        exp_clf = setup(data=data, target='Survived', session_id=123)

    # 比較模型
    st.write("### Comparing Machine Learning Models...")
    with st.spinner("Comparing models. Please wait..."):
        best_models = compare_models(n_select=16)

    # 顯示模型比較結果
    st.write("### Model Comparison Results")
    comparison_results = pull()
    st.dataframe(comparison_results)

    # 顯示最佳模型
    st.write("### Best Model")
    st.write(best_models[0])

else:
    st.info("Please upload a Titanic dataset CSV file to start.")
