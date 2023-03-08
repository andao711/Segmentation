#Import Libs
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
import streamlit as st
import streamlit.components.v1 as components
import pickle
import dill
import warnings
warnings.filterwarnings('ignore')

# Load file csv
@st.cache_data(max_entries=1000)
def load_csv(file):
    csv=pd.read_csv(file)
    return csv

# Load file txt
@st.cache_data(max_entries=1000)
def load_txt(file):
    txt=pd.read_csv(file, delimiter ='\s+',header=None,names=["customer_ID","date", "quantity", "price"])
    return txt

#load parquet
@st.cache_data(max_entries=1000)
def load_parquet(file):
    data = pd.read_parquet(file, engine='pyarrow')
    return data

# Convert df
@st.cache_data(max_entries=1000)
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

# load html file
@st.cache_resource
def load_html(file):
    HtmlFile = open(file, 'r')
    source_code = HtmlFile.read() 
    return components.html(source_code,height = 400)

# Load dill
@st.cache_resource
def load_dill(pkl_filename):
    with open(pkl_filename, 'rb') as file:  
        model = dill.load(file)
    return model

# Load pickle
@st.cache_resource
def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as file:  
        model = pickle.load(file)
    return model

# Load functions
read_data = load_dill('functions/read_data.pkl')
create_RFM = load_dill('functions/create_RFM.pkl')
preprocess =load_dill('functions/preprocess.pkl')
PCA_model = load_dill('functions/PCA_model.pkl')
Label_model = load_dill('functions/Label_model.pkl')
average_RFM = load_dill('functions/average_RFM.pkl')
model=load_pickle('model_Kmeans.pkl')

#File upload
@st.cache_data(max_entries=1000)
def upload_file(uploaded_file_1,uploaded_file_2):
    new_df1=''
    if uploaded_file_1 is not None:
        new_df1=load_txt(uploaded_file_1)
        st.write("Thanks for your RAW file")
        new_df1 = read_data(new_df1)
        new_df1= create_RFM(new_df1)
        return new_df1
    elif uploaded_file_2 is not None:
        new_df1=load_csv(uploaded_file_2)
        st.write("Thanks for your RFM file")
        return new_df1
    return new_df1

#load file result
Customer = load_parquet('files/KMeans_results.parquet')
Customer['customer_ID']=Customer['customer_ID'].astype(str)
df=load_parquet('files/data_read.parquet')
new_df=load_parquet('files/RFM_data.parquet')
new_df=new_df[['Recency','Frequency','Monetary']].copy()

#--------------
# GUI
st.markdown("<h1 style='text-align: center; color: grey;'>Data Science Project 3</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: blue;'>Segmentation Customers</h1>", unsafe_allow_html=True)
menu = ["Business Objective", "Build Project","New Prediction"]

choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':   
    st.subheader("Business Objective")
    st.image("images/new1.png")
    st.write("""
    An e-commerce company wants to segment its customers and determine marketing strategies according to these segments. For example, it is desirable to organize different campaigns to retain customers who are very profitable for the company, and different campaigns for new customers.
    """)
    st.write("""
    Customer segmentation is the process of dividing customers into groups based on common characteristics so companies can market to each group effectively and appropriately. In this project, we have applied RFM & Machine Learning algorithms to cluster customers from file CDNOW_master.txt.
    """)
    st.image("images/RFM.png")
    st.write("""
    RFM analysis is a technique used to categorize customers according to their purchasing behaviour. RFM stands for the three dimensions:
    1. Recency: This is the date when the customer made the last purchase. It is calculated by subtracting the customer's last shopping date from the analysis date.
    2. Frequency: This is the total number of purchases of the customer. In a different way, it gives the frequency of purchases made by the customer.
    3. Monetary: It is the total monetary value spent by the customer.
    """)
    st.write("""##### Our Task:""")
    st.write("""=> Problem/ Requirement: Use RFM & Machine Learning algorithms in Python for Customers segmentation.""")
    st.image("images/2.png")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.image("images/new.jpg")
    st.write("##### Dataset")
    st.write("""
    The file CDNOW_master.txt contains the entire purchase history up to the end of June 1998 of the cohort of 23,570 individuals who made their first-ever purchase at CDNOW in the first quarter of 1997. This CDNOW dataset was first used by Fader and Hardie (2001).
    Each record in this file, 69,659 in total, comprises four fields: the customer's ID, the date of the transaction, the number of CDs purchased, and the dollar value of the transaction.
    """)
    st.write("##### 1. Some data")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))

    st.write("##### 2. Visualize data")
    st.write("###### Sales in different Months")
    st.image("images/month_year.png")
    st.write("May have more than 50% of customers who bought from Q1/1997 did not return")
    st.write("###### Pareto chart")
    st.image("images/pareto.png")
    st.write("80% of company's revenue comes from top 30% of customers")
    st.write("###### Cohor chart")
    st.image("images/cohor.png")
    st.write("""
    The orders have decrease trend. Most of the customers started buying from Q1/1997. And only three new customers since Q4/1997. Three customers who bought from Q4/1997 haven't any new orders after that.
    """)

    st.write("##### 3. Build model with Kmeans and PCA...")
    st.image("images/PCA.png")
    st.write("Based on this visualization, we can see that the first PCA components explain around 80% of the dataset variance")

    st.write("##### 4. Evaluation")
    st.image('images/table.png')
    st.write("Silhouette score: "+str(0.6337074570139166))
    st.write("###### Tree plot")
    st.image("images/chart.png")
    st.write("###### 2D Scatter")
    load_html("images/2D_scatter.html")
    st.write("###### 3D Scatter")
    load_html("images/3D_scatter.html")
    st.write("###### Snake plot")
    st.image("images/snake_plot.png")
    st.write("Summary: With this silhouette score and this above charts, this model is good enough for Customers Segmentation")


elif choice == 'New Prediction':
    st.image("images/1.jpg")
    st.write("- Upload data: Choose raw file or RFM file & click Start -> Export new_file.csv with customers segmetation")
    st.write("- Input new customer: Input your new customer's Recency, Frequency, Monetary")
    st.write("- Customer_ID: Search your customer_ID")
    st.subheader("Select data")
    lines = None
    type = st.radio("Upload data (txt/csv) or Input new customer or Input customer ID?", options=("Upload", "Input_new_customer","Customer_ID"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a RAW txt/csv file", type=['txt','csv'])
        uploaded_file_2 = st.file_uploader("OR choose a RFM csv file", type=['csv'])
        if st.button('👉Start👈'):
            if (uploaded_file_1 is None) and (uploaded_file_2 is None):
                st.write("Please upload file!")
            else:
                new_df1=upload_file(uploaded_file_1,uploaded_file_2)
                lines = preprocess(new_df1)
                PCA_components = PCA_model(lines)
                lines=PCA_components.iloc[:,:1]  
                st.write("Content:")    
                y_pred_new = model.predict(lines)
                #Calculate average RFM values and size for each cluster:
                rfm_rfm_k3 = Label_model(new_df1,model)
                st.dataframe(rfm_rfm_k3.head())
                #csv
                csv=convert_df(rfm_rfm_k3)
                st.download_button(label='📥Download result file',data=csv,file_name='new_result.csv', mime='text/csv')
        
    if type=="Input_new_customer":
        Recency = st.text_input(label="Input Recency of Customer:")
        Frequency = st.number_input(label="Input Frequency of Customer:")
        Monetary = st.number_input(label="Input Monetary of Customer:")
        if st.button('👉Start👈'):
            if (Recency!="")&(Frequency!=0)&(Monetary!=0):
                data_input = pd.DataFrame({'Recency': [int(Recency)],
                                       'Frequency': [Frequency],
                                       'Monetary': [Monetary]})
                new_df1=new_df.copy()
                new_df1=new_df1.append(data_input)
                lines = preprocess(new_df1)
                PCA_components = PCA_model(lines)
                lines=PCA_components.iloc[:,:1]
                lines=lines.iloc[-1]
                st.write("Content:")
                y_pred_new=model.predict([lines])
                y_pred_new="Loyal" if y_pred_new==1 else ("Promising" if y_pred_new == 2 else "Hibernating")
                st.write("New predictions: "+y_pred_new)
                data_input['Segment']=y_pred_new
                st.dataframe(data_input,use_container_width=True)
            
    if type=="Customer_ID":
        email = st.number_input(label="Input your customer_ID:",format='%i',step=1)
        if st.button('🔍Search'):
            if email!=0:
                if email in Customer['customer_ID']:
                    email=Customer[Customer['customer_ID']==str(email)]
                    st.dataframe(email,use_container_width=True)
                else:
                    st.write("Not found customer_ID!")