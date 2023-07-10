#Import Libs
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
import streamlit as st
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

# Read data
@st.cache_data(max_entries=1000)
def read_data(df):
    # Convert date to datetime format
    df[['date']] = df[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[6:],s[4:6], s[0:4]))
    string_to_date = lambda x : datetime.strptime(x, "%d/%m/%Y").date()
    df['date']= df['date'].apply(string_to_date)
    df['date'] = df['date'].astype('datetime64[ns]')
    #drop null
    df.dropna(inplace = True)
    #drop duplicate
    df.drop_duplicates(inplace = True)
    #remove the orders have price = 0
    df = df[df["price"] > 0]
    
    return df

# Create RFM
@st.cache_data(max_entries=1000)
def create_RFM(df):
    # Convert string to date, get max date of dataframe
    max_date = df['date'].max().date()
    #RFM
    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: x.count()
    Monetary = lambda x : round(sum(x), 2)
    df_RFM = df.groupby('customer_ID').agg({'date': Recency,
                                        'quantity': Frequency,  
                                        'price': Monetary })
    # Rename the columns of DataFrame
    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
    # Descending Sorting 
    df_RFM = df_RFM.sort_values('Monetary', ascending=False)
    #Outliers
    new_df = df_RFM[['Recency','Frequency','Monetary']].copy()
    z_scores = stats.zscore(new_df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = new_df[filtered_entries]
    
    return new_df

#Scaling data
@st.cache_data(max_entries=1000)
def preprocess(df):
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    return norm

#PCA
@st.cache_data(max_entries=1000)
def PCA_model(df):
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(df)
    PCA_components = pd.DataFrame(principalComponents)
    return PCA_components

#Calculate average RFM values and size for each cluster:
@st.cache_data(max_entries=1000)
def Label_model(new_df,_model):
    rfm_rfm = new_df.assign(K_Cluster = _model.labels_)
    rfm_rfm["Segment"]=rfm_rfm['K_Cluster'].map(lambda x:"Loyal" if x ==1
                                              else "Promising" if x == 2 
                                              else "Hibernating")
    return rfm_rfm

@st.cache_data(max_entries=1000)
def average_RFM(rfm_rfm):
    rfm_agg2 = rfm_rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(1)
    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)
    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()
    return rfm_agg2

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

#Save
@st.cache_data(max_entries=1000)
def save_dill(pkl_filename,_model_):
    with open(pkl_filename, 'wb') as file:  
        dill.dump(_model_,file)

#----------------
#Save functions
file_name=['functions/read_data.pkl','functions/create_RFM.pkl','functions/preprocess.pkl','functions/PCA_model.pkl','functions/Label_model.pkl','functions/average_RFM.pkl','functions/upload_file.pkl']
model_name=[read_data,create_RFM,preprocess,PCA_model,Label_model,average_RFM,upload_file]

for i in range(len(file_name)):
    save_dill(file_name[i],model_name[i])
