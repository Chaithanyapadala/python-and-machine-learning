import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("PARTY IN THE PUB")
st.header("PUBS INFORMATION")

df=pd.read_csv('data/pubs.csv')
genre = st.radio(
     "what do you want in the dataset??",
     ('head','tail'))

if genre == 'head':
     st.dataframe(df.head())
else:
    st.dataframe(df.tail())
genre = st.radio(
     "what do you want about the pub dataset??",
     ('column  names','Details about the data set'))

if genre == 'column  names':
     st.dataframe(df.columns)
else:
    st.text('Number of pubs: {}'.format(df.shape[0]))
    st.text('Num of columns: {}'.format(df.shape[1]))

st.header("Number of branches of each individual Pub")

st.dataframe(df.name.value_counts())

st.header("Number of pubs in each postcode")

st.dataframe(df.postcode.value_counts())
