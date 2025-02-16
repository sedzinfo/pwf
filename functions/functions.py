# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:20:38 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import numpy as np
import pandas as pd
import string as st
import random as rnd
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
##########################################################################################
# POPULATE DATAFRAME
##########################################################################################
def populate_dataframe(ncols=5,nrows=5,value=np.nan):
    collumn_names=range(1,ncols+1)
    mydata=pd.DataFrame(columns=collumn_names)
    mydata=mydata.reindex(range(1,nrows+1))
    mydata[mydata.isnull()]=value
    return mydata
# populate_dataframe()
##########################################################################################
# REMOVE ZERO VARIANCE COLLUMNS
##########################################################################################
def remove_zero_variance_collumns(mydata):
    mydata=mydata.drop(mydata.std()[mydata.std()==0].index.values,axis=1)
    return mydata
# remove_zero_variance_collumns(pd.DataFrame({'col1':[1,1,1,1,1,1,1,1],'col2':[1,1,1,1,1,1,1,2]}))
##########################################################################################
# REMOVE NA COLLUMNS
##########################################################################################
def remove_na_collumns(mydata):
    mydata=mydata.dropna(axis=1,how='all')
    return mydata
# remove_na_collumns(pd.DataFrame({'col1':[1,1,1,1],'col2':[np.NaN,np.NaN,np.NaN,np.NaN]}))
##########################################################################################
# GENERATE MISSING DATA
##########################################################################################
def generate_missing(vector,c=1):
    vector.ravel()[np.random.choice(vector.size,c,replace=True)]=np.nan
    return vector
# generate_missing(np.random.randn(10))
##########################################################################################
# GENERATE MISSING DATAFRAME
##########################################################################################
def generate_missing_df(mydata,p1=.1,p2=.9):
    mask=np.random.choice([True,False],size=mydata.shape,p=[p1,p2])
    return mydata.mask(mask)
# generate_missing_df(generate_normal())
##########################################################################################
# GENERATE NORMAL
##########################################################################################
def generate_normal(ncols=5,nrows=5,mean=0,sd=1):
    mydata=populate_dataframe(ncols,nrows)
    for x in range(ncols+1):
        mydata[x]=np.random.normal(mean,sd,size=nrows)
    return(mydata)
# generate_normal()
##########################################################################################
# GENERATE UNIFORM
##########################################################################################
def generate_uniform(ncols=5,nrows=5,mini=0,maxi=1,decimals=2):
    mydata=populate_dataframe(ncols,nrows)
    for x in range(ncols+1):
        mydata[x]=np.random.uniform(mini,maxi,size=nrows)
    mydata=mydata.round(decimals=decimals)
    return(mydata)
# generate_uniform()
##########################################################################################
# GENERATE FACTOR EXACT
##########################################################################################
def generate_factor_exact(name=[rnd.choice(st.ascii_uppercase) for _ in range(2)],length=10):
    vector=np.repeat(name,repeats=length/len(name))
    return vector
# generate_factor_exact()
##########################################################################################
# GENERATE FACTOR RANDOMIZED
##########################################################################################
def generate_factor_randomized(name=[rnd.choice(st.ascii_uppercase) for _ in range(2)],length=10):
    vector=np.random.choice(name,size=length,replace=True)
    return vector
# generate_factor_randomized()
##########################################################################################
# RANDOM STRING
##########################################################################################
def random_string(name=st.ascii_uppercase,vector_length=10,string_length=10):
    vector=[''.join(rnd.choice(name) for _ in range(string_length)) for _ in range(vector_length)]
    return vector
# random_string()
##########################################################################################
# GENERATE MULTIPLE RESPONCE VECTOR
##########################################################################################
def generate_multiple_responce_vector(responces=range(4),responded=range(4),vector_length=10):
    # responces=''.join(map(str,responces))
    # responded=''.join(map(str,responded))
    vector=[' '.join(str(np.random.choice(responces,size=np.random.choice(responded),replace=False))) for _ in range(vector_length)]
    # vector=[int(x) for x in vector]
    return vector
# generate_multiple_responce_vector()
##########################################################################################
# GENERATE CORRELATION MATRIX
##########################################################################################
def generate_correlation_matrix(correlation_martix,nrows=1000):
    mydata=pd.DataFrame(np.random.multivariate_normal(mean=np.repeat(0,len(correlation_martix)),cov=correlation_martix,size=nrows))
    return mydata
# generate_correlation_matrix(generate_normal().corr()).corr()
##########################################################################################
# SIMULATE CORRELATION FROM SAMPLE
##########################################################################################
def simulate_correlation_from_sample(cordata,nrows=1000):
    mydata=pd.DataFrame(np.random.multivariate_normal(mean=np.array(cordata.mean()),cov=cordata.cov(),size=nrows))
    return mydata
# simulate_correlation_from_sample(generate_correlation_matrix(generate_normal().corr()).corr())
##########################################################################################
# DISPLAY LOWER DIAGONAL
##########################################################################################
def matrix_triangle(m,triangle,off_diagonal=np.nan,value=np.nan):
    if triangle=="lower":
        m=np.tril(m)
        np.fill_diagonal(m,value)
    if triangle=="upper":
        m=np.triu(m)
        np.fill_diagonal(m,value)
    np.fill_diagonal(m,off_diagonal)
    return m
# m=generate_correlation_matrix(generate_normal().corr()).corr()
# print(matrix_triangle(pd.DataFrame.as_matrix(m),triangle="lower"))
# print(matrix_triangle(pd.DataFrame.as_matrix(m),triangle="upper"))
##########################################################################################
# DISPLAY UPPER DIAGONAL AND LOWER DIAGONAL
##########################################################################################
def display_upper_lower_diagonal(m_upper,m_lower,value=np.nan):
    lower=matrix_triangle(m=m_lower,value=value)
    upper=matrix_triangle(m=m_upper,value=value)
    m=upper+lower
    return m
# cordata1=generate_normal(ncols=3).corr().as_matrix(columns=None)
# cordata2=generate_normal(ncols=3).corr().as_matrix(columns=None)
##########################################################################################
# LIST TO NUMBER STRING
##########################################################################################
def list_to_number_string(value):
    if isinstance(value, (list, tuple)):
        return str(value)[1:-1]
    else:
        return value
