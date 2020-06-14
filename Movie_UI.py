#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tkinter as tk 
from tkinter import messagebox,simpledialog,filedialog
from tkinter import *
import tkinter
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


# In[4]:


root= tk.Tk() 
root.title("IMDB_MOVIE Rating prediction")
root.geometry("1300x1200")


# In[6]:


def upload_data():
    global m_df
    m_df= askopenfilename(initialdir = "Dataset")
    #pathlabel.config(text=train_data)
    text.insert(END,"Dataset loaded\n\n")


# In[8]:


def data():
    global m_df
    text.delete('1.0',END)
    m_df= pd.read_csv('movie_metadata.csv')
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,m_df.head())
    text.insert(END,"column names\n\n")
    text.insert(END,m_df.columns)
    text.insert(END,"Total no. of rows and coulmns\n\n")
    text.insert(END,m_df.shape)


# In[9]:


def statistics():
    text.delete('1.0',END)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,m_df.head())
    stats=m_df.describe()
    text.insert(END,"\n\nStatistical Measurements for Data\n\n")
    text.insert(END,stats)
    null=m_df.isnull().sum()
    text.insert(END,null)


# In[10]:


def preprocess():
    text.delete('1.0',END)
    global m_df
    m_df.dropna(axis=0,subset=['director_name', 'num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_2_name','actor_1_facebook_likes','actor_1_name','actor_3_name','facenumber_in_poster','num_user_for_reviews','language','country','actor_2_facebook_likes','plot_keywords'],inplace=True)
    m_df.drop('movie_imdb_link', axis=1, inplace=True)
    m_df.drop('color',axis=1,inplace=True)
    m_df["content_rating"].fillna("R", inplace = True)
    m_df["aspect_ratio"].fillna(m_df["aspect_ratio"].median(),inplace=True)
    m_df["budget"].fillna(m_df["budget"].median(),inplace=True)
    m_df['gross'].fillna(m_df['gross'].median(),inplace=True)
    m_df.drop_duplicates(inplace=True)
    m_df["language"].value_counts()
    m_df.drop('language',axis=1,inplace=True)
    value_counts=m_df["country"].value_counts()
    vals = value_counts[:2].index
    m_df['country'] = m_df.country.where(m_df.country.isin(vals), 'other')
    m_df["country"].value_counts()
    m_df.drop('director_name', axis=1, inplace=True)
    m_df.drop('actor_1_name',axis=1,inplace=True)
    m_df.drop('actor_2_name',axis=1,inplace=True)
    m_df.drop('actor_3_name',axis=1,inplace=True)
    m_df.drop('movie_title',axis=1,inplace=True)
    m_df.drop('plot_keywords',axis=1,inplace=True)
    m_df.drop('genres',axis=1,inplace =True)
    #m_df.drop('Profit',axis=1,inplace=True)
    m_df['Other_actor_facebbok_likes']=m_df["actor_2_facebook_likes"] + m_df['actor_3_facebook_likes']
    m_df.drop('actor_2_facebook_likes',axis=1,inplace=True)
    m_df.drop('actor_3_facebook_likes',axis=1,inplace=True)
    m_df.drop('cast_total_facebook_likes',axis=1,inplace=True)
    m_df['critic_review_ratio']=m_df['num_critic_for_reviews']/m_df['num_user_for_reviews']
    m_df.drop('num_critic_for_reviews',axis=1,inplace=True)
    m_df.drop('num_user_for_reviews',axis=1,inplace=True)
    m_df = pd.get_dummies(data = m_df, columns = ['country'] , prefix = ['country'] , drop_first = True)
    m_df = pd.get_dummies(data = m_df, columns = ['content_rating'] , prefix = ['content_rating'] , drop_first = True)
    null=m_df.isnull().sum()
    text.insert(END,null)


# In[11]:


def train_test():
    text.delete('1.0',END)
    global X,y
    global x_train,x_test,y_train,y_test,X_train,X_test
    text.delete('1.0',END)
    X=m_df.drop(['imdb_score'],axis=1)
    y=m_df['imdb_score']
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=100)
    sc_X = MinMaxScaler()
    x_train = sc_X.fit_transform(X_train)
    x_test = sc_X.transform(X_test)
    text.insert(END,"Train and Test model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(m_df))+"\n")
    text.insert(END,"Training Size : "+str(len(x_train))+"\n")
    text.insert(END,"Test Size : "+str(len(x_test))+"\n")
    return x_train,x_test,y_train,y_test,X_train,X_test


# In[12]:


def RF():
    global New_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    text.delete('1.0',END)
    clf = RandomForestRegressor(n_estimators=50, max_features='sqrt')
    clf = clf.fit(x_train, y_train)
    predictions=clf.predict(x_test)
    res = pd.DataFrame(predictions)
    res.index = X_test.index # its important for comparison
    res['predictions'] = predictions

    res.to_csv("prediction_results@rf.csv")
    features = pd.DataFrame()
    features['Feature'] = X.columns
    features['Importance'] = clf.feature_importances_
    features.sort_values(by=['Importance'], ascending=False, inplace=True)
    features.set_index('Feature', inplace=True)
    selector = SelectFromModel(clf, prefit=True)
    train_reduced = selector.transform(x_train)
    new_x_train=pd.DataFrame(train_reduced,columns=['num_voted_users','duration' ,'title_year', 'budget','gross','critic_review_ratio','movie_facebook_likes','director_facebook_likes','Other_actor_facebbok_likes','facenumber_in_poster'])
    test_reduced = selector.transform(x_test)
    new_x_test=pd.DataFrame(test_reduced,columns=['num_voted_users','duration' ,'title_year', 'budget','gross','critic_review_ratio','movie_facebook_likes','director_facebook_likes','Other_actor_facebbok_likes','facenumber_in_poster'])
    new_reduced=selector.transform(x_test)
    new_data=pd.DataFrame(new_reduced,columns=['num_voted_users','duration' ,'title_year', 'budget','gross','critic_review_ratio','movie_facebook_likes','director_facebook_likes','Other_actor_facebbok_likes','facenumber_in_poster'])
    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 5,
              'max_features': 'sqrt',
              'max_depth': 6}

    rf = RandomForestRegressor(**parameters)
    rf.fit(new_x_train, y_train)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    
    
    
    
    


# In[16]:


def LR():
    text.delete('1.0',END)
    lm=LinearRegression()
    
    lm.fit(x_train,y_train)
    predictions=lm.predict(x_test)
    res = pd.DataFrame(predictions)
    res.index = X_test.index # its important for comparison
    res['predictions'] = predictions

    res.to_csv("prediction_results@lr.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    


# In[17]:


def KNN():
    text.delete('1.0',END)
    knn = KNeighborsRegressor()
    knn = knn.fit(x_train, y_train)
    
    predictions = knn.predict(x_test)
    res = pd.DataFrame(predictions)
    res.index = X_test.index # its important for comparison
    res['predictions'] = predictions

    res.to_csv("prediction_results@knn.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    


# In[18]:


def svr():
    text.delete('1.0',END)
    global x_train,x_test,y_train,y_test
    svr = SVR(kernel='rbf')
    svr = svr.fit(x_train, y_train)
    
    predictions = svr.predict(x_test)
    res = pd.DataFrame(predictions)
    res.index = X_test.index # its important for comparison
    res['predictions'] = predictions

    res.to_csv("prediction_results@svr.csv")
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    


# In[19]:


def input_values():
    text.delete('1.0',END)
    global New_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    global rf
    global num_voted_users #our 1st input variable
    num_voted_users = float(entry1.get()) 
    
    global duration #our 2nd input variable
    duration = float(entry2.get())

    global title_year 
    title_year = float(entry3.get())

    global budget
    budget = float(entry4.get())

    global gross
    gross = float(entry5.get())
    
    global critic_review_ratio
    critic_review_ratio = float(entry6.get())
    
    global movie_facebook_likes
    movie_facebook_likes= float(entry7.get())
    
    global director_facebook_likes
    director_facebook_likes = float(entry8.get())
    
    global Other_actor_facebbok_likes
    Other_actor_facebbok_likes = float(entry9.get())
    
    global facenumber_in_poster
    facenumber_in_poster = float(entry10.get())
    
    list1=[[num_voted_users,duration ,title_year, budget,gross,critic_review_ratio,movie_facebook_likes,director_facebook_likes,Other_actor_facebbok_likes,facenumber_in_poster]]

    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

    rf = RandomForestRegressor(**parameters)
    rf.fit(new_x_train, y_train)

    
    Prediction_result  = rf.predict(list1)
    text.insert(END,"Among all Classifiers Random Forest Regressor having greater accuracy score\n\n")
    text.insert(END,"New values are predicted from Random Forest Regressor\n\n")
    text.insert(END,"Predicted IMDB_SCORE for the New inputs\n\n")
    text.insert(END,Prediction_result)


# In[ ]:


font = ('times', 14, 'bold')
title = Label(root, text='IMDB_score prediction')  
title.config(font=font)           
title.config(height=2, width=120)       
title.place(x=0,y=5)

font1 = ('times',13 ,'bold')
button1 = tk.Button (root, text='Upload Data1',width=13,command=upload_data) 
button1.config(font=font1)
button1.place(x=60,y=100)

button2 = tk.Button (root, text='Data',width=13,command=data)
button2.config(font=font1)
button2.place(x=60,y=150)

button3 = tk.Button (root, text='statistics',width=13,command=statistics)  
button3.config(font=font1)
button3.place(x=60,y=200)


button4 = tk.Button (root, text='preprocess',width=13,command=preprocess)
button4.config(font=font1) 
button4.place(x=60,y=250)

button5 = tk.Button (root, text='Train & Test',width=13,command=train_test)
button5.config(font=font1) 
button5.place(x=60,y=300)

title = Label(root, text='Application of ML models')
#title.config(bg='RoyalBlue2', fg='white')  
title.config(font=font1)           
title.config(width=25)       
title.place(x=250,y=70)

button6 = tk.Button (root, text='RF',width=15,bg='pale green',command=RF)
button6.config(font=font1) 
button6.place(x=300,y=100)

button7 = tk.Button (root, text='LR',width=15,bg='sky blue',command=LR)
button7.config(font=font1) 
button7.place(x=300,y=150)

button8 = tk.Button (root, text='KNN',width=15,bg='orange',command=KNN)
button8.config(font=font1) 
button8.place(x=300,y=200)

button9 = tk.Button (root, text='SVR',width=15,bg='violet',command=svr)
button9.config(font=font1) 
button9.place(x=300,y=250)



title = Label(root, text='Enter Input values for the New Prediction')
title.config(bg='black', fg='white')  
title.config(font=font1)           
title.config(width=40)       
title.place(x=60,y=380)

font3=('times',9,'bold')
title1 = Label(root, text='*You Should enter scaled values between 0 and 1')
 
title1.config(font=font3)           
title1.config(width=40)       
title1.place(x=50,y=415)

def clear1(event):
    entry1.delete(0, tk.END)

font2=('times',10)
entry1 = tk.Entry (root) # create 1st entry box
entry1.config(font=font2)
entry1.place(x=60, y=450,height=30,width=150)
entry1.insert(0,'num_voted_users')
entry1.bind("<FocusIn>",clear1)

def clear2(event):
    entry2.delete(0, tk.END)

font2=('times',10)
entry2 = tk.Entry (root) # create 1st entry box
entry2.config(font=font2)
entry2.place(x=150, y=450,height=30,width=150)
entry2.insert(0,'duration')
entry2.bind("<FocusIn>",clear2)


def clear3(event):
    entry3.delete(0, tk.END)

font2=('times',10)
entry3 = tk.Entry (root) # create 1st entry box
entry3.config(font=font2)
entry3.place(x=300, y=450,height=30,width=150)
entry3.insert(0,'title_year')
entry3.bind("<FocusIn>",clear3)

def clear4(event):
    entry4.delete(0, tk.END)

font2=('times',10)
entry4 = tk.Entry (root) # create 1st entry box
entry4.config(font=font2)
entry4.place(x=60, y=500,height=30,width=150)
entry4.insert(0,'budget')
entry4.bind("<FocusIn>",clear4)

def clear5(event):
    entry5.delete(0, tk.END)

font2=('times',10)
entry5 = tk.Entry (root) # create 1st entry box
entry5.config(font=font2)
entry5.place(x=150, y=500,height=30,width=150)
entry5.insert(0,'gross')
entry5.bind("<FocusIn>",clear5)

def clear6(event):
    entry6.delete(0, tk.END)

font2=('times',10)
entry6 = tk.Entry (root) # create 1st entry box
entry6.config(font=font2)
entry6.place(x=300, y=500,height=30,width=150)
entry6.insert(0,'critic_review_ratio')
entry6.bind("<FocusIn>",clear6)

def clear7(event):
    entry7.delete(0, tk.END)

font2=('times',10)
entry7 = tk.Entry (root) # create 1st entry box
entry7.config(font=font2)
entry7.place(x=60, y=550,height=30,width=150)
entry7.insert(0,'movie_facebook_likes')
entry7.bind("<FocusIn>",clear7)

def clear8(event):
    entry8.delete(0, tk.END)

font2=('times',10)
entry8 = tk.Entry (root) # create 1st entry box
entry8.config(font=font2)
entry8.place(x=150, y=550,height=30,width=150)
entry8.insert(0,'director_facebook_likes')
entry8.bind("<FocusIn>",clear8)
            
def clear9(event):
    entry9.delete(0, tk.END)

font2=('times',10)
entry9 = tk.Entry (root) # create 1st entry box
entry9.config(font=font2)
entry9.place(x=300, y=550,height=30,width=150)
entry9.insert(0,'Other_actor_facebbok_likes')
entry9.bind("<FocusIn>",clear9)

              
def clear10(event):
    entry10.delete(0, tk.END)

font2=('times',10)
entry10 = tk.Entry (root) # create 1st entry box
entry10.config(font=font2)
entry10.place(x=60, y=600,height=30,width=150)
entry10.insert(0,'facenumber_in_poster')
entry10.bind("<FocusIn>",clear10)



Prediction = tk.Button (root, text='Prediction',width=15,fg='white',bg='green',command=input_values)
Prediction.config(font=font1) 
Prediction.place(x=180,y=650)



font1 = ('times', 11, 'bold')
text=Text(root,height=32,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,xscrollcommand=scroll.set)
text.place(x=550,y=70)
text.config(font=font1)

root.mainloop()

