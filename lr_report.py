import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#setting the page index name and layout wide
st.set_page_config(
    page_title='Interactive Linear Model: Weekly Report',
    layout='wide'
)

# prepare the front end
st.title('Linear Regression Model: Weekly Report')
#st.selectbox('')


st.write("Use the slider to select the number of quotes you'd like to sample")
quotes=st.slider(label='Choose Quotes:', min_value=1, max_value=500, value=100) # quotes slider
#nb_apps=st.sidebar.slider(label='Choose NB Applications:', min_value=1, max_value=100, value=20) # nb slider
#rw_apps=st.sidebar.slider(label='Choose RW Applications:', min_value=1, max_value=50, value=10) # rw slider


# insert the dataframe and build the model
df=pd.read_excel(r'new_report.xlsx')

# building our linear regression model
def predict_nb():
    global nb_score
    
    x=df[['quotes']]
    y=df['nb_apps']

    x_train, x_test, y_train, y_test=train_test_split(x, y)

    lr=LinearRegression()
    lr.fit(x_train, y_train)
    nb_score=lr.score(x_test, y_test)
    y_pred=lr.predict([[quotes]])
    

    return y_pred

# displaying our values
nb_pred = np.round(predict_nb())
st.subheader(f'NB Applications: {nb_pred[0]} _Predicted Applications_') # in "{nb_pred[0]}"" the "[0]"" makes it show as a regular number like this: "38.43" instead of this: "[38.43]"
st.caption(f'Accuracy Score: {round(100.0*(nb_score), 2)}%')

# building our linear regression model
def predict_rw():
    global rw_score
    
    x=df[['quotes']]
    y=df['rw_apps']

    x_train, x_test, y_train, y_test=train_test_split(x, y)

    lr=LinearRegression()
    lr.fit(x_train, y_train)
    rw_score=lr.score(x_test, y_test)
    y_pred=lr.predict([[quotes]])
    

    return y_pred

# displaying our values
rw_pred = np.round(predict_rw())
st.subheader(f'RW Applications: {rw_pred[0]} _Predicted Applications_')

st.caption(f'Accuracy Score: {round(100.0*(rw_score), 2)}%')

# plotting nb prediction and rw prediction side by side
fig = px.scatter(df, x="quotes", y=["nb_apps", 'rw_apps'], opacity=0.7)
fig.add_scatter(x=[quotes], y=[nb_pred[0]], mode='markers', marker=dict(size=10), name='Predicted NB Value') # Add the predicted value to the plot
fig.add_scatter(x=[quotes], y=[rw_pred[0]], mode='markers', marker=dict(size=10), name='Predicted RW Value') # "marker=dict(size=10, color='red')" this changes the dot on the chart, very important

st.plotly_chart(fig)
