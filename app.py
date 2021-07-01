import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_Training = st.beta_container()

st.markdown(
    """
    <style>
    .main{
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)

    return taxi_data

with header:
    st.title('Welcome to my awesome datascience project')
    st.text("In this project I'm looking for new opportunity")

with dataset:
    st.header('NYC taxi dataset')
    st.text("I found this dataset on Kaggle.com")
    
    taxi_data = get_data('data/train.csv')
    st.write(taxi_data.head(10))
    
    st.subheader("Bar chart of vendors")
    vendor = pd.DataFrame(taxi_data['vendor_id'].value_counts()).head(50)
    st.bar_chart(vendor)



with features:
    st.header('The features I created')
    st.text("In this project I'm looking for new opportunity")

    st.markdown('* **first feature:**')
    st.markdown('* **second feature:**')
    st.markdown('* **third feature:**')
    st.markdown('* **forth feature:**')



with model_Training:
    st.header('Time to train the model')
    st.text("In this project I'm looking for new opportunity")

    sel_col, disp_col = st.beta_columns(2)
    
    max_depth = sel_col.slider("What should be the max depth of the model?", min_value=10, max_value=100, value=10, step=10)

    n_estimators = sel_col.selectbox("How many trees should be there?", options=[100,200,300,400, 'No limits'], index=0)
    disp_col.text("Here is the list of features:")
    disp_col.write(taxi_data.columns)
    input_feature = sel_col.text_input("Which model should be chosen?", "trip_duration")

    if n_estimators == 'No limits':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    X = taxi_data[[input_feature]]
    y = taxi_data[["passenger_count"]]
    regr.fit(X,y)
    prediction = regr.predict(y)

    disp_col.subheader("Mean absolute error of the model:")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean squared error of the model:")
    disp_col.write(mean_squared_error(y, prediction))
    
    disp_col.subheader("R2 score of the model:")
    disp_col.write(r2_score(y, prediction))

