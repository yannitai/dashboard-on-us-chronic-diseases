import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

###import the data###
df = pd.read_csv('data/us_chronic_disease_indicators.csv')

df_unknown =df[(df.question == 'Mortality from heart failure') &
            (df.datavaluetypeid == 'NMBR')]
df_unknown = df_unknown.copy()

race=[]
for i in range(len(df_unknown)):
    if df_unknown.iloc[i].stratificationcategoryid1=='RACE':
        race.append(df_unknown.iloc[i].stratification1)
    else:
        race.append('unknown')

gender=[]
for i in range(len(df_unknown)):
    if df_unknown.iloc[i].stratificationcategoryid1=='GENDER':
        gender.append(df_unknown.iloc[i].stratification1)
    else:
        gender.append('unknown')

overall=[]
for i in range(len(df_unknown)):
    if df_unknown.iloc[i].stratificationcategoryid1=='OVERALL':
        overall.append(df_unknown.iloc[i].stratification1)
    else:
        overall.append('unknown')

df_unknown['Gender']=gender
df_unknown['Race']=race
df_unknown['Overall']=overall
        
encode_cols = ['Gender', 'Race','Overall','locationabbr']
nomial_cols = df_unknown[['Gender', 'Race','Overall','locationabbr']]
dummies = pd.get_dummies(nomial_cols, columns = encode_cols, drop_first=True)

###Combine data###
X=pd.merge(df_unknown['yearstart'],dummies,left_index=True, right_index=True)
Y=df_unknown['datavalue']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

###RF model###
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
#pickle the model
RF_model = open('model.pkl', 'wb')
pickle.dump(rf_model, RF_model)

def open_MLmodel():
    """
    Callout the trained RF model.
    """
    global RF_res
    RF_open=open('model.pkl','rb')
    RF_res=pickle.load(RF_open)


###Join in the predict variables and predict the datavalue###
def prediction_value(year,location,gender,race):
    """
    Use the RF model to predict the datavalue of the disease.
    """
    global X_pred_v,dt_value
    
    open_MLmodel()
    gender_idx=np.where(dummies.columns == 'Gender_{}'.format(gender), 1, 0)
    race_idx=np.where(dummies.columns == 'Race_{}'.format(race), 1, 0)
    location_idx=np.where(dummies.columns == 'locationabbr_{}'.format(location), 1, 0)
    predict_idx=gender_idx+race_idx+location_idx
    predict_value=np.append(year,predict_idx)
    X.loc[0]=predict_value
    X_pred=X.iloc[0]
    X_pred_v=X_pred.to_frame().T
    dt_value=RF_res.predict(X_pred_v)
    
    return dt_value