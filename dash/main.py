import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import visualization
import prediction

app = dash.Dash(__name__)
server = app.server

# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv('/home/yannitai/dash-gcp-template/data/us_chronic_disease_indicators.csv')

df1= df[(df['topic'] == 'Cardiovascular Disease') &
                (df['locationabbr'] == 'MI') &
                (df['stratification1'].isin(['Male', 'Female'])) &
                (df['datavaluetypeid'] == 'CRDPREV')]

####Problem 3___data####
df_race_cruderate = df[(df['locationabbr'] =='US') &
                (df['stratificationcategory1'] == 'Race/Ethnicity') &
                (df['datavaluetype'] == 'Average Annual Crude Rate')]
####Problem 4___mapping plot####
df_race = df[(df['yearstart'] == 2021) & (~df['locationabbr'].isin(['US','PR', 'GU'])) &
           (df['question'] == 'Obesity among adults aged >= 18 years') &
           (df['datavaluetypeid'] == 'AGEADJPREV') &
           (df['stratificationcategoryid1'] == 'RACE')]

df_gender = df[(df['yearstart'] == 2021) & (~df['locationabbr'].isin(['US','PR', 'GU'])) &
           (df['question'] == 'Obesity among adults aged >= 18 years') &
           (df['datavaluetypeid'] == 'AGEADJPREV') &
           (df['stratificationcategoryid1'] == 'GENDER')]

######################
###DashBoard Layout###
######################
app.layout = html.Div([
    html.Div([
        html.H1('Dashboard on Chronic Diseases',style={'textAlign':'center'}),
        html.H3(children='Plot of Crude Prevalence Over Years Among Different Questions'
        ),
        html.P('Below, you can choose any question that interests you to understand the differences in crude prevalence among different genders.'),
        html.H4("Select a question: "),

        dcc.Dropdown(
            df1.question.unique(),'Awareness of high blood pressure among adults aged >= 18 years', id='dropdown-selection'
        ),
        dcc.Graph(
            id='graph-content'
        ),

        html.H3("Alcohol Use Rate in Youth by State"),
        html.P('In the first chart, we will present the Alcohol Use Rate in Youth by State for different years.'),
        dcc.RadioItems(
            id='alcohol_radio',
            options=[2013,2015,2017,2019],
            value='2019',
            inline=True),
        dcc.Graph(
            id='graph_alcohol'
        ),
        html.P('Then, we can further examine the differences across different years.'
        ),
        dcc.Graph(figure=visualization.alcohol_use()),
    
        html.H3(children='Annual Average Crude Rate by Race'
        ),
        html.P('The following chart shows the crude rates of different races across different health issues.'),
        dcc.Dropdown(
            df_race_cruderate.question.unique(),"Invasive cancer of the oral cavity or pharynx, incidence", id='cancer_question'
        ),
        dcc.Graph(
            id='cancer_graph'
        )
    ]),
    html.Div([
        html.H3("Heatmap of Obesity Rate among group aged >= 18 years in 2021"),
        html.P('Next, we can observe the heat map of the Obesity Rate based on categories of interest such as gender or race.')
    ]),

    html.Div([
        html.Div([
            html.P("Select a candidate: "),
            dcc.RadioItems(
                id='gender_candidate',
                options=df_gender.stratification1.unique(),
                value='Male',
                inline=True),  
                       
            dcc.Graph(id="gender_map")
        ],style={'padding': 5, 'flex': 1}
        ),
        html.Div([
            html.P("Select a candidate: "),
            dcc.RadioItems(
                id='race_candidate',
                options=df_race.stratification1.unique(),
                value='Other, non-Hispanic',
                inline=True),

            dcc.Graph(id="race_map")
        ],style={'padding': 5, 'flex': 1})  
    ],style={'display': 'flex', 'flexDirection': 'row'}),
    html.Div([
        html.H3("Predictions of the future counts of 'Mortality from heart failure' disease"),
        html.P("You can select the value of your interest to get the possible value of 'Mortality from heart failure' disease.")
    ]),
    
    html.Div([
        html.Div([
            html.P('Enter a future year(after 2023):'),
            dcc.Input(id='future_year_input',
            placeholder="future year",type='number'),
            
            html.P(""),
            html.P('Select a state:'),
            dcc.Dropdown(df[~df['locationabbr'].isin(['US','PR', 'GU'])].locationabbr.unique(),
            placeholder='Select a state',id='location_input')
        ],style={'padding': 5, 'flex': 1}),
        html.Div([
            html.P('Select a gender:'),
            dcc.RadioItems(df[df.stratificationcategory1=='Gender'].stratification1.unique(),
            inline=True,
            id='gender_radio'),

            html.P(""),
            html.P("Select a race:"),
            dcc.RadioItems(np.array(['American Indian or Alaska Native', 'White, non-Hispanic',
            'Black, non-Hispanic', 'Hispanic', 'Asian or Pacific Islander']),
            inline=True,
            id='race_radio')
        ],style={'padding': 5, 'flex': 1})
    ],style={'display': 'flex', 'flexDirection': 'row'}),
    
    
    html.Div([
        html.Div([
            html.H4(id='value_cho'),
            html.H4(id='predict_result')
        ])
    ])    
])



@app.callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection','value')
)
def crude_prevelance(value):
    return visualization.update_graph(value)


@app.callback(
    Output('graph_alcohol','figure'),
    Input('alcohol_radio','value')
)
def alcohol_selection(value):
    return visualization.alcohol_select(value)

@app.callback(
    Output('cancer_graph','figure'),
    Input('cancer_question','value')
)
def cancer_race_graph(value):
    return visualization.cancer_race(value)

#Problem4 callback function
@app.callback(
    Output('gender_map','figure'),
    Input('gender_candidate','value')
)
def gender_choropleth(value):
    return visualization.gender_map(value)

@app.callback(
    Output('race_map','figure'),
    Input('race_candidate','value')
)
def race_choropleth(value):
    return visualization.race_map(value)

@app.callback(
    #"The predicted value of counts of 'Mortality from heart failure' disease is"
    Output('predict_result','children'),
    Input('future_year_input','value'),
    Input('location_input','value'),
    Input('gender_radio','value'),
    Input('race_radio','value')
)
def prediction_model(year,location,gender,race):
    result=prediction.prediction_value(year,location,gender,race)[0]
    return f"The predicted value of counts of 'Mortality from heart failure' disease is {result}"

@app.callback(
    #"The predicted value of counts of 'Mortality from heart failure' disease is"
    Output('value_cho','children'),
    Input('future_year_input','value'),
    Input('location_input','value'),
    Input('gender_radio','value'),
    Input('race_radio','value')
)
def value_choice(year,location,gender,race):
    return f"The value you selected is year={year}, location={location}, Gender={gender}, Race={race}"

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
