import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import matplotlib.pyplot as plt


###import the data###
df = pd.read_csv('data/us_chronic_disease_indicators.csv')

###Problem 1###
df1= df[(df['topic'] == 'Cardiovascular Disease') &
                (df['locationabbr'] == 'MI') &
                (df['stratification1'].isin(['Male', 'Female'])) &
                (df['datavaluetypeid'] == 'CRDPREV')]

def update_graph(value):
    """
    This is the CALLBACK function for Crude Prevelance.
    In this problem, we are looking for Crude Prevelance (percent) over years
    by the questions that people might want to know. 
    """
    dff_m = df1[(df1.question==value) & (df1.stratification1=="Male")].groupby(df1["yearstart"])
    dff_f = df1[(df1.question==value) & (df1.stratification1=="Female")].groupby(df1["yearstart"])
    mu=dff_m['datavalue'].mean()
    fmu=dff_f['datavalue'].mean()
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(row=1, col=1,trace=go.Scatter(x=mu.index, y=mu.values,name="Male")) 
    fig.add_trace(row=1, col=2,trace=go.Scatter(x=fmu.index, y=fmu.values,name="Female")) 

    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Crude Prevelance (percent)", range=[30, 100], row=1, col=1)
    fig.update_yaxes(title_text="Crude Prevelance (percent)", range=[30, 100], row=1, col=2)

    return fig

###Problem2###
df2 = df[(df['yearstart'] == 2019) &
                (~df['locationabbr'].isin(['US','PR', 'GU'])) &
                (df['question'] == 'Alcohol use among youth')  &
                (df['stratificationcategoryid1'] == 'OVERALL')]

df3 = df[(df['yearstart'] == 2017) &
                (~df['locationabbr'].isin(['US','PR', 'GU'])) &
                (df['question'] == 'Alcohol use among youth')  &
                (df['stratificationcategoryid1'] == 'OVERALL')]

df4 = df[(df['yearstart'] == 2015) &
                (~df['locationabbr'].isin(['US','PR', 'GU'])) &
                (df['question'] == 'Alcohol use among youth')  &
                (df['stratificationcategoryid1'] == 'OVERALL')]

df5 = df[(df['yearstart'] == 2013) &
                (~df['locationabbr'].isin(['US','PR', 'GU'])) &
                (df['question'] == 'Alcohol use among youth')  &
                (df['stratificationcategoryid1'] == 'OVERALL')]

def alcohol_use():
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df5['locationabbr'], y=df5['datavalue'],
        name='2013',
        marker_color='#C70039'
    ))

    fig2.add_trace(go.Bar(
        x=df4['locationabbr'], y=df4['datavalue'],
        name='2015',
        marker_color='#FF5733'
    ))

    fig2.add_trace(go.Bar(
        x=df3['locationabbr'], y=df3['datavalue'],
        name='2017',
        marker_color='#FFC300'
    ))

    fig2.add_trace(go.Bar(
        x=df2['locationabbr'], y=df2['datavalue'],
        name='2019',
        marker_color='green'
    ))
    fig2.update_layout(width=1500,height=600,xaxis_title="States", yaxis_title="Percentage",
        barmode='group', xaxis_tickangle=-45,xaxis={'categoryorder':'total descending'})

    return fig2

###Problem3###
#crude rate by race
####Problem 3___data####
df_race_cruderate = df[(df['locationabbr'] =='US') &
                (df['stratificationcategory1'] == 'Race/Ethnicity') &
                (df['datavaluetype'] == 'Average Annual Crude Rate')]

def cancer_race(value):
    """
    The CALLBACK function of cancer rates by races. 
    """
    df_pivot=df_race_cruderate[df_race_cruderate['question']==value].pivot(index='yearstart', columns='stratification1', values='datavalue')
    fig=px.line(df_pivot)
    fig.update_traces(mode='markers+lines')
    fig.update_layout(title=value,xaxis_title="Year", yaxis_title="Average Annual Crude Rate", legend_title="Race")

    return fig

###Proble4###
#create map from selected candidate about the obesity rate
df_race = df[(df['yearstart'] == 2021) & (~df['locationabbr'].isin(['US','PR', 'GU'])) &
           (df['question'] == 'Obesity among adults aged >= 18 years') &
           (df['datavaluetypeid'] == 'AGEADJPREV') &
           (df['stratificationcategoryid1'] == 'RACE')]

df_gender = df[(df['yearstart'] == 2021) & (~df['locationabbr'].isin(['US','PR', 'GU'])) &
           (df['question'] == 'Obesity among adults aged >= 18 years') &
           (df['datavaluetypeid'] == 'AGEADJPREV') &
           (df['stratificationcategoryid1'] == 'GENDER')]

def gender_map(value):
    """
    The CALLBACK function of obesity heat map by gender.
    """
    gender=df_gender[df_gender['stratification1']==value]
    fig = px.choropleth(gender, locations='locationabbr', locationmode="USA-states",
                    color='datavalue',scope="usa",labels={'datavalue':'ObesityRate'})
    fig.update_layout(title='Heatmap of Obesity Rate among {} aged >= 18 years in 2021'.format(value),font=dict(size=10))

    return fig

def race_map(value):
    """
    The CALLBACK function of obesity heat map by race.
    """
    race=df_race[df_race['stratification1']==value]
    fig = px.choropleth(race, locations='locationabbr', locationmode="USA-states",
                        color='datavalue',scope="usa",labels={'datavalue':'ObesityRate'})
    fig.update_layout(title='Heatmap of Obesity Rate among {} group aged >= 18 years in 2021'.format(value),font=dict(size=10))

    return fig
#######
def alcohol_select(value):
    df_alcohol= df[(df['yearstart'] == value) & 
                (~df['locationabbr'].isin(['US','PR', 'GU'])) &
                (df['question'] == 'Alcohol use among youth')  &
                (df['stratificationcategoryid1'] == 'OVERALL')]

    fig=px.bar(df_alcohol, x='locationabbr', y='datavalue',color='datavalue'
           ,height=400,labels={'locationabbr':'State','datavalue':'Percentage'})
    fig.update_layout(xaxis={'categoryorder':'total descending'}
                  ,xaxis_title="States", yaxis_title="Percentage",
                 title="Alcohol Use Rate in Youth by State in {}".format(value))
    return fig