import os
import ast
import sys
import logging
import json

import dash, dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import time
import plotly.graph_objs as go
import plotly.express as px
from flask_caching import Cache

import numpy as np
import pandas as pd
from vecLib.connectMongoDB import getCollection

import warnings

warnings.filterwarnings("ignore")

""" ----------------------------------------------------------------------------
 Configurations
---------------------------------------------------------------------------- """
cfg = dict()
# When running in Pythonanywhere
appDataPath = '/home/yangz/apps-food_mapping/data'
assetsPath = '/home/yangz/apps-food_mapping/assets'

if os.path.isdir(appDataPath):
    cfg['app_data_dir'] = appDataPath
    cfg['assets dir'] = assetsPath
    cfg['cache dir'] = 'cache'

# when running locally
else:
    cfg['app_data_dir'] = 'data'
    cfg['assets dir'] = 'assets'
    cfg['cache dir'] = 'tmp/cache'

cfg['topN'] = 50
cfg['timeout'] = 5 * 60  # Used in flask_caching
cfg['cache threshold'] = 10000  # corresponds to ~350MB max

cfg['plotly_configure'] = {
    'Liverpool': {'centre': [53.409003, -2.969830], 'maxp': 80, 'zoom': 11}
}

cfg['mapbox_token'] = open(os.path.join(cfg['assets dir'], 'mapbox_token.txt')).read()

cfg['logging format'] = 'pid %(process)5s [%(asctime)s] ' + \
                        '%(levelname)8s: %(message)s'

# ------------------------------------------------------------------------------#
logging.basicConfig(format=cfg['logging format'], level=logging.INFO)
logging.info(f"System: {sys.version}")

t0 = time.time()


""" ----------------------------------------------------------------------------
 App initialisation
 Select theme from: https://www.bootstrapcdn.com/bootswatch/
---------------------------------------------------------------------------- """
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.DARKLY]
    # external_stylesheets = [dbc.themes.CYBORG]
)

server = app.server
cache = Cache(server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': cfg['cache dir'],
    'CACHE_THRESHOLD': cfg['cache threshold']
})
app.config.suppress_callback_exceptions = True

colors = {
    'primary': '#375a7f',
    'border': '#3c4b5a',
    'info': '#3498db',
    'success': '#00bc8c',
    'background': '#444444',
    'font': '#7FDBFF'
}


""" ----------------------------------------------------------------------------
Prepare Data
1. load data functions: df_busStops, df_stores, df_LSOA, liverpool_geo_data
2. prepare regional_geo_data and geo_sectors

---------------------------------------------------------------------------- """

def getBusStops():
    coll_name = 'appStopsLiverpool'
    csv_file = 'data/' + coll_name + '.csv'
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        if 'Unnamed: 0' in df:
            del df['Unnamed: 0']
    else:
        df = getCollection(coll_name)
        df.to_csv(csv_file)
    df.rename(columns={'CommonName': 'Info'}, inplace=True)
    df['Symbol'] = 'bus'
    df['Color'] = 'blue'
    df[df['Landmark'] == 'Station']['Symbol'] = 'rail'
    return df


# ------------------------------------------------------
def getStores():
    collName = 'appShopLatLong'
    csvFile = 'data/' + collName + '.csv'
    if os.path.isfile(csvFile):
        df = pd.read_csv(csvFile)
        if 'Unnamed: 0' in df:
            del df['Unnamed: 0']
    else:
        df = getCollection(collName)
        df.to_csv(csvFile)

    df['LSOA'].fillna('Out of Range', inplace=True)
    df.rename(columns={'Shop Name': 'Info'}, inplace=True)
    if isinstance(df.loc[0, 'Hours'][0], str):
        f = lambda x: np.array(ast.literal_eval(x))
        df['Hours'] = df['Hours'].apply(f)


    df['Symbol'] = 'grocery'
    df['Color'] = '#d73027'
    df_temp = pd.DataFrame()

    storeHealth = df[['Store Type', 'Healthfulness']].drop_duplicates()
    storeHealth = storeHealth.sort_values(by='Healthfulness', na_position='first')
    storeHealth.reset_index(inplace=True, drop=True)
    storeHealth['Color'] = ['#820700', '#d73027', '#fc650d',
                            '#ffee00', '#72d600',
                            '#1a9850', '#004f22', '#00ffff']
    storeHealth['Symbol'] = ['circle', 'beer', 'ice-cream',
                             'fuel', 'bank', 'shop',
                             'grocery', 'star']

    for iType, iShopColor, iShopSymbol in zip(storeHealth['Store Type'], storeHealth['Color'], storeHealth['Symbol']):
        df.loc[df['Store Type'] == iType, 'Symbol'] = iShopSymbol
        df.loc[df['Store Type'] == iType, 'Color'] = iShopColor


    return df, storeHealth


# -----------------------------------------------------
def getLSOA(df_stores):
    collName = 'appLSOA'
    csvFile = 'data/' + collName + '.csv'
    if os.path.isfile(csvFile):
        df = pd.read_csv(csvFile)
        if 'Unnamed: 0' in df:
            del df['Unnamed: 0']
    else:
        df = getCollection(collName)
        df.to_csv(csvFile)

    df['Healthfulness (Median)'] = df['Healthfulness (Median)'].round(2)
    df['Daily Opening Hours'] = df['Daily Opening Hours'].round(2)
    df = df.reindex(columns=['LSOA', 'IMDDecile', 'IMDRank', 'Vehicle Accessibility', 'Healthfulness (Median)',
                             'Daily Opening Hours', 'LSOA Name'])
    df['text'] = df['LSOA'] + '<br>OMDDecile ' + df['IMDDecile'].astype('str') + '<br>Healthfulness ' \
                 + df['Healthfulness (Median)'].astype('str') + '<br>Vehicle Access ' + \
                 df['Vehicle Accessibility'].astype('str')

    # Sandbox
    keepCols = ['Store Type', 'Info', 'LSOA', 'Healthfulness', 'Hours']
    df_stores_sandbox = df_stores[keepCols].copy()
    df_stores_sandbox.drop(df_stores_sandbox[df_stores_sandbox['LSOA'] == 'Out of Range'].index, inplace = True)
    df_stores_sandbox['Weekly Hours'] = [df_stores_sandbox['Hours'][idx].sum() for idx in df_stores_sandbox.index]
    df_stores_sandbox.drop(columns=['Hours'], axis = 0, inplace = True)
    df_stores_sandbox = df_stores_sandbox.merge(df[['LSOA', 'Vehicle Accessibility', 'IMDDecile', 'IMDRank']], how = 'left', on = 'LSOA')

    return df, df_stores_sandbox


""" ----------------------------------------------------------------------------
# functions get geo data and sector
---------------------------------------------------------------------------- """


def getGeoData(fname):
    infile = os.path.join(cfg['assets dir'], fname)
    with open(infile, "r") as read_file:
        regional_geo_data = json.load(read_file)
    return regional_geo_data


# --------------------------------------------=====
def getGeoSector(geo_data):
    geoSector = dict()
    for feature in geo_data['features']:
        sector = feature['properties']['lsoa11cd']
        geoSector[sector] = feature
    return geoSector


""" ----------------------------------------------------------------------------
* prepare data -> before callbacks
* dash initialisation  
---------------------------------------------------------------------------- """
df_busStops = getBusStops()
df_stores, storeHealth = getStores()
df_LSOA, df_stores_sandbox = getLSOA(df_stores)
df_storeHealth = pd.DataFrame(columns=storeHealth['Store Type'])
df_storeHealth.loc[0, :] = storeHealth.loc[:,'Healthfulness'].values
df_storeHealth.iloc[0,0] = 'NA'

# ---------------- geo data and geo sector
regional_geo_path = 'liverpoolmini.json'
regional_geo_data = getGeoData(regional_geo_path)
regional_geo_sector = dict()
regional_geo_sector = getGeoSector(regional_geo_data)


"""----------------------------------------------------------------------------
App Settings
App input: lsoa, Store type, Day+Time range, Update button
App input1: basemap type: Vehicle access, Deprivation
----------------------------------------------------------------------------"""
sectors = ['All']
sectors.extend(df_LSOA['LSOA'].unique().tolist())
initial_sector = 'E01006524'
initial_geo_sector = [regional_geo_sector[initial_sector]]

baseMapTypes = ['HealthFood Access', 'Vehicle Access', 'IMD', 'Healthfulness', 'none']
initial_baseMapType = 'IMD'
tabLabel = ['Info', 'Store distribution', 'Open hour', 'Sandbox']


""" ----------------------------------------------------------------------------

 Dash Layout
 
---------------------------------------------------------------------------- """

app.layout = html.Div(
    id="root",
    children=[
        # header -------------------------------------------------------------------------------------#
        html.Div(
            id="header",
            children=[
                html.Div([html.H1(children='Liverpool City Food Mapping')],
                         style={'display': 'inline-block',
                                'width': '70%',
                                #'color': colors['info'],
                                'padding': '10px 0px 0px 20px'}  # padding: top, right, bottom, left
                         )

                # -- LOGO --
            ]
        ),

        html.Div([html.H5(children='LSOA in Liverpool City Region')],
                 style={'display': 'inline-block',
                        'color': colors['info'],
                        'padding': '5px 0px 5px 20px'}

                 ),

        # Selection control 1: Region and graph_type ------------------------------------------------- #
        html.Div(
            id='selection1',
            children=[
                html.Div([
                    dcc.Dropdown(
                        id='lsoa_dropdown',
                        options=[{'label': r, 'value': r} for r in sectors],
                        value=[initial_sector],
                        placeholder="Select the LSOA code",
                        clearable=True,
                        multi=True,
                        style={'color': 'black'}
                    )], style={
                    'display': 'inline-block',
                    'padding': '0px 5px 10px 15px',
                    'width': '58%'},
                    className="ten columns"
                ),

                html.Div([
                    dbc.RadioItems(
                        id='graph_type',
                        options=[{'label': i, 'value': i} for i in baseMapTypes],
                        value=initial_baseMapType,
                        inline=True
                    )], style={
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'padding': '5px 0px 10px 10px',
                    'width': '38%'},
                    className="seven columns"
                )
            ],
            style={'padding': '5px 0px 10px 20px'},
            className='row'
        ),

        # App container ----------------------------------------------------------------------------- #
        html.Div(
            id="app_container",
            children=[
                # left_col: Map -------------------------------------------------------------------- #
                html.Div(
                    id='left_col',
                    children=[
                        html.Div(
                            id="choropleth-container",
                            children=[
                                html.Div([
                                    html.Div([
                                        html.H5(id='choropleth_title'),
                                    ], style={'display': 'inline-block',
                                              'width': '60%'},
                                        className='twelve columns'
                                    ),

                                    html.Div([
                                        dbc.FormGroup(
                                            dbc.Checklist(
                                                id='bus_store_checklist',
                                                options=[{'label': 'Bus stops', 'value': 'Bus stops'},
                                                         {'label': 'Stores', 'value': 'Stores'}],
                                                value=[],
                                                inline=True
                                            )
                                        )], style={'display': 'inline-block',
                                                   'textAlign': 'right',
                                                   'width': '40%'},
                                        className="seven columns"
                                    )
                                ]),

                                dcc.Graph(id='choropleth'),

                                # ------------------ Store filters ----------------------------------------------
                                html.H5(["Store Filter :"],
                                       style={ 'display': 'inline-block',
                                               'textAlign': 'left',
                                               'padding': '30px 0px 0px 0px'} # padding: top, right, bottom, left
                                       ),
                                dbc.FormGroup(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(id = 'filter_day',
                                                         options=[
                                                             {'label':'NA', 'value': -1},
                                                             {'label':'Mon.', 'value':0},
                                                             {'label':'Tue.', 'value':1},
                                                             {'label':'Wed.', 'value':2},
                                                             {'label':'Tur.', 'value':3},
                                                             {'label':'Fri.', 'value':4},
                                                             {'label':'Sat.', 'value':5},
                                                             {'label':'Sun.', 'value':6}

                                                         ],
                                                         #value = -1,
                                                         placeholder = "Day of Week",
                                                         clearable=True,
                                                         style={'color': 'black'}
                                                         ),
                                            width=2
                                        ),

                                        dbc.Col(
                                            dcc.RangeSlider(id="filter_time",
                                                            min=-1,
                                                            max=23,
                                                            step=1,
                                                            value=[-1, 16],
                                                            marks={
                                                                -1: {'label':'NA', 'style' :{'color':'warning'}},
                                                                0: '0AM', 1: '1', 2: '2',  3: '3', 4: '4',
                                                                5: '5', 6: '6', 7: '7', 8: '8',
                                                                9: '9', 10: '10', 11: '11', 12: '12PM',
                                                                13: '13', 14: '14', 15: '15', 16: '16',
                                                                17: '17', 18: '18', 19: '19', 20: '20',
                                                                21: '21', 22: '22', 23: '23' }
                                                            ),
                                            width = 8,
                                        ),

                                        dbc.Col(
                                            dbc.Button("Run", color="success", id='filter_run'),
                                            width = 2
                                        )

                                    ],
                                    row = True,
                                    style={ #'display': 'inline-block',
                                            'textAlign': 'centre',
                                            'padding': '0px 0px 10px  0px'} # padding: top, right, bottom, left
                                ),
                                dbc.Card(dbc.CardBody(id = 'filter_alert')),

                            ])
                    ], style={'display': 'inline-block',
                              'padding': '20px 10px 10px 20px',
                              'width': '58%'},
                    className='twelve columns'
                ),

                # right_col: Analysis -------------------------------------------------------------- #
                html.Div(
                    id="analysis_container",
                    children=[
                        dbc.Card([
                            dbc.CardHeader(
                                # tab1_content = dbc.Card(),
                                dbc.Tabs(
                                    id="analysis_tabs",
                                    active_tab=tabLabel[0],
                                    className='tab_container',
                                    card=True,
                                    children=[
                                        dbc.Tab(
                                            label=tabLabel[0],
                                            tab_id=tabLabel[0]
                                        ),

                                        dbc.Tab(
                                            label=tabLabel[1],
                                            tab_id=tabLabel[1],
                                        ),

                                        dbc.Tab(
                                            label=tabLabel[2],
                                            tab_id=tabLabel[2],
                                        ),

                                        dbc.Tab(
                                            label = tabLabel[3],
                                            tab_id = tabLabel[3],
                                        )

                                    ])
                            ),

                            dbc.CardBody(html.Div(id="analysis_content", className="card_text")),


                        ]),


                    ], style={'display': 'inline-blocker',
                              'padding': '20px 15px 10px 15px',
                              'width': '38%'},
                    className="eight columns"
                )
            ], style={'padding': '5px 0px 10px 20px'},
            className="row"
        ),

        html.Hr(style={'borderColor': '#575757'}),

        # ---------------------------------------------------------------------------------------------------
        # Notes and credits
        # ---------------------------------------------------------------------------------------------------
        html.Div([

            html.H5(
                dbc.Button(
                    'About Food-Mapping Dashboard',
                    color = 'Primary', #'info'
                    id = 'note_button'
                )

            ),

            dbc.Collapse(
                dbc.CardBody([
                    dcc.Markdown(
                        '''
                        * **LSOA:** Lower Super Output Areas are the small areas that are 
                        socially homogenous and have a population size between 1000 - 1500.
                        * **HealthFood Access:** in processing ...
                        * **Vehicle Access:** Proportion of households with access to cars or vans
                        * **IMDDecile (IMD):** Index of Multiple Deprivation is in the range 1 to 10. 1 = the most 
                        deprived area and 10 = the least deprived area. 
                        * **Healthfulness:** In the range -2 to 2. The store with a higher score is likely to be healthier.  
                        * Store types and corresponding healthfulness scores
                        '''
                    ),
                    html.Div([
                        dbc.Table.from_dataframe(df_storeHealth, striped=True, bordered=True, dark=True, hover=True)
                        ], style = {'width': '60%'})
                ]),
                id='note_body'
            )
        ],
            style={'textAlign': 'left',
                   'padding': '0px 0px 5px 20px',
                   #'width': '60%'
                 },
            # className="fourteen columns"
        )
    ]
)

''' -----------------------------------------------------------------------------------------------
Callback functions:
1. graph-type -> choropleth-title
2. region, store_type, bus_checklist, graph-type -> choropleth 
3. region, analysis_tab -> analysis_tabs_content
------------------------------------------------------------------------------------------------'''

""" ----------------------------------------------------------------------------
Making Graphs -- used in callbacks                    
Symbol: https://labs.mapbox.com/maki-icons/
---------------------------------------------------------------------------- """
# df_storeFiltered = filter_stores(df_stores, storeFilter)
def filter_stores(storeFilter):
    f_d = storeFilter['day']
    if f_d is None: f_d = -1

    f_t = storeFilter['time']
    if f_d == -1 and -1 in f_t:
        return df_stores
    elif f_d == -1:
        store_idx = [False] * df_stores.shape[0]
        for idx in df_stores.index:
            temp = df_stores.loc[idx, 'Hours']
            store_idx[idx] = sum([temp[id][f_t].sum() for id in np.arange(0, 7)]) == 14
            df_storeFiltered = df_stores.loc[store_idx, :].copy()
        return df_storeFiltered
    elif -1 in f_t:
        store_idx = [df_stores.loc[idx, 'Hours'][f_d].sum() > 0 for idx in df_stores.index]
        df_storeFiltered = df_stores.loc[store_idx, :].copy()
        return df_storeFiltered
    else:
        store_idx = [df_stores.loc[idx, 'Hours'][f_d][f_t].sum() == 2 for idx in df_stores.index]
        df_storeFiltered = df_stores.loc[store_idx, :].copy()
        return df_storeFiltered


def get_scattergeo(df, name, symbolSize, hoverTemplate):
    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            name=name,
            mode="markers",
            lon=df['Longitude'],
            lat=df['Latitude'],
            opacity=1,
            marker={'size': np.ones(len(df)) * symbolSize,
                    'color': df['Color'],
                    #'symbol': 'star',#df['Symbol'],
                    'allowoverlap': True,
                    'opacity': 1
                    },
        )
    )
    fig.update_traces(hovertemplate=hoverTemplate, showlegend=True)
    #fig.update_layout( legend_x = 1)
    return fig



def get_Choropleth(df, geo_data, arg, marker_opacity,
                   marker_line_width, marker_line_color, fig=None):
    # get_Choropleth(df, geo_data, arg, marker_opacity = 0.4, marker_line_width = 1, marker_line_color='#6666cc')

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geo_data,
            locations=df['LSOA'],
            featureidkey="properties.lsoa11cd",
            colorscale=arg['colorscale'],
            z=arg['z_vec'],
            zmin=arg['min_value'],
            zmax=arg['max_value'],
            text=arg['text_vec'],
            hoverinfo="text",
            marker_opacity=marker_opacity,
            marker_line_width=marker_line_width,
            marker_line_color=marker_line_color,
            colorbar_title=arg['title'],
            colorbar_title_side='right'
        )
    )
    return fig

# @cache.memoize(timeout=cfg['timeout'])
def get_figure(df, gType, geo_data, geo_sectors, bschecklist, storeFilter):
    config = {'doubleClickDelay': 1000}  # Set a high delay to make double click easier

    _cfg = cfg['plotly_configure']['Liverpool']

    arg = dict()

    if gType == 'Vehicle Access':
        arg['min_value'] = np.percentile(np.array(df['Vehicle Accessibility']), 10)
        arg['max_value'] = np.percentile(np.array(df['Vehicle Accessibility']), 90)
        arg['z_vec'] = df['Vehicle Accessibility']
        arg['text_vec'] = df['text']
        arg['colorscale'] = "Viridis"
        arg['title'] = "Proportion of households with access to vehicles"
        mOpacity = 0.4
        lineColor = '#6666cc'


    elif gType == 'IMD':
        arg['min_value'] = np.percentile(np.array(df['IMDDecile']), 10)
        arg['max_value'] = np.percentile(np.array(df['IMDDecile']), 90)
        arg['z_vec'] = df['IMDDecile']
        arg['text_vec'] = df['text']
        arg['colorscale'] = 'Viridis' # "Plasma"
        arg['title'] = "Index of Multiple Deprivation Decile"
        mOpacity = 0.4
        lineColor = '#6666cc'


    elif gType == 'Healthfulness':
        arg['min_value'] = -2
        arg['max_value'] = 2
        arg['z_vec'] = df['Healthfulness (Median)']
        arg['text_vec'] = df['text']
        arg['colorscale'] = 'Viridis' #"Jet"
        arg['title'] = "Tentative Healthfulness Score (range -2 to 2)"
        mOpacity = 0.4
        lineColor = '#6666cc'

    else:
        arg['min_value'] = -2
        arg['max_value'] = 2
        arg['z_vec'] = df['Healthfulness (Median)'] * 0 + 1
        arg['text_vec'] = df['LSOA']
        arg['colorscale'] = 'Viridis'
        arg['title'] = " "
        mOpacity = 0
        lineColor = '#ffffff'
        # geo_data = ['nan']

    fig = get_Choropleth(df, geo_data, arg, marker_opacity=mOpacity,
                         marker_line_width=1, marker_line_color=lineColor)

    # --------------Highlight selections
    if geo_sectors is not None: #and len(bschecklist) == 0:
        fig = get_Choropleth(df, geo_sectors, arg, marker_opacity=1.0,
                             marker_line_width=3, marker_line_color='aqua', fig=fig)

    # ---------------------- stores scatter_geo plot
    #  bus_store_checklist
    # [{'label': 'Bus stops', 'value':'Bus stops'},
    #  {'label': 'Stores', 'value':'Stores'}],
    filter_alert = html.P('Please enable store mapping on the map.')
    if len(bschecklist) > 0:
        fig.update_traces(showscale=False)
        if 'Bus stops' in bschecklist:
            bus_fig = get_scattergeo(df_busStops, name='Bus stops',
                                     symbolSize=6, hoverTemplate=df_busStops['Info'])
            fig.add_trace(bus_fig.data[0])

        if 'Stores' in bschecklist:
            df_storeFiltered = filter_stores(storeFilter)
            filter_alert = html.P('No stores available.')
            if df_storeFiltered.shape[0] > 0:
                N = pd.DataFrame(columns=storeHealth['Store Type'], index=['Available', 'Total'], dtype='int')
                for iType in storeHealth['Store Type']:
                    df0 = df_storeFiltered[df_storeFiltered['Store Type'] == iType].copy()
                    msg = '<b>' + df0['Info'] + '</b><br><b>LSOA: </b>' + df0['LSOA'] + '<br><b>Healthfulness: </b>' + \
                          df0['Healthfulness'].astype('str') + \
                          '<br><b>Opening time: </b><br>  Monday: ' + df0['Opening Time Monday'] + \
                          '<br>  Tuesday: ' + df0['Opening Time Tuesday'] + \
                          '<br>  Wednesday: ' + df0['Opening Time Wednesday'] + \
                          '<br>  Thursday: ' + df0['Opening Time Thursday'] + \
                          '<br>  Friday: ' + df0['Opening Time Friday'] + \
                          '<br>  Saturday: ' + df0['Opening Time Saturday'] + \
                          '<br>  Sunday: ' + df0['Opening Time Sunday']
                    N['All'] = N.sum(axis=1)
                    N.loc['Available', iType] = df0.shape[0]
                    N.loc['Total', iType] = df_stores[df_stores['Store Type'] == iType].shape[0] # total number
                    store_fig = get_scattergeo(df0, name=iType, symbolSize=12, hoverTemplate=msg)
                    fig.add_trace(store_fig.data[0])

                filter_alert = dbc.Table.from_dataframe(N, striped = True, hover = True, bordered= True, dark = True)




    fig.update_layout(
        mapbox=dict(
            #accesstoken=cfg['mapbox_token'],
            center={"lat": _cfg['centre'][0], "lon": _cfg['centre'][1]},
            zoom=_cfg['zoom'],
            #style='streets'
            style = 'open-street-map'

        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
        autosize=True,
        font=dict(color=colors['font']),
        paper_bgcolor=colors['background'],
        uirevision='true',
        height = 500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )



    return fig, filter_alert


# ------------------------------------------------------------------------------------
# callback: update_map_title()
# ------------------------------------------------------------------------------------
@app.callback(
    Output('choropleth_title', 'children'),
    [Input('graph_type', 'value'),
     Input('bus_store_checklist', 'value')])
def update_map_title(gType, bs_checklist):
    l0 = len(bs_checklist)
    if l0 > 0:
        if l0 == 1:
            return bs_checklist[0] + ' in Liverpool City'
        elif l0 > 1:
            return bs_checklist[0] + ' and ' + bs_checklist[1] + ' in Liverpool City'
    elif gType is 'none':
        return "Liverpool City Region"
    else:
        return f'{gType} at Liverpool City Region'


# ------------------------------------------------------------------------------------
# callback: Update choropleth with graph_type update & sectors
# ------------------------------------------------------------------------------------
@app.callback(
    Output('choropleth', 'figure'),
    Output('filter_alert', 'children'),
    [Input('graph_type', 'value'),
     Input('lsoa_dropdown', 'value'),
     Input('bus_store_checklist', 'value'),
     Input('filter_run', 'n_clicks'),
     State('filter_day', 'value'),
     State('filter_time', 'value')])
def update_choropleth(gType, lsoa_sectors, bs_checklist, filter_r, filter_d, filter_t):
    # For high-lighting mechanism
    # changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    geo_sectors = dict()


    for k in regional_geo_data.keys():
        if k != 'features':
            geo_sectors[k] = regional_geo_data[k]

        # elif 'All' in lsoa_sectors:
        #     geo_sectors[k] = [regional_geo_sector[iSector] for iSector in lsoa_sectors
        #                       if iSector in regional_geo_sector]

        else:
            geo_sectors[k] = [regional_geo_sector[iSector] for iSector in lsoa_sectors
                              if iSector in regional_geo_sector]

    # get_figure(df, gType, geo_data, geo_sectors, bschecklist):
    storeFilter = {'day': filter_d, 'time': filter_t, 'run':filter_r}
    fig, filter_alert = get_figure(df_LSOA, gType, app.get_asset_url(regional_geo_path), geo_sectors, bs_checklist, storeFilter)
    return fig, filter_alert


# ------------------------------------------------------------------------------------
# callback: Update postcode dropdown values with clickData, selectedData and region
# ------------------------------------------------------------------------------------
@app.callback(
    Output('lsoa_dropdown', 'value'),
    [Input('choropleth', 'clickData'),
     Input('choropleth', 'selectedData'),
     Input('bus_store_checklist', 'value'),
     State('lsoa_dropdown', 'value'),
     State('choropleth', 'clickData')])
def update_lsoa_dropdown(clickData,  selectedData, bus_store, lsoa,
                         clickData_state):
    # Logic for initialisation or when Schoold sre selected
    if dash.callback_context.triggered[0]['value'] is None:
        return lsoa

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # if len(bus_store) > 0 or 'lsoa' in changed_id:
    #     clickData_state = None
    #     return []

    # --------------------------------------------#
    if 'selectedData' in changed_id:
        lsoa = [D['location'] for D in selectedData['points'][:cfg['topN']]]
    elif clickData is not None and 'location' in clickData['points'][0]:
        sector = clickData['points'][0]['location']
        if sector in lsoa:
            lsoa.remove(sector)
        elif len(lsoa) < cfg['topN']:
            lsoa.append(sector)

    return lsoa



''' -------------------------------------------------------------------------------------------------
Analysis Tabs
------------------------------------------------------------------------------------------------- '''

@cache.memoize(timeout=cfg['timeout'])
def create_analysis_data(lsoa_selected):
    # f_lsoa_selected, df_stores_selected = reate_analysis_data(lsoa_selected)
    df_lsoa_selected = df_LSOA.iloc[np.isin(df_LSOA['LSOA'], lsoa_selected), 0:-1].copy()

    cols = ['Store Type', 'Info', 'LSOA', 'Healthfulness', 'Hours']

    df_stores_selected = df_stores.loc[np.isin(df_stores['LSOA'], lsoa_selected), cols].copy()
    return df_lsoa_selected, df_stores_selected


# -------------- Tab1: 'Info'
# @cache.memoize(timeout=cfg['timeout'])
def create_analysis_table(df_lsoa_selected):
    # table = create_analysis_table(df_lsoa_selected)
    table = dash_table.DataTable(
        data=df_lsoa_selected.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_lsoa_selected.columns],
        page_action='none',
        sort_action='native',
        style_table={'height': '400px', 'overflowY': 'auto'},
        style_header={'backgroundColor': '#333333',
                      'fontWeight': 'bold'},
        style_cell={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'padding': '2px',
            'height': 'auto',
            'backgroundColor': '#555555',
            'color': 'white'}
    )
    # table = dbc.Table.from_dataframe(df_lsoa_selected, striped=True, bordered=True, hover=True,
    #                                  responsive=True, size='sm')
    return table


# ------------- Tab2 layout
# @cache.memoize(timeout=cfg['timeout'])
def create_heatmap_layout():
# analysis_heatmap_layout = create_heatmap_layout()
    analysis_heatmap_layout = [
        html.Div([
            #dbc.Label("Y-Axis"),
            dbc.FormGroup(
                    dbc.RadioItems(
                        options=[
                            {"label": 'LSOA', "value": 0},
                            {"label": 'IMD', "value": 1},
                        ],
                        value= 0,
                        id="switches-yaxis",
                        inline=True,
                    )
                )
            ],style = {'font-size':'small' }
        ),
        dcc.Graph(id = 'heatmap')
    ]
    return analysis_heatmap_layout


#  ------------ Tab3 layout
# @cache.memoize(timeout=cfg['timeout'])
def create_shopHour_layout():
    analysis_shopHour_layout = [
        html.Div([

            html.Div([
                dbc.Label("Choose a store type"),
                dbc.FormGroup(
                    dbc.RadioItems(
                        options=[
                            {"label": 'All', "value": 0},
                            {"label": 'Off-license (-1)', "value": 1},
                            {"label": "Discount home goods store (-0.8)", "value": 2},
                            {"label": 'Convenience store (-0.6)', "value": 3},
                            {"label": 'Discount supermarket (0.4)', "value": 4},
                            {"label": 'Small Supermarket (0.8)', "value": 5},
                            {"label": 'Large/Premium supermarket (1.75)', "value": 6}
                        ],
                        value= 0,
                        id="radio_storeType",
                        inline=True,
                    )
                )],style = {'font-size':'small' }
            ),
            dcc.Graph(id = 'heatmap_shopHour')
        ])
    ]
    return analysis_shopHour_layout


# ------------ Tab4 layout
# @cache.memoize(timeout=cfg['timeout'])
def create_sandbox_layout():
    axes_var = ['Store Type', 'LSOA', 'Healthfulness', 'Weekly Hours', 'Vehicle Accessibility', 'IMDDecile', 'IMDRank']
    analysis_sandbox_layout = html.Div(
        [
            dbc.Row(
                [
                # col1
                    dbc.Col([
                        html.P("X-axis"),
                        dcc.Dropdown(
                            id = 'sandbox_dropdown_x',
                            options = [{'label':i, 'value':i} for i in axes_var],
                            value = 'Store Type',
                            style={'color': 'black'}

                        )],
                        width = 6,
                    ),
                    # col2
                    dbc.Col([
                        html.P("Y-axis"),
                        dcc.Dropdown(
                            id = 'sandbox_dropdown_y',
                            options = [{'label':i, 'value':i} for i in axes_var],
                            value = 'Healthfulness',
                            style={'color': 'black'}
                        )],
                        width = 6)
                ],
                style={'display': 'inline-blocker',
                       'padding': '0px 10px 10px 10px',
                       },
            ),
        html.Div(
            dcc.Graph(id = 'sandbox_graph'),
            style={'display': 'inline-blocker',
                   'padding': '10px 10px 10px 10px',
                   }
        )

        ])

    return analysis_sandbox_layout


# ------------------------------------------------------------------------------------
# callback: update_analysis_content(active_tab, lsoa_selected)
# ------------------------------------------------------------------------------------
@app.callback(
    Output("analysis_content", "children"),
    [Input("analysis_tabs", "active_tab"),
     Input('lsoa_dropdown', 'value')]
)
# @cache.memoize(timeout=cfg['timeout'])
def update_analysis_content(active_tab, lsoa_selected):
    n_lsoa = len(lsoa_selected)
    if n_lsoa == 0:
        return html.P('Please select the regions.')

    else:
        if 'All' in lsoa_selected:
            lsoa_selected = sectors

        df_lsoa_selected, df_stores_selected = create_analysis_data(lsoa_selected)
        # 'Info': return a table
        if active_tab == tabLabel[0]:
            table = create_analysis_table(df_lsoa_selected)
            return table

        # 'Store distribution': return a fig
        elif active_tab == tabLabel[1]:
            analysis_heatmap_layout = create_heatmap_layout()
            return analysis_heatmap_layout
                #dcc.Graph(figure = fig_heatmap, id='heatmap_density')
        # 'Open hour'
        elif active_tab ==tabLabel[2]:
            analysis_shopHour_layout = create_shopHour_layout()
            return analysis_shopHour_layout

        elif active_tab == tabLabel[3]: # sandbox
            analysis_sandbox_layout = create_sandbox_layout()
            return analysis_sandbox_layout


# ------------------------------------------------------------------------------------
# Tab2
# callback: fig_heatmap = create_analysis_heatmap(df_stores_selected, df_lsoa_selected)
# ------------------------------------------------------------------------------------
@app.callback(
    Output("heatmap", "figure"),
    [Input("switches-yaxis", "value"),
     Input('lsoa_dropdown', 'value')
     ]
)
# @cache.memoize(timeout=cfg['timeout'])
def create_analysis_heatmap(yType, lsoa_selected):

    if 'All' in lsoa_selected:
        lsoa_selected = sectors

    df_lsoa_selected, df_stores_selected = create_analysis_data(lsoa_selected)
    df_display = df_stores_selected.merge(df_lsoa_selected[['LSOA', 'IMDRank', 'IMDDecile']], how='right', on='LSOA')
    df_display.sort_values(by = ['IMDRank', 'Healthfulness'], inplace=True, ascending=False, ignore_index=True)
    title = 'Store Distribution'
    if yType == 0:
        fig = px.density_heatmap(df_display, x='Healthfulness', y='LSOA', marginal_x='histogram',
                                 color_continuous_scale=px.colors.sequential.Viridis)
    elif yType == 1:
        fig = px.density_heatmap(df_display, x='Healthfulness', y='IMDDecile', marginal_x='violin',
                                 marginal_y='violin', color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_yaxes(range = [-1, 12])
    else:
        fig = go.Figure()

    fig.update_layout(plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'],
                      autosize=True,
                      height = 450,
                      #margin={'l': 20, 'b': 30, 'r': 10, 't': 40},
                      coloraxis_colorbar=dict(
                          title="Number of Stores",
                          thicknessmode="pixels", thickness=20,
                          yanchor="top", y=1,
                          ticks="outside"
                      ),
                      font_color=colors['font'])
    fig.update_xaxes(range = [-2, 2])
    # fig.update_yaxes(title = 'LSOA (Arranged IMDRank)')
    return fig


# ---------------------------------------------------------------------
# Tab 3
# callback: fig_heatmap = create_analysis_shopHour(iType, lsoa_selected)
#-----------------------------------------------------------------------
@app.callback(
    Output("heatmap_shopHour", "figure"),
    [Input("radio_storeType", "value"),
     Input('lsoa_dropdown', 'value')
     ]
)
# @cache.memoize(timeout=cfg['timeout'])
def create_analysis_shopHour(iType, lsoa_selected):
    if 'All' in lsoa_selected:
        lsoa_selected = sectors

    df_lsoa_selected, df_stores_selected = create_analysis_data(lsoa_selected)
    fig = go.Figure()
    y_day = ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.']
    x_time = np.arange(0, 23, 1)
    ads = ''
    if iType == 0:
        df = df_stores_selected.copy()
        iLabel = 'All'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"
    elif iType == 1:
        df = df_stores_selected[df_stores_selected['Healthfulness']==-1].copy()
        iLabel = 'Off-license (-1)'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"

    elif iType == 2:
        df = df_stores_selected[df_stores_selected['Healthfulness']==-0.8].copy()
        iLabel = 'Discount home goods store (-0.8)'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"
    elif iType == 3:
        df = df_stores_selected[df_stores_selected['Healthfulness']==-0.6].copy()
        iLabel = 'Convenience store (-0.6)'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"

    elif iType == 4:
        df = df_stores_selected[df_stores_selected['Healthfulness']==0.4].copy()
        iLabel = 'Discount supermarket (0.4)'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"

    elif iType == 5:
        df = df_stores_selected[df_stores_selected['Healthfulness']==0.8].copy()
        iLabel = 'Small Supermarket (0.8)'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"

    elif iType == 6:
        df = df_stores_selected[df_stores_selected['Healthfulness']==1.75].copy()
        iLabel = 'Large/Premium supermarket (1.75)'
        if df.shape[0] > 0:
            data = df['Hours'].sum(axis = 0)
            fig.add_trace(go.Heatmap( x = x_time, y = y_day, z = data, colorscale='Viridis'))
        else:
            ads = "<br><b>There are no available stores"


    else:
        ads = "<b>There are no available stores"

    fig.update_layout(
        title = f'Heatmap of opening hours ({iLabel}){ads}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        autosize=True,
        height = 400,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 40},
        coloraxis_colorbar=dict(
            title="Number of Stores",
            thicknessmode="pixels", thickness=20,
            yanchor="top", y=1,
            ticks="outside"
        ),
        font_color=colors['font']
    )
    fig.update_xaxes(title = 'Hours of Day',
                     tick0 = 0,
                     dtick = 1)

    return fig


# --------------------------------------------------------------------------
# Tab: 4
# callback: fig_heatmap = create_analysis_heatmap(df_stores_selected, df_lsoa_selected)
# ---------------------------------------------------------------------------
@app.callback(
    Output("sandbox_graph", "figure"),
    [Input("sandbox_dropdown_x", "value"),
     Input('sandbox_dropdown_y', 'value')
     ]
)
# @cache.memoize(timeout=cfg['timeout'])
def create_analysis_sandbox(varX, varY):
    df = df_stores_sandbox.sort_values('Healthfulness', na_position='first')
    fig3 = px.scatter(df, x = varX, y = varY, color='Healthfulness', hover_data=['Info', 'Store Type'],
                      color_continuous_scale = 'Viridis')
    fig3.update_layout(
        title = f'Sandbox: {varX} VS. {varY}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        autosize=True,
        height = 400,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 40},
        font_color=colors['font'],
        hovermode = 'closest'
    )
    fig3.update_traces(marker = dict(size = 12, line = dict(width = 2, color = '#ffffff')))
    return fig3



"""------------------------------------------------------------------------------------
callback: open notes area
-------------------------------------------------------------------------------------- """
@app.callback(
    Output('note_body', 'is_open'),
    [Input('note_button', 'n_clicks')],
    [State('note_body', 'is_open')]
)
# @cache.memoize(timeout=cfg['timeout'])
def make_note(n1, is_open):
    if n1:
        return not is_open
    return is_open



''' -----------------------------------------------------------------------------------------------
END
----------------------------------------------------------------------------------------------- '''

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

logging.info(f'Data Preparation completed in {time.time() - t0 :.1f} seconds')

if __name__ == "__main__":
    logging.info(sys.version)

    # If running locally in Anaconda env:
    app.run_server(debug=False)
    # app.run_server()

    # If running on AWS/Pythonanywhere production
    # app.run_server(
    #     port=8050,
    #     host='0.0.0.0'
    # )

""" ----------------------------------------------------------------------------
Terminal cmd to run:
gunicorn app:server -b 0.0.0.0:8050
---------------------------------------------------------------------------- """
# or better:
# msg = 'hello {} world'.format(my_var)
