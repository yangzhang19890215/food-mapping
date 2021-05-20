"""
Aims: generating collection [appLSOA]
Required Collection:
    1. Create collection: [appStopsLiverpool] from 'preprocessing_busStops.py'
    2. Create collection: [appShopLatLong] from 'preprocessing_storeData.py'
"""
from connectMongoDB import getCollection, createCollection, dropCollection
import pandas as pd
import numpy as np
import json

''' --------------------------------------------------------------------------------
loading data from mongoDB
-------------------------------------------------------------------------------- '''
# load data using Python JSON module
with open('../assets/liverpoolmini.json', 'r') as f:
    data = json.loads(f.read())
# Flatten data
df_lsoa = pd.json_normalize(data, record_path = 'features', meta=['name'])
df_deprivation = getCollection('deprivation')
df_healthfulness = getCollection('store_type_reference')
df_store = getCollection('appShopLatLong')
df_bus = getCollection('appStopsLiverpool')

# processing
df_store['Hours'] = np.array(df_store['Hours'])
df_store['Daily Opening Hours'] = df_store['Hours'].apply(np.sum, axis = 1).apply(np.sum, axis = 0)/7
#df = df_store_openHour.groupby('LSOA')[cols].mean()
df = df_store.groupby('LSOA')[['Daily Opening Hours', 'Healthfulness']].median()

# deprivation
df_deprivation.rename(columns={'LSOA Code': 'LSOA'}, inplace = True)
df_deprivation.set_index('LSOA', inplace = True)

df_LSOAinfo = pd.concat([df_deprivation, df], join= 'outer', axis = 1)
df_LSOAinfo['LSOA'] = df_LSOAinfo.index
df_LSOAinfo.reset_index(drop = True, inplace= True)

# vehicle accessibility
df_vehicle = getCollection('Car and Van Ownership Census 2011')
df_vehicle.rename(columns={'Proportion of households with access to one or more cars or vans':'Vehicle Accessibility',
                           'geography code (LSOA?)':'LSOA'}, inplace=True)

df_LSOAinfo = df_LSOAinfo.merge(df_vehicle[['LSOA', 'Vehicle Accessibility']].copy(), how = 'left', on = 'LSOA')
df_LSOAinfo.rename(columns={'Index of Multiple Deprivation Rank': 'IMDRank',
                            'Index of Multiple Deprivation Decile': 'IMDDecile'}, inplace=True)

cols = ['LSOA', 'LSOA Name', 'IMDDecile', 'IMDRank', 'Healthfulness', 'Daily Opening Hours', 'Vehicle Accessibility']

df_export = df_LSOAinfo[cols].copy()
df_export.rename(columns={'Healthfulness': 'Healthfulness (Median)',
                          'Daily Opening Hour':'Daily Opening Hour (Median)'}, inplace = True)

cols = ['IMDDecile', 'IMDRank', 'Vehicle Accessibility']
df_export[cols] = df_export[cols].astype(float)


'''-------------------------------------------------------------------------------------------
# update MongoDB
-------------------------------------------------------------------------------------------- '''
mylist = df_export.to_dict('records')
dropCollection('appLSOA')
createCollection(mylist, 'appLSOA')






# ----------------------IGNORE ---------------------------------------- !!! ------------------

# def create_appShopLatLong(df_lsoa, df_healthfulness):
#     collName = 'ShopLatLong'
#     df = getCollection(collName)
#
#     # add col of LSOA
#     for i in df_lsoa.index:
#
#         print('{} over {}'.format(i + 1, df_lsoa.shape[0]))
#         coords = df_lsoa.loc[i, 'geometry.coordinates']
#
#         if coords.__len__() == 1:
#             poly = Polygon(coords[0])
#             # plt.plot(*poly.exterior.xy)
#             for ii in df.index:
#                 point = Point(df.loc[ii, ['Longitude']], df.loc[ii, ['Latitude']])
#                 if point.within(poly):
#                     df.loc[ii, 'LSOA'] = df_lsoa.loc[i, 'properties.lsoa11cd']
#
#                 # #xs = [p.x for p in point]
#                 # #ys = [p.y for p in point]
#                 # plt.scatter(point.x, point.y)
#         # plt.show()
#         else:
#             for i1 in range(0, coords.__len__()):
#                 poly = Polygon(coords[i1][0])
#                 for ii in df.index:
#                     point = Point(df.loc[ii, ['Longitude']], df.loc[ii, ['Latitude']])
#                     if point.within(poly):
#                         df.loc[ii, 'LSOA'] = df_lsoa.loc[i, 'properties.lsoa11cd']
#
#
#     # healthfulness score
#     df1 = pd.merge(df, df_healthfulness[['Store Type',
#                                          'Tentative Healthfulness Score (range -2 to 2)(C Black et al)']], how = 'left', left_on = 'Store Type', right_on = 'Store Type')
#
#     df1.rename(columns = {'Tentative Healthfulness Score (range -2 to 2)(C Black et al)':'Healthfulness'}, inplace = True)
#     mylist = df1.to_dict('records')
#     dropCollection('appShopLatLong')
#     createCollection(mylist, 'appShopLatLong')
#     print("Collection 'appShopLatLong' created in MongoDB")


# collName = 'lsoa_la'
# df_lsoa11nm = getCollection(collName)
# keepCols = ['LSOA11CD', 'LSOA11NM']
# df_lsoa11nm = df_lsoa11nm[keepCols].copy()
# df1 = pd.merge(df, df_lsoa11nm, how = 'left', left_on = 'LSOA', right_on = 'LSOA11CD')


#