"""
Generating [appShopLatLong] from Priya's original table [ShopLatLong]
including:
1. Extract postcode from Address and get Latitude and Longitude
2. Add Healthfulness score
3. Open hours collection from Google place api and generate opening hour matrix

-- uncomment the last three lines to enable update the MongoDB
"""

from connectMongoDB import getCollection, dropCollection, createCollection
import pandas as pd
import requests
import numpy as np
import os


def containsLetterAndNumber(input):
    return input.isalnum() and not input.isalpha() and not input.isdigit()


""" ----------------------------------------------------------------------------------------------------------
Task1:
Extract 'Postcode' from 'Address'
Extract latitude and longitude from api:  http://api.postcodes.io/postcodes/
----------------------------------------------------------------------------- """
collName = 'ShopLatLong'
df = getCollection(collName)

df_add = df['Address'].str.split(', | ',expand = True)
df_add = df_add.fillna('XXX')
df_tp = df_add.applymap(containsLetterAndNumber)
df_tp = df_add[df_tp]
df_postcode = pd.DataFrame(columns=['Postcode'])
for i in df_tp.index:
    temp = df_tp.loc[i, :].dropna()
    df_postcode.loc[i, 'Postcode'] = temp.iloc[-2] + ' ' + temp.iloc[-1]

df_postcode['Postcode'] = df_postcode['Postcode'].str.upper()

for i in df_postcode.index:
    url = f"http://api.postcodes.io/postcodes/{df_postcode.loc[i, 'Postcode']}"
    jsonValue = requests.get(url).json()
    if 'result' in jsonValue:
        df_postcode.loc[i, 'Longitude'] = jsonValue['result']['longitude']
        df_postcode.loc[i, 'Latitude'] = jsonValue['result']['latitude']
        df_postcode.loc[i, 'LSOA'] = jsonValue['result']['codes']['lsoa']


df_postcode0 = df_postcode.loc[df_postcode['Latitude'].isna(),:].copy()
print(df_postcode0)
df_postcode.iloc[42,0] = 'WA11 9AA'
df_postcode.iloc[42,1:3] = [-2.708678, 53.458639]

df_postcode.iloc[52,1:3] = [-2.598469, 53.472422]

df_postcode.iloc[103, 0] = 'CH41 9DF'
df_postcode.iloc[103,1:3] = [-3.016032, 53.386132]
df_postcode.iloc[103, 3] = 'E01007291'

df_postcode.iloc[104, 0] ='L20 4SZ'
df_postcode.iloc[104, 1:3] = [-2.992919, 53.451899]
df_postcode.iloc[104, 3] = "E01007009"

df_postcode.iloc[111, 0] ='L36 9YG'
df_postcode.iloc[111, 1:3] = [-2.839285, 53.412567]
df_postcode.iloc[111, 3] = "E01006481"

df_postcode.iloc[117, 0] ='WA8 0TA'
df_postcode.iloc[117, 1:3] = [-2.721675, 53.364201]
df_postcode.iloc[117, 3] = "E01012441"

df_postcode.iloc[167, 1:3] = [-2.722334, 53.297445]
df_postcode.iloc[167, 3] = "E01018688"

df_postcode.iloc[206, 0] ='L37 1NU'
df_postcode.iloc[206, 1:3] = [-3.081347, 53.562207]
df_postcode.iloc[206, 3] = "E01012441"

df_postcode.iloc[238, 1:3] = [-3.007471, 53.341287]

df_postcode.iloc[289, 1:3] = [-2.941742, 53.469021]

df_postcode.iloc[303, 0] ='L18 9SB'
df_postcode.iloc[303, 1:3] = [-2.905299, 53.370691]
df_postcode.iloc[303, 3] = "E01006680"

df_postcode['LSOA'].fillna('Out of Range', inplace=True)
df.drop(['Longitude', 'Latitude'], axis=1, inplace=True)
df_stores = pd.concat([df, df_postcode], axis = 1)

# Shop Name formatting
temp = df_stores[['Common Name', 'Shop Name']].copy()
temp = temp.apply(lambda x: x.astype(str).str.upper())

temp1 = temp['Shop Name'].str.split(' - | ', expand = True)
temp2 = temp['Common Name'].str.split(' - | ', expand = True)
idx_noName = temp2[temp2[0] != temp1[0]].index
temp.loc[idx_noName, 'Shop Name'] = temp.loc[idx_noName, 'Common Name'] + ' - ' + temp.loc[idx_noName, 'Shop Name']
df_stores['Shop Name'] = temp['Shop Name'].copy()




""" ----------------------------------------------------------------------------------------------------------
Task2:
Add Healthfulness Score from collection [store_type_reference]
----------------------------------------------------------------------------- """
# Add Healthfulness Score
df_healthfulness = getCollection('store_type_reference')
df1 = pd.merge(df_stores, df_healthfulness[['Store Type',
                                     'Tentative Healthfulness Score (range -2 to 2)(C Black et al)']], how = 'left', left_on = 'Store Type', right_on = 'Store Type')

df1.rename(columns = {'Tentative Healthfulness Score (range -2 to 2)(C Black et al)':'Healthfulness'}, inplace = True)



""" ----------------------------------------------------------------------------------------------------------
Task3:
Extract Open hour from GoogleMap Place API
1) Get place_id first
2) USe place_id to request shop details
----------------------------------------------------------------------------- """
apiKey = open(os.path.join('../assets/', 'googlemap_key.txt')).read()
# ---------- get field_id from google map
for i in df1.index:
    print(i)
    latlon = df1.loc[i, 'Latitude'].astype('str') + ',' + df1.loc[i, 'Longitude'].astype('str')
    cName = df1.loc[i, 'Common Name']
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latlon}&radius=100&name={cName}&key={apiKey}"
    fileJson = requests.get(url).json()
    if len(fileJson['results']) > 0:
        if 'place_id' in fileJson['results'][0]:
            df1.loc[i, 'place_id'] = fileJson['results'][0]['place_id']
    else:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latlon}&radius=500&name={cName}&key={apiKey}"
        fileJson = requests.get(url).json()
        if len(fileJson['results']) > 0:
            if 'place_id' in fileJson['results'][0]:
                df1.loc[i, 'place_id'] = fileJson['results'][0]['place_id']

df_places = df1.copy()
df_places['Open_weekday'] = 'N/A'
idx_shop = df_places[df_places['place_id'].isna() == False].index
for idx in idx_shop:
    print('place detail ', idx)
    placeId = df_places.loc[idx, 'place_id']
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={placeId}&fields=opening_hours/weekday_text,price_level&key={apiKey}"
    fileJson = requests.get(url).json()
    if len(fileJson['result']) > 0:
        try:
            df_places.loc[idx, 'Price_leve'] = fileJson['result']['price_level']
        except:
            df_places.loc[idx, 'Price_leve'] = np.nan

        try:
            df_tp = pd.json_normalize(fileJson['result']['opening_hours']).loc[0, 'weekday_text']
            df_places.at[idx, 'Open_weekday'] = df_tp
        except:
            print(idx)


days = ['Monday', 'Tuesday','Wednesday', 'Thursday','Friday','Saturday','Sunday']
cols = ['Opening Time ' + d for d in days]
df_p = df_places.copy()
for idx in df_p.index:
    weekdayText = df_p.loc[idx, 'Open_weekday']
    if weekdayText != 'N/A':
        for iDay in np.arange(0, 7):
            txtSplit = weekdayText[iDay].split()
            if txtSplit[1] == 'Closed':
                df_p.loc[idx, cols[iDay]] = 'Closed'
            elif sum([ikey in txtSplit for ikey in ['24', 'hours']]) == 2:
                df_p.loc[idx, cols[iDay]] = '00:00 - 24:00'
            else:

                t0 = txtSplit[1]
                try:
                    idx_t1 = txtSplit.index('PM')
                    t1 = str(int(txtSplit[idx_t1-1].split(':')[0]) + 12) + ':'+ txtSplit[idx_t1-1].split(':')[1]
                except:
                    t1 = str(int(txtSplit[4].split(':')[0]) + 12) + ':'+ txtSplit[4].split(':')[1]
                df_p.loc[idx, cols[iDay]] = f"{t0} - {t1}"

df_p1 = df_p[cols].copy()
if df_p1[cols].isna().sum().sum() > 0:
    df_p1[df_p1[cols].isna()] = "N/A"
df_p[cols] = df_p1[cols].copy()



""" ----------------------------------------------------------------------------------------------------------
Task4:
Creating opening hour matrix for the heatmap
----------------------------------------------------------------------------- """
# opening hour matrix
def calOpenHour(df0):
    days = ['Monday', 'Tuesday','Wednesday', 'Thursday','Friday','Saturday','Sunday']
    cols = ['Opening Time ' + d for d in days]
    df_store = df0[cols].copy()
    if df_store[cols].isna().sum().sum() > 0:
        df_store[df_store[cols].isna()] = "00:00 - 00:00"
    df_store[df_store[cols] == 'Closed'] = "00:00 - 00:00"
    df_store[df_store[cols] == 'N/A'] = "00:00 - 00:00"

    df = df0.copy()
    df[cols] = df_store[cols].copy()
    df['Hours'] = [np.zeros((7, 24)).astype('int') for _ in range(len(df))]
    for iDay, idx in zip(days, np.arange(0, len(days)).astype('int')):
        newName = iDay
        colName = 'Opening Time ' + iDay
        temp = df[colName].str.replace(' - ', ':', regex='False')
        temp = temp.str.replace('.',':', regex = 'False')
        temp = temp.str.replace(' -',':', regex = 'False')
        temp = temp.str.split(':', expand = True).astype(int)
        #temp["Hours"] = [np.zeros(24) for _ in range(len(temp))]
        for i in temp.index:
            if temp.loc[i, 0] >= temp.loc[i,2]:
                df.loc[i, 'Hours'][idx, temp.loc[i, 0]:temp.loc[i,2]] = 1
            else:
                df.loc[i, 'Hours'][idx, temp.loc[i, 0]:temp.loc[i,2]] = 1

    df[cols] = df0[cols].copy()
    return df

df1 = df_p.copy()
df_store_openHour = calOpenHour(df1)
df_store_openHour['Healthfulness'] = df_store_openHour['Healthfulness'].astype(float, copy=True)
df_store_openHour['Hours'] = [df_store_openHour.loc[i, 'Hours'].tolist() for i in df_store_openHour.index]
keepCols = ['Store Type', 'Common Name', 'Shop Name', 'Address','Postcode', 'Latitude', 'Longitude', 'LSOA',
            'Healthfulness', 'Hours', 'Opening Time Monday', 'Opening Time Tuesday',
            'Opening Time Wednesday', 'Opening Time Thursday', 'Opening Time Friday',
            'Opening Time Saturday', 'Opening Time Sunday', 'place_id', 'Open_weekday']

mylist = df_store_openHour[keepCols].to_dict('records')


""" ---------------------------------------------------------------------------------------------------------
Upload to the database 'Be Careful'
--------------------------------------------------------------------------------------------------------- """
# dropCollection('appShopLatLong')
# createCollection(mylist, 'appShopLatLong')
# print("Collection 'appShopLatLong' updated in MongoDB")

