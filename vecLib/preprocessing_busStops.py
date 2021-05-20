from connectMongoDB import getCollection, createCollection, dropCollection
import pandas as pd
import numpy as np
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# --------------------------
# collection: StopsLiverpool
# ---------------------------
def create_appStopsLiverpool(df_lsoa):
    collName = 'StopsLiverpool'
    df = getCollection(collName)
    keepCols = ['CommonName', 'Longitude', 'Latitude' , 'Landmark']
    df = df[keepCols].copy()

    # add col of LSOA
    for i in df_lsoa.index:

        print('{} over {}'.format(i + 1, df_lsoa.shape[0]))
        coords = df_lsoa.loc[i, 'geometry.coordinates']

        if coords.__len__() == 1:
            poly = Polygon(coords[0])
            # plt.plot(*poly.exterior.xy)
            for ii in df.index:
                point = Point(df.loc[ii, ['Longitude']], df.loc[ii, ['Latitude']])
                if point.within(poly):
                    df.loc[ii, 'LSOA'] = df_lsoa.loc[i, 'properties.lsoa11cd']

                # #xs = [p.x for p in point]
                # #ys = [p.y for p in point]
                # plt.scatter(point.x, point.y)
        # plt.show()
        else:
            for i1 in range(0, coords.__len__()):
                poly = Polygon(coords[i1][0])
                for ii in df.index:
                    point = Point(df.loc[ii, ['Longitude']], df.loc[ii, ['Latitude']])
                    if point.within(poly):
                        df.loc[ii, 'LSOA'] = df_lsoa.loc[i, 'properties.lsoa11cd']

    mylist = df.to_dict('records')
    dropCollection('appStopsLiverpool')
    createCollection(mylist, 'appStopsLiverpool')
    print("Collection 'appStopsLiverpool' has uploaded to MongoDB")


with open('../assets/liverpoolmini.json', 'r') as f:
    data = json.loads(f.read())
# Flatten data
df_lsoa = pd.json_normalize(data, record_path = 'features', meta=['name'])
create_appStopsLiverpool(df_lsoa)