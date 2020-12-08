import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
####################For Gas Models##############################################
from typing import Optional
import pickle
import os
from sklearn.linear_model import LinearRegression
import requests
################################################################################

log = logging.getLogger(__name__)
router = APIRouter()

GAS_MODELS = {}
PADDS = {'1a': 
             ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 
             'Connecticut', 'Rhode Island'],
         '1b':
             ['New York', 'New Jersey', 'Pennsylvania', 'Deleware', 'Maryland'],
         '1c':
             ['West Virginia', 'Virginia', 'North Carolina', 'South Carolina',
             'Georgia', 'Florida'],
         '2':
             ['North Dakota', 'South Dakota', 'Nebraska', 'Kansas', 'Oklahoma',
             'Minnesota', 'Iowa', 'Missouri', 'Wisconsin', 'Illinois', 
             'Tennessee', 'Kentucky', 'Indiana', 'Ohio', 'Michigan'],
         '3':
             ['New Mexico', 'Texas', 'Arkansas', 'Louisiana', 'Mississippi', 
             'Alabama'],
         '4':
             ['Idaho', 'Utah', 'Montana', 'Wyoming', 'Colorado'],
         '5':
             ['Washington', 'Oregon', 'California', 'Nevada', 'Arizona', 
             'Alaska', 'Hawaii']
        }

class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: float = Field(..., example=3.14)
    x2: int = Field(..., example=-42)
    x3: str = Field(..., example='banjo')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('x1')
    def x1_must_be_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'x1 == {value}, must be > 0'
        return value

class GasItem(BaseModel):
    '''
    Use this data model to parse the request body JSON for gas predictions.
    '''

    coords: str = Field(..., example='long,lat;long,lat;long,lat')
    month: int = Field(..., example = 7)
    day: int = Field(..., example = 13)
    year: int = Field(..., example = 2021)
    mpg: Optional[float] = None

    # TODO: Put in validators here. We are doing this live for first attempt

@router.on_event('startup')
async def load_models():
    '''
    Loading all the gas models into the global space on startup.
    '''
    global GAS_MODELS

    # os safe file paths
    # docker-compose pwd is /usr/src/app. then our app is another directory up
    # for the less than intuitive file path /usr/src/app/app/gas_models/
    newengland = os.path.join(os.getcwd(),'app', 'gas_models', 'new_england_gas_model.pckl')
    centralatlantic = os.path.join(os.getcwd(),'app', 'gas_models', 'central_atlantic_gas_model.pckl')
    loweratlantic = os.path.join(os.getcwd(),'app', 'gas_models', 'lower_atlantic_gas_model.pckl')
    midwest = os.path.join(os.getcwd(),'app', 'gas_models', 'midwest_gas_model.pckl')
    gulfcoast = os.path.join(os.getcwd(),'app', 'gas_models', 'gulf_coast_gas_model.pckl')
    rockymnt = os.path.join(os.getcwd(),'app', 'gas_models', 'rocky_mountain_gas_model.pckl')
    westcoast = os.path.join(os.getcwd(),'app', 'gas_models', 'west_coast_gas_model.pckl')

    GAS_MODELS['1a'] = pickle.load(open(newengland, 'rb'))
    GAS_MODELS['1b'] = pickle.load(open(centralatlantic, 'rb'))
    GAS_MODELS['1c'] = pickle.load(open(loweratlantic, 'rb'))
    GAS_MODELS['2'] = pickle.load(open(midwest, 'rb'))
    GAS_MODELS['3'] = pickle.load(open(gulfcoast, 'rb'))
    GAS_MODELS['4'] = pickle.load(open(rockymnt, 'rb'))
    GAS_MODELS['5'] = pickle.load(open(westcoast, 'rb'))

@router.post('/predict')
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `x1`: positive float
    - `x2`: integer
    - `x3`: string

    ### Response
    - `prediction`: boolean, at random
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """

    X_new = item.to_df()
    log.info(X_new)
    y_pred = random.choice([True, False])
    y_pred_proba = random.random() / 2 + 0.5
    return {
        'prediction': y_pred,
        'probability': y_pred_proba
    }

@router.post('/predict/gas')
async def predict_gas(item: GasItem):
    '''
    Predicts the total cost of gas for a road trip between coordinates. 

    ### Request Body
    - 'coords': a string of semicolon separated coordinate pairs formated as 
    'long,lat;long,lat;long,lat'. Each coordinate pair represents a stop on the 
    user's road trip.
    - 'month': an integer containing the month of the road trip
    - 'day': an integer containing the day of the road trip
    - 'year': an integer containing the year of the road trip
    - 'mpg': an optional float specifying the miles per gallon of the vehicle
    being used on the road trip. If no value is passed, 27 mpg will be used as
    a default value

    ### Response
    - 'prediction': a float with the total cost of of gas for the entire length
    of the trip.
    '''

@router.get('/test')
async def test():
    '''
    A very simple test get request used for ad hoc testing. 
    TODO: Remove once branch is complete
    '''

    return coord_to_state((-111.0429, 45.6770))

def coord_to_state(coord):
    '''
    A helper function that converts coordinates into state names using the 
    MapBox api. USA coordinates only.

    ### Params
    - coord: a tuple of floats representing a long, lat geocoordinates

    ### Returns
    - A string with the name of the state the coordinates are within
    '''
    base_url = 'https://api.mapbox.com/geocoding/v5/mapbox.places/'
    token = os.environ.get('MAPBOX_TOKEN')
    constructed_url = base_url + str(coord[0]) + ',' + str(coord[1]) + '.json?access_token=' + token
    
    # search for the region feature
    resp = requests.get(constructed_url).json()['features']

    # TODO: add retry logic incase this service is down
    # TODO: Consider rate limit logic here
    # TODO: try except for coords outside of the US
    
    for feature in resp:
        if 'region' in feature['place_type']:
            return feature['text']
    
    return 'state not found'