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
import datetime
from time import sleep, time
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

class GasItem(BaseModel):
    '''
    Use this data model to parse the request body JSON for gas predictions.
    '''
    # ref https://pydantic-docs.helpmanual.io/usage/types/
    # & https://github.com/samuelcolvin/pydantic/blob/master/pydantic/fields.py
    coords: str = Field(..., example = '-122.3321,47.6062;-116.2023,43.6150;-115.1398,36.1699')
    year: int = Field(..., example = 2021)
    month: int = Field(..., gt = 0, le = 12, example = 7)
    day: int = Field(..., gt = 0, le = 31, example = 13)
    mpg: Optional[float] = Field(27.0, gt = 0.0, example = 27.0)

    # ref https://pydantic-docs.helpmanual.io/usage/validators/
    @validator('coords')
    def coords_greater_than_one(cls, v):
        '''Validate there are more than one coordinate passed in'''
        split = v.split(';')
        assert len(split) >= 2, "Not enough coordinates passed in. Ensure coordinates follow the 'long,lat:long,lat' format"
        return v

    @validator('coords')
    def coords_are_paired(cls, v):
        '''Validate coordinates pairs are two values exactly'''
        split = v.split(';')

        split = [tuple(i.split(',')) for i in split]
        
        for pair in split:
            assert len(pair) == 2, f'Coordinate pairs must be exactly 2 values. {pair} has to many or to few values'
        return v

    @validator('coords')
    def coords_are_numeric(cls, v):
        '''Validates that coordinate strings are numeric'''
        split = v.replace(';', ',').split(',')
        for num in split:
            msg = f'Coordinates must be numeric. {num} is not numeric'
            clean_num = num.strip().lstrip('-').replace('.', '')
            assert clean_num.isdecimal(), msg
        return v

    @validator('coords')
    def coords_in_range(cls, v):
        '''Validates coordinates are within -180 and 180 long, -90 and 90 lat'''
        split = v.split(';')
        split = [i.split(',') for i in split]

        for pair in split:
            # This isn't a big deal, because coords_are_paired has already run
            # and if there were anything other than exactly two values, it would
            # have been caught
            assert float(pair[0]) >= -180, f'Longitude must be greater than -180 ({pair[0]}, {pair[1]})'
            assert float(pair[0]) <= 180, f'Longitude must be less than 180 ({pair[0]}, {pair[1]})'
            assert float(pair[1]) >= -90, f'Latitude must be greater than -90 ({pair[0]}, {pair[1]})'
            assert float(pair[1]) <= 90, f'Latitude must be less than 90 ({pair[0]}, {pair[1]})'
        return v

    @validator('coords')
    def coords_in_usa(cls, v):
        '''Validates coordinate pairs are roughly within the continental
        United States'''
        lon = [-124.785, -66.947028]
        lat = [24.446667, 49.384472]
        split = v.split(';')
        split = [i.split(',') for i in split]
        
        for pair in split:
            # coords_are_paired has already run. Pairs are exactly 2
            assert float(pair[0]) > lon[0], f'Longitude is west of the contiguous United States ({pair[0]}, {pair[1]})'
            assert float(pair[0]) < lon[1], f'Longitude is east of the contiguous United States ({pair[0]}, {pair[1]})'
            assert float(pair[1]) > lat[0], f'Latitude is south of the contiguous United States ({pair[0]}, {pair[1]})'
            assert float(pair[1]) < lat[1], f'Latitude is north of the contiguous United States ({pair[0]}, {pair[1]})'
        return v

    @validator('day')
    def day_must_be_in_month(cls, v, values, **kwargs):
        '''Validate that the day is valid for the month''' 
        try:
            datetime.datetime(year = values['year'],
                              month = values['month'], 
                              day = v)
        except:
            m = values['month']
            raise ValueError(f'{v} is too many days for the month: {m}')
        return v

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

@router.post('/predict/gas', tags = ['Predictions'])
async def predict_gas(item: GasItem):
    '''
    Predicts the total cost of gas for a road trip between coordinates within 
    the contiguous United States.

    ### Request Body
    - `coords`: a string of semicolon separated coordinate pairs formated as 
    'long,lat;long,lat;long,lat'. Each coordinate pair represents a stop on the 
    user's road trip.
    - `month`: an integer containing the month of the road trip
    - `day`: an integer containing the day of the road trip
    - `year`: an integer containing the year of the road trip
    - `mpg`: an __optional__ float for the miles per gallon of the vehicle
    being used on the road trip. If no value is passed, 27 mpg will be used as
    a default value

    ### Response
    - `total`: a float with the total cost of of gas predicted for the entire 
    length of the trip.
    '''
    month = item.month
    day = item.day
    year = item.year
    meter_to_mile = 0.00062137119224
    mpg = item.mpg
    total = 0
    distance_in_region = split_by_region(item.coords)

    resp = {}

    for i, distance in enumerate(distance_in_region['distances']):
        region = distance_in_region['regions'][i]
        miles = distance * meter_to_mile
        regional_rate = region_gas_predictions(region, month, day, year)
        total += (miles / mpg) * regional_rate

    resp['total'] = round(total, 2)
    return resp

# @router.get('/test')
# async def test():
#     '''
#     A very simple test get request used for ad hoc testing. Good for helper
#     functions
#     TODO: Remove once branch is complete
#     '''

#     return str(split_by_region('-122.3321,47.6062;-116.2023,43.6150;-115.1398,36.1699'))

class RateLimiter():
    '''
    A helper class for respecting api rate limits and taking into account 
    runtime. Best practice to create a new instance for each endpoint you are
    querying.

    ### Params
    - `rate`: an integer representing the number of calls allowed per minute
    - `endpoint`: a string containing the endpoint name. I recommend putting 
    the endpoint url here for clarity. But basically anything to help you 
    remember.
    '''
    def __init__(self, rate, endpoint = 'RateLimiter'):
        self._calls = 0
        self._timer = 0
        self.rate  = rate
        self.endpoint = endpoint

    def call(self):
        '''
        Logs a call to the endpoint and verifies the total calls for the last 
        minute are under the api's rate limit. Sleeps to respect the ratelimit.
        '''

        self._calls += 1

        if self._timer == 0:
            self._timer = time()

        if self._calls >= self.rate:
            elapsed = time() - self._timer
            if elapsed <= 60.0:
                remaining = 60.0 - elapsed
                sleep(remaining)
            self._calls = 0
            self._timer = time()

def coord_to_state(coord):
    '''
    A helper function that converts coordinates into state names using the 
    MapBox api. USA coordinates only.

    ### Params
    - `coord`: a tuple of floats representing a (long, lat) pair of 
    geocoordinates

    ### Returns
    - A string with the name of the state the coordinates are within
    '''
    base_url = 'https://api.mapbox.com/geocoding/v5/mapbox.places/'
    token = os.environ.get('MAPBOX_TOKEN')
    constructed_url = base_url + str(coord[0]) + ',' + str(coord[1]) + '.json?access_token=' + token
    
    # 5 tries, with back off, for 500, 502, 503, 504, no custom mapbox errors
    tries = 5
    backoff_factor = .3
    backoff = backoff_factor * (2 ** (tries - 1)) #4.8 seconds
    wait = 0.0
    for i in range(tries):
        resp = requests.get(constructed_url)

        if resp.status_code in [500, 502, 503, 504]:
            print(f'Mapbox geocoding endpoint down retry #{i}')
            sleep(wait)
            wait += backoff
            continue
        else:
            resp = resp.json()['features']

    # response contains multiple features, state names stored as 'region'
    for feature in resp:
        if 'region' in feature['place_type']:
            sleep(0.001666) # 600 per minute rate limit
            return feature['text']
    
    sleep(0.001666) # 600 per minute rate limit
    return 'state not found'

def coord_to_region(coord):
    '''
    A helper function that takes a coordinate pair and returns the appropriate
    PADD region identifier key, ie 1a, 3, 1c, etc

    ### Params
    - `coord`: a tuple of floats representing a (long, lat) pair of 
    geocoordinates

    ### Returns
    - A string with the PADD region identifier
    '''
    state = coord_to_state(coord)
    for key in PADDS:
        if state in PADDS[key]:
            return key
        
    return 'Not in padds'

def region_gas_predictions(region, month, day, year):
    '''
    A helper function that takes a PADD region and the date and returns the 
    predicted price per gallong for gas

    ### Params
    - `region`: A string containing the PADD code for the region
    - `month`: an integer with the numeric month
    - `day`: an integer with the numeric day in the month
    - `year`: an integer with the four digit year

    ### Returns
    - a float representing the price per gallon for gasoline in that region on
    that date
    '''
    # TODO: check for missing regions or 'Not in padds'
    # TODO: validate month, day, year for inappropriate input
    # TODO: Throw a 500 error, and meaningful error log
    return GAS_MODELS[region].predict([[month, day, year]])[0]

def split_by_region(coords):
    '''
    A helper function that takes the entire route, and splits it into sections
    by PADD region. Returns a dictionary of lists with corresponding regions 
    and distance traveled in each region. 
    
    ### Params
    - `coords`: a string with long,latitude pairs separated by semicolons. 
    formatted like this:'-122.3321,47.6062;-116.2023,43.6150;-115.1398, 36.1699'

    ### Returns
    - a dictionary containing meters traveled in a region, and the corresponding
    region. Formatted like this:
    {
    'distances': [12.4, 40.9, 400.4],
    'regions': ['5', '4', '5']
    }
    '''
    route = {'coordinates': coords,
            'steps': 'true'}

    token = os.environ.get('MAPBOX_TOKEN')
    url = 'https://api.mapbox.com/directions/v5/mapbox/driving?access_token='
    url += token

    trip = requests.post(url, data = route)

    distance = 0
    prev_reg = 0
    cur_reg = 0
    reg_dist_map = {'distances': [], 'regions': []}

    # TODO: rethink/optimize this some. It takes 5 seconds to run locally.

    # a route is made up of multiple legs determined by destinations
    for leg in trip.json()['routes'][0]['legs']:
        # legs are made up of steps it takes to travel the leg
        for step in leg['steps']:
            prev_reg = cur_reg
            
            # collect distance traveled
            distance += step['distance']
            
            # check the end location of each step.manuever
            # TODO is 'intersections' really the appropriate place to look?
            # explore some more
            coords = step['intersections'][-1]['location']
            
            cur_reg = coord_to_region(coords)
            
            if prev_reg == 0:
                prev_reg = cur_reg
            elif cur_reg != prev_reg:
                reg_dist_map['distances'].append(distance)
                distance = 0
                reg_dist_map['regions'].append(prev_reg)

    reg_dist_map['distances'].append(distance)
    reg_dist_map['regions'].append(prev_reg)

    return reg_dist_map