import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
from typing import Optional

log = logging.getLogger(__name__)
router = APIRouter()


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
    A very simple test get request
    '''

    return 'Test worked'