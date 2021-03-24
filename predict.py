import re
import spacy
import joblib
import logging
import numpy as np
import pandas as pd
from joblib import load
# from collections import Counter
# nlp = spacy.load('en-core-web-sm')
# import en_core_web_sm
# nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import CountVectorizer
# from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()

# class Item(BaseModel):
#     """Use this data model to parse the request body JSON."""

#     x1: float = Field(..., example=3.14)
#     x2: int = Field(..., example=-42)
#     x3: str = Field(..., example='banjo')

#     def to_df(self):
#         """Convert pydantic object to pandas dataframe with 1 row."""
#         return pd.DataFrame([dict(self)])

#     @validator('x1')
#     def x1_must_be_positive(cls, value):
#         """Validate that x1 is a positive number."""
#         assert value > 0, f'x1 == {value}, must be > 0'
#         return value

classifier = joblib.load("assets/classifier.pkl")
tfidfVectorizer = joblib.load("assets/tfidfVectorizer.pkl")

stops = {'cannot', 'sometime', 'put', 'take', 'whereas', 'had', 'which', 'even', 'see', 'those', 'can', 'another', 'five', 'down', 'myself', 'whether', 'an', 'less', 'because', 'be', 'thereupon', 'forty', 'did', 'due', 'any', 'thus', 'when', 'into', 'might', 'eleven', 'hereafter', 'over', 'does', 'quite', 'same', 'while', 'herein', 'own', 'ca', 'keep', 'hers', 'sometimes', "n't", 'everything', 'becoming', 'give', 'until', '‘m', 'now', 'ours', 'very', 'yourselves', "'re", 'by', '’ll', 'against', 'hereupon', 'these', 'whence', 'three', 'someone', '’re', 'the', 'here', 'then', 'onto', 'why', 'whom', '‘s', 'them', 'they', 'herself', 'something', 'else', 'am', 'get', 'just', 'whole', 'done', 'n’t', '’ve', 'twelve', 'seeming', 'thru', 'whenever', 'has', 'however', 'between', 'no', 'our', 'beforehand', 'wherever', 'became', '‘ll', 'first', 'mine', 'last', 'ourselves', 'six', 'their', 'i', 'than', 'formerly', 'therefore', 'do', '’s', 'move', 'otherwise', 'may', 'only', 'but', 'seemed', 're', 'whose', 'one', 'among', 'become', 'how', 'show', 'back', 'from', 'regarding', 'yet', 'every', 'fifteen', 'made', 'where', "'m", "'ve", 'although', 'so', 'beyond', 'more', 'none', 'nevertheless', 'under', 'about', 'it', 'others', 'yourself', 'afterwards', 'nothing', 'below', 'latterly', 'within', 'front', 'for', 'namely', 'ever', 'part', 'whereafter', 'amongst', 'eight', 'top', 'whoever', 'off', 'seems', 'before', 'been', 'have', 'always', 'his', 'alone', 'few', 'around', 'hereby', 'he', 'everyone', 'latter', 'nobody', 'nowhere', 'other', 'further', 'except', 'ten', 'thence', 'fifty', 'whither', 'your', 'meanwhile', 'would', 'some', "'s", 'are', 'thereafter', 'thereby', 'empty', 'various', '’m', 'serious', 'please', 'him', 'behind', 'who', 'next', 'several', 'third', 'itself', 'was', 'at', 'through', 'must', 'amount', 'besides', 'throughout', 'anyone', 'if', 'as', 'go', 'elsewhere', 'were', 'after', 'each', 'during', 'bottom', 'everywhere', 'a', 'say', 'using', 'should', 'she', 'somehow', 'though', 'up', '‘ve', 'again', 'n‘t', 'both', 'many', 'to', 'full', 'nor', "'ll", 'per', '‘re', 'two', 'out', 'all', 'somewhere', 'along', 'its', 'side', 'therein', 'unless', 'whatever', 'yours', 'also', 'anyway', 'enough', 'being', 'moreover', 'noone', 'that', 'above', 'well', 'my', 'towards', 'sixty', 'really', 'already', 'such', 'make', 'via', 'wherein', "'d", 'four', 'since', 'becomes', 'and', 'too', 'themselves', 'toward', 'could', '‘d', 'you', 'not', 'seem', 'twenty', 'never', 'anything', 'almost', 'doing', 'anywhere', 'himself', 'most', 'this', 'once', 'perhaps', 'still', 'anyhow', 'either', 'much', 'will', 'without', 'rather', 'what', 'upon', 'hundred', 'of', 'often', 'together', 'whereby', 'in', 'beside', 'former', 'there', 'is', 'me', 'nine', 'across', 'used', 'neither', 'or', 'we', '’d', 'mostly', 'indeed', 'on', 'hence', 'least', 'name', 'call', 'with', 'whereupon', 'her', 'us'}

def tokenize(text):
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    words = letters_only.lower().split()                                              
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words ))


def get_prediction(input):
    array1 = []
    cleaned1 = tokenize(input)
    array1.append(cleaned1)

    x_tfid = tfidfVectorizer.transform(array1).toarray()
    answer = classifier.predict(x_tfid)
    answer = str(answer[0])
    return answer

@router.post('/predict')
async def predict(item: str):
    """
    ## How to use:
    * Click "try it out."
    * Enter various items in the kickstart campaign needed
    * Monetary goal, time live, etc
    * This will give a response of whether or not the campaign is likely to succeed
    ## Needed Info:
    - `item`: item1
    ## Response:
    - Whether or not the kickstarter is likely to be a success or not.
    """

    success_failure = get_prediction(item)
    return {
        'success_failure' : success_failure
    }
