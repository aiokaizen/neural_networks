### Objective

The AI should be able to analyze a certain period in the past, and then be able to predict the
future of the market by providing a value from -1 to 1. -1 being a very strong sell and a 1 being a very
strong purchase, 0 signifies uncertainty, or no action.


### Notes

LSTM: Long Short Term Memory - A neural network that stores memory about the previous inputs.
    This type of NN is really good for financial data, or any type of sequential data.
    It works by taking information about the past and using it to predict the future.

Sliding windows: each window has 8 days for example, each window is split into 5 days
    that we want to predict from and a 3 days that we want to predict.

The volume is so large compared to other data entries like open, close, ... therefore, it has
to be scaled down for convenience.

#### Inputs:
- trading balance
- spread value
- market symbol

- (open, close, high, low, volume) of the past x days (defaults to:    60)    # 60 days
- (open, close, high, low, volume) of the past x hours (defaults to:   240)   # 10 days
- (open, close, high, low, volume) of the past x minutes (defaults to: 1440)  # 1 day
- (open, close, high, low, volume) of the past x seconds (defaults to: 3600)  # 1 hour


# Using sliding windows

With sliding windows, we will try a window of 13 days, 10 days that we want to predict
from, and a 3 days that we want to predict.

- (open, close, high, low, volume) of the past x days (defaults to:    10)
- (open, close, high, low, volume) of the past x hours (defaults to:   600)    # 10 days
- (open, close, high, low, volume) of the past x minutes (defaults to: 36000)  # 10 days
- (open, close, high, low, volume) of the past x seconds (defaults to: 86400)  # 1 day



### Tutorials:

- https://www.youtube.com/watch?v=e4PsaCF2odk
