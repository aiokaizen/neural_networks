import blankly
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.optim as optim
from torch.autograd import Variable


def episode_gen(data, seq_length, output_size):

    """ e.g. episode_gen(data, 8, 3) """

    x = []
    y = []

    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length - output_size)]
        _y = data[(i + seq_length - output_size):(i + seq_length)]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def init_nn(symbol, state):

    interface = state.interface
    resolution = state.resolution
    variables = state.variables

    variables['bars'] = interface.history(symbol, 300, resolution)
    variables['prices'] = variables['bars']['close'].to_list()

    # Indicators RSI / MACD
    rsi = blankly.indicators.rsi(state.variables['prices'])
    macd = blankly.indicators.macd(state.variables['prices'])

    # Episode generation parameters
    seq_length = 8
    output_size = 3

    # 25 is used because MACD needs 26 data points to calculate ( we start from 0)
    x = [
        variables['prices'][i] / variables['prices'][i - 1]
        for i in range(25, len(variables['prices']))
    ]
    x, y = episode_gen(x, seq_length, output_size)
    y = Variable(torch.Tensor(y)).unsqueeze(0)

    # RSI data gathering. RSI is calculated across 14 day periods
    x_rsi = rsi[11:]
    x_rsi, _ = episode_gen(x_rsi, seq_length, output_size)

    # MACD data gathering
    macd_vals, _ = episode_gen(macd[0], seq_length, output_size)
    macd_signals, _ = episode_gen(macd[1], seq_length, output_size)

    # Volume
    volumes = variables['bars']['volume'].to_list()
    state.max_vol_scaler = max(volumes)
    state.min_vol_scaler = min(volumes)
    vol, _ = episode_gen(volumes[25:], seq_length, output_size)

    # 5 represents the number of features we have:
    # prices, rsi, macd_vals, macd_signals, volumes
    data_size = len(macd_signals)
    x_agg = np.zeros((data_size, seq_length - output_size, 5))

    for i in range(len(data_size)):
        for j in range(seq_length - output_size):
            x_agg[i][j][0] = x[i][j]
            x_agg[i][j][1] = x_rsi[i][j]
            x_agg[i][j][2] = macd_vals[i][j]
            x_agg[i][j][3] = macd_signals[i][j]

            # Normalizing the volume
            x_agg[i][j][4] = (
                (vol[i][j] - state.min_vol_scaler) / (state.max_vol_scaler - state.min_vol_scaler)
            )

    x_tot = Variable(torch.Tensor(x_agg))

    num_epochs = 15000
    learning_rate = 0.0002

    state.lstm = LSTM(5, 25, batch_first=True)
    state.lin = nn.Linear(25, 3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': state.lstm.parameters()},
        {'params': state.lin.parameters()}
    ], lr=learning_rate)

    for epoch in range(num_epochs):
        outputs, (h_n, c_n) = state.lstm(x_tot)
        out = state.lin(h_n)
        # Sigmoid function returns values between 0 and 1, we add 0.5 to
        # shift it to 0.5 and 1.5. This allows us to go bellow the min value,
        # while also staying in the positive range.
        out = 0.5 + F.sigmoid(out)

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()  # Backward propagation
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch}, loss: {loss.item()}")

    # state.lstm.load_state_dict(torch.load('lstm_pm.pth'))
    # state.lin.load_state_dict(torch.load('lin_pm.pth'))
    torch.save(state.lstm.state_dict(), 'lstm_bar.pth')
    torch.save(state.lin.state_dict(), 'lin_bar.pth')

    '''
        We use this in the trading algorithm for more stability.
        Essentially, instead of relying in a single output of the model
        to tell us whether to buy or sell, we average the readings from different
        calculations (3 days before, 2 days before, a day before)
    '''

    state.lastthree = [[0, 0], [0, 0], [0, 0]]

def bar_lstm(bar, symbol, state):
    new_bar = pd.DataFrame([bar])
    state.variables['bars'] = pd.concat([state.variables['bars'], new_bar], ignore_index=True)
    state.variables['history'].append(bar['close'])  # Add latest price to current list of data

    into = [state.variables['history'][i] / state.variables['history'][i - 1] for i in range(-5, 0)]

    rsi = blankly.indicators.rsi(state.variables['history'])
    rsi_in = np.array(rsi[-5:])

    macd = blankly.indicators.macd(state.variables['history'])
    macd_vals = np.array(macd[0][-5:])
    macd_signals = np.array(macd[1][-5:])

    volume = state.variables['bars']['volume'].to_list()[-5:]

    pred_in = np.zeros((1, len(into), 5))
    for i in range(len(into)):
        pred_in[0][i][0] = into[i]
        pred_in[0][i][0] = rsi_in[i]
        pred_in[0][i][0] = macd_vals[i]
        pred_in[0][i][0] = macd_signals[i]
        pred_in[0][i][0] = volume[i] / state.max_vol_scaler

    pred_in = torch.Tensor(pred_id)

    # Run the data through the trained model.
    # The field 'out' stores the prediction values we want.
    out, (h, c) = state.lstm(pred_in)
    out = state.lin(h)
    out = 0.5 + F.sigmoid(out)

    # The logic of averaging across three days
    state.lastthree[0][0] += out[0][0][0]
    state.lastthree[0][1] += 1

    state.lastthree[1][0] += out[0][0][1]
    state.lastthree[1][1] += 1

    state.lastthree[2][0] += out[0][0][2]
    state.lastthree[2][1] += 1

    # The price increase is calculated by dividing the sum of next day predictions
    # by the number of predictions for the next day.
    price_avg = state.lastthree[0][0] / state.lastthree[0][1]

    curr_value = blanky.trunc(
        state.interface.account[state.base_asset].available, 2
    )
    if price_avg > 1:
        # If we think price will increase, we buy
        buy = blankly.trunc(
            state.interface.cash * 2 * (price_avg.item() - 1) / bar['close'], 2
        )
        if buy > 0:
            state.interface.market_order(symbol, side='buy', size=buy)
    elif curr_value > 0:
        # If we think price will decrease, we sell
        cv = blankly.trunc(
            curr_value * 2 * (1 - price_avg.item()), 2
        )
        if cv > 0:
            state.interface.market_order(symbol, side='sell', size=cv)

    print("price prediction:", price_avg)

    # Shift the values of our last three days
    state.lastthree = [
        state.lastthree[1],
        state.lastthree[2],
        [0, 0]
    ]

def test():
    from blankly.exchanges.interfaces.ftx.ftx import FTX
    # exchange = blankly.FTX()
    exchange = FTX()
    strategy = blankly.Strategy(exchange)
    strategy.add_bar_event(
        bar_lstm, symbol='ETH-USD', resolution='1d', init=init_nn
    )
    # Backtest to one year starting with
    result = strategy.backtest(to="1y", initial_values={'USD': 10000})
    print(result)

if __name__ == '__main__':
    test()
