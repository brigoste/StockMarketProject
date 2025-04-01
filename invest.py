import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Subfunctions
def eval_buy(price, day, buy_interval, alpha, local_min):
    if buy_interval < 1:
        buy_interval = 1
    if day % buy_interval == 0:
        if price > (1+alpha)*local_min:
            return True
        else:
            return False
    else:
        return False
    
def eval_sell(price, day, sell_interval, beta, local_max):
    if sell_interval < 1:
        sell_interval = 1
    if day % sell_interval == 0:
        if price < (1-beta)*local_max:
            return True
        else:
            return False
    else:
        return False
    
def eval_risk(price, risk_ratio):
    # TODO
    return
    
def buy_shares(price, money_holding, shares_holding, trading_expenses, total_trades, buy_ratio, risk_ratio):
    if money_holding <= 0:
        return money_holding, trading_expenses, shares_holding, total_trades
    else: 
        shares_to_purchase = money_holding / price
        money_holding -= buy_ratio*shares_to_purchase*price
        trading_expenses += buy_ratio*shares_to_purchase*price
        shares_holding += buy_ratio*shares_to_purchase
        total_trades += 1
        return money_holding, trading_expenses, shares_holding, total_trades

def sell_shares(price, money_holding, shares_holding, trading_revenue, total_trades, sell_ratio, risk_ratio):
    shares_to_sell = shares_holding
    money_holding += sell_ratio*shares_to_sell*price
    trading_revenue += sell_ratio*shares_to_sell*price
    shares_holding -= sell_ratio*shares_to_sell
    total_trades += 1
    return money_holding, trading_revenue, shares_holding, total_trades

def update_moving_average(prices, day, column, mov_avg_interval):
    if day == 0:
        moving_average = prices[day, column]
    elif day < mov_avg_interval:
        moving_average = np.mean(prices[0:day, column])
    else:
        moving_average = np.mean(prices[day-int(mov_avg_interval):day, column])

    return moving_average

def update_market_direction(price, prev_price, direction, prev_direction):
    prev_direction = direction
    if price > prev_price:
        direction = 1
    elif price < prev_price:
        direction = 0
    
    return direction, prev_direction

def update_local_min(price, prev_price, direction, prev_direction, local_min):    
    if direction == 1 and prev_direction == 0:
        local_min = prev_price
    return local_min, direction 
    
def update_local_max(price, prev_price, direction, prev_direction, local_max):
    if direction == 0 and prev_direction == 1:
        local_max = prev_price
    return local_max, direction
        
def update_savings(money_holding, day, weekly_invest):
    if day % 7 == 0:
        money_holding += weekly_invest
    return money_holding

def dca_buy(price, day, dca_money_holding, dca_shares_holding, dca_trading_expenses, dca_total_trades, dca_buy_interval, dca_rate):
    if dca_money_holding <= dca_rate:
        return dca_money_holding, dca_trading_expenses, dca_shares_holding, dca_total_trades
    else:
        if day % dca_buy_interval == 0:
            if price == 0:
                print("PRICE ERROR")
                dca_shares_to_purchase = 0
            else: 
                dca_shares_to_purchase = dca_rate / price
            dca_money_holding -= float(dca_rate)
            dca_trading_expenses += float(dca_rate)
            dca_shares_holding += float(dca_shares_to_purchase)
            dca_total_trades += 1
        return dca_money_holding, dca_trading_expenses, dca_shares_holding, dca_total_trades

def liquidate_shares(price, money_holding, shares_holding, trading_revenue, total_trades):
    money_holding += shares_holding*price
    trading_revenue += shares_holding*price
    shares_holding -= shares_holding
    total_trades += 1
    return money_holding, trading_revenue, shares_holding, total_trades

def investing_scenario(x, price_data, vars, alph=0.1, bet=0.1, buy_rat=1.0, sell_rat=0.0, mov_avg_int=1, buy_int=1, sell_int=1, buy_col=3, sell_col=2, risk_rat=0.0, week_inv=100):
    
    # vars is a vector that defines the indeces of the variables in the x vector to be optimized
    # Initialize vector to hold the optimized variables
    v = np.zeros(10)
    v[0] = alph
    v[1] = bet
    v[2] = buy_rat
    v[3] = sell_rat
    v[4] = risk_rat
    v[5] = mov_avg_int
    v[6] = buy_int
    v[7] = sell_int
    v[8] = buy_col
    v[9] = sell_col
    # Replace the variables to be optimized with the optimized variables
    v[vars[0]] = x[0]
    v[vars[1]] = x[1]
    # Pass out the variables to the investment strategy
    alph = v[0]
    bet = v[1]
    buy_rat = v[2]
    sell_rat = v[3]
    risk_rat = v[4]
    mov_avg_int = int(np.round(v[5]))
    buy_int = int(np.round(v[6]))
    sell_int = int(np.round(v[7]))
    buy_col = int(np.round(v[8])) 
    sell_col = int(np.round(v[9]))
    
    # reset variables for the next iteration
    buy_dir = 0 # 0 for bearish and 1 for bullish
    prev_buy_dir = 0 # 0 for bearish and 1 for bullish
    sell_dir = 0 # 0 for bearish and 1 for bullish
    prev_sell_dir = 0 # 0 for bearish and 1 for bullish
    loc_min = price_data[0,buy_col] # minimum value of the stock price
    loc_max = price_data[0,sell_col] # maximum value of the stock price
    buy_avg = price_data[0,buy_col] # opening price of the first day
    sell_avg = price_data[0,sell_col] # opening price of the first day
    
    money = 100 # amount of money in the wallet available for investment
    shares = 0 # amount of shares in the portfolio available for selling
    revenue = 0 # profit from the investment strategy
    expenses = 0 # expenses from the investment strategy
    profit = 0 # total profit from the investment strategy
    trades = 0 # total number of trades made in the investment strategy
    
    for d in range(price_data.shape[0]-1): # iterate through the rows of the data for number of days
        prev_buy_avg = buy_avg
        prev_sell_avg = sell_avg
        if d > 0:
            buy_avg = update_moving_average(price_data, d, buy_col, mov_avg_int)
            sell_avg = update_moving_average(price_data, d, sell_col, mov_avg_int)
            buy_dir, prev_buy_dir = update_market_direction(buy_avg, prev_buy_avg, buy_dir, prev_buy_dir)
            sell_dir, prev_sell_dir = update_market_direction(sell_avg, prev_sell_avg, sell_dir, prev_sell_dir)
            loc_min, buy_dir = update_local_min(buy_avg, prev_buy_avg, buy_dir, prev_buy_dir, loc_min)
            loc_max, sell_dir = update_local_max(sell_avg, prev_sell_avg, sell_dir, prev_sell_dir, loc_max)   
                            
        if eval_buy(price_data[d,buy_col], d, buy_int, alph, loc_min):
            money, expenses, shares, trades = buy_shares(price_data[d,buy_col], money, shares, expenses, trades, buy_rat, risk_rat)
            
        if eval_sell(price_data[d,sell_col], d, sell_int, bet, loc_max):
            money, revenue, shares, trades = sell_shares(price_data[d,sell_col], money, shares, revenue, trades, sell_rat, risk_rat)
            
        money = update_savings(money, d, week_inv)
                
    # calculate the total profit from the investment strategy
    money, revenue, shares, trades = liquidate_shares(price_data[d,sell_col], money, shares, revenue, trades)
    profit = -(revenue - expenses)

    # print("P: ", f"{profit:.2f}", "T: ", trades, "A: ", alph, " B: ", bet, " BR: ", buy_rat, " SR: ", sell_rat, " RR: ", risk_rat, " BI: ", buy_int, " SI: ", sell_int, " BC: ", buy_col, " SC: ", sell_col)
             
    return profit
              
              
def dca_scenario(price_data, buy_col, sell_col, buy_rat, sell_rat, risk_rat, dca_int, week_inv):
    
    dca_money = 100 # amount of money in the wallet available for investment
    dca_shares = 0 # amount of shares in the portfolio available for selling
    dca_revenue = 0 # profit from the investment strategy
    dca_expenses = 0 # expenses from the investment strategy
    dca_profit = 0 # total profit from the investment strategy
    dca_trades = 0 # total number of trades made in the investment strategy
    
    dca_inv = week_inv*(dca_int/7) # average investment per dca interval
    
    for d in range(price_data.shape[0]-1): # iterate through the rows of the data for number of days

        dca_money, dca_expenses, dca_shares, dca_trades = dca_buy(price_data[d,buy_col], d, dca_money, dca_shares, dca_expenses, dca_trades, dca_int, dca_inv)
        dca_money = update_savings(dca_money, d, week_inv)
                            
    # calculate the total profit from the investment strategy
    dca_money, dca_revenue, dca_shares, dca_trades = liquidate_shares(price_data[d,sell_col], dca_money, dca_shares, dca_revenue, dca_trades)
    dca_profit = dca_revenue - dca_expenses
    
    return dca_profit, dca_trades
