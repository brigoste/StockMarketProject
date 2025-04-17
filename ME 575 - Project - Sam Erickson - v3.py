import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import FiniteDifference as FD
import matplotlib.pyplot as plt
import GradientFree as gf
from scipy.optimize import minimize as opt
import os

import warnings
warnings.filterwarnings('ignore')

# global price_data


# Critical Values
alpha_array = np.linspace(0.01, 0.1, 10)
beta_array = np.linspace(0.1, 0.01, 10)
buy_ratio_array = np.linspace(1.0, 0.01, 100)
sell_ratio_array = np.linspace(0.0, 0.1, 11)
risk_ratio_array = np.linspace(0.1, 1.0, 10) # percentage of acceptable risk relative to bonds or expected return
buy_interval_array = np.linspace(1, 7, 7) # number of days between buying and selling evaluations
sell_interval_array = np.linspace(1, 7, 7) # number of days between selling and buying evaluations
moving_average_array = np.linspace(0, 21, 1) # number of days for the moving average calculation       21
weekly_investment = 100 # $ per week added to the investment portfolio
dca_investment = 100 # $ per week added to the investment portfolio
dca_interval_array = np.linspace(7, 7, 1) # number of days between dollar cost averaging evaluations

# Subfunctions
def eval_buy(price, day, buy_interval, alpha, local_min):
    if day % np.real(buy_interval) == 0:
        if price > (1+np.real(alpha))*np.real(local_min):
            return True
        else:
            return False
    else:
        return False
    
def eval_sell(price, day, sell_interval, beta, local_max):
    if day % np.real(sell_interval) == 0:
        if price < (1-np.real(beta))*np.real(local_max):
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

def optimal_scenario(x):
    [alph,bet,buy_rat,sell_rat,risk_rat,buy_int,sell_int,mov_avg_int] = x
    buy_col = 0
    sell_col = 1
    # mov_avg_int = ma
    week_inv = 100
    # reset variables for the next iteration
    buy_dir = 0 # 0 for bearish and 1 for bullish
    prev_buy_dir = 0 # 0 for bearish and 1 for bullish
    sell_dir = 0 # 0 for bearish and 1 for bullish
    prev_sell_dir = 0 # 0 for bearish and 1 for bullish
    day_count = 0 # number of days since the start of the data
    loc_min = price_data[0,1] # minimum value of the stock price
    loc_max = price_data[0,2] # maximum value of the stock price
    buy_avg = price_data[0,0] # opening price of the first day
    sell_avg = price_data[0,1] # opening price of the first day
    
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
    profit = revenue - expenses

    # print("P: ", f"{profit:.2f}", "T: ", trades, "A: ", alph, " B: ", bet, " BR: ", buy_rat, " SR: ", sell_rat, " RR: ", risk_rat, " BI: ", buy_int, " SI: ", sell_int, " BC: ", buy_col, " SC: ", sell_col)
             
    return profit#, trades
              
def dca_scenario(price_data, buy_col, sell_col, buy_rat, sell_rat, risk_rat, dca_int, week_inv):
    # reset variables for the next iteration
    buy_dir = 0 # 0 for bearish and 1 for bullish
    prev_buy_dir = 0 # 0 for bearish and 1 for bullish
    sell_dir = 0 # 0 for bearish and 1 for bullish
    prev_sell_dir = 0 # 0 for bearish and 1 for bullish
    day_count = 0 # number of days since the start of the data
    loc_min = price_data[0,buy_col] # minimum value of the stock price
    loc_max = price_data[0,sell_col] # maximum value of the stock price
    buy_avg = price_data[0,buy_col] # opening price of the first day
    sell_avg = price_data[0,sell_col] # opening price of the first day
    
    dca_money = 100 # amount of money in the wallet available for investment
    dca_shares = 0 # amount of shares in the portfolio available for selling
    dca_revenue = 0 # profit from the investment strategy
    dca_expenses = 0 # expenses from the investment strategy
    dca_profit = 0 # total profit from the investment strategy
    dca_trades = 0 # total number of trades made in the investment strategy
    
    dca_inv = week_inv*(dca_int/7) # average investment per dca interval
    
    for d in range(price_data.shape[0]-1): # iterate through the rows of the data for number of days

        dca_money, dca_expenses, dca_shares, dca_trades = dca_buy(prices[d,buy_col], d, dca_money, dca_shares, dca_expenses, dca_trades, dca_int, dca_inv)
        dca_money = update_savings(dca_money, d, week_inv)
                            
    # calculate the total profit from the investment strategy
    dca_money, dca_revenue, dca_shares, dca_trades = liquidate_shares(price_data[d,sell_col], dca_money, dca_shares, dca_revenue, dca_trades)
    dca_profit = dca_revenue - dca_expenses
    
    return dca_profit, dca_trades


# Load the data
# df = pd.read_csv("WINTER 2025 - ME 575\ME 575 - Project\HistoricalData_SP500_Daily_2012-Present.csv", header=0)
folder = "StockMarketProject\\"
files = ["SP500_Daily_2_5_2015_to_2_4_2025.csv","Tesla_Daily_7_1_2023_to_3_20_2025.csv","Amazon_Daily_3_28_2017_to_5_20_2025.csv"]
# filename = os.path.join(folder,files[0])
filename = files[0]
df = pd.read_csv(filename, header=0)

df_rows = df.shape[0]
print(df_rows)
df_columns = df.shape[1]
print(df_columns)

prices = np.zeros((df_rows, df_columns-1))

df_Date = np.linspace(0, df_rows, df_rows) # Days since the start of the data ()
# df_Date = df['Date']
# df_Date.reverse()
# print(df_Date)
df_Close = list(reversed(df['Close']))
# print(df_Close)
df_Open = list(reversed(df['Open']))
# print(df_Open)
df_High = list(reversed(df['High']))
# print(df_High)
df_Low = list(reversed(df['Low']))
# print(df_Low)

prices[:,0] = df_Open
prices[:,1] = df_Close
prices[:,2] = df_High
prices[:,3] = df_Low

# Count the number of zero values in the data
zero_values = np.count_nonzero(prices == 0)
print("Num Zeros: ", zero_values)
# Correct any zero values in the data with averages of the surrounding values
for i in range(df_columns-1):
    for j in range(df_rows):
        if prices[j,i] == 0:
            if j == 0:
                prices[j,i] = np.mean(prices[j+1:j+2,i])
            elif j == df_rows-1:
                prices[j,i] = np.mean(prices[j-2:j-1,i])
            else:
                prices[j,i] = np.mean(prices[j-1:j+1,i])
# Count the number of zero values in the data
zero_values = np.count_nonzero(prices == 0)
print("Num Zeros: ", zero_values)

price_data = prices                 # set global variable here for the opimizer

# Display the first 5 rows of the data
print(df.head())
# Display the last 5 rows of the data
print(df.tail())
# Display the shape of the data
print(df.shape)
# Display the data types of the columns
print(df.dtypes)
# Display the summary statistics of the data
print(df.describe())
# Check for missing values in the data
print(df.isnull().sum())
# Plot the closing price of the S&P 500 index over time
plot_initial_figure = True
if(plot_initial_figure):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Open'])
    plt.plot(df['Date'], df['Close'])
    plt.plot(df['Date'], df['High'])
    plt.plot(df['Date'], df['Low'])
    plt.legend(['Open', 'Close', 'High', 'Low'])
    plt.gca().invert_xaxis() # Flip the x-axis
    plt.title('S&P 500 Index Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


max_profit = 0
optimal_alpha = 0
optimal_beta = 0
optimal_buy_ratio = 0
optimal_sell_ratio = 0
optimal_risk_ratio = 0
optimal_buy_interval = 0
optimal_sell_interval = 0
optimal_buy_column = 0
optimal_sell_column = 0
optimal_dca_buy_interval = 0
optimal_dca_rate = 0
# Main Loop

#variables: 
# rr  -> (0.1,1)
# sr  -> (0,0.1)
# br -> (0.01,0.1)
# si -> (1,7)
# bi -> (1,7)
# dca_i -> (7,1) (opposite of si/bi)
# ma -> moving average. Maybe don't do this one? (0,21) days
# b -> (0.01,0.1)
# a -> (0.01, 1.0)

run_loop_analysis = False

if(run_loop_analysis):
    for rr in risk_ratio_array:
        for sr in sell_ratio_array:
            for br in buy_ratio_array:
                for si in sell_interval_array: 
                    for bi in buy_interval_array:
                        for dca_i in dca_interval_array:
                            for ma in moving_average_array:
                                for b in beta_array:
                                    for a in alpha_array:
                                        for i in range(df_columns-1):
                                            for j in range(df_columns-1):
                                                
                                                if ma == 0:
                                                    total_profit, total_trades = optimal_scenario([prices, i, j, 1, a, b, br, sr, rr, bi, si, weekly_investment])
                                                else: 
                                                    total_profit, total_trades = optimal_scenario([prices, i, j, ma, a, b, br, sr, rr, bi, si, weekly_investment])
                                                
                                                dca_total_profit, dca_total_trades = dca_scenario(prices, i, j, br, sr, rr, dca_i, weekly_investment)
                                                
                                                opt_percentage = 100*((total_profit - dca_total_profit) / dca_total_profit)
                                                
                                                if total_profit > max_profit:
                                                    max_percentage = opt_percentage
                                                    max_profit = total_profit
                                                    optimal_alpha = a
                                                    optimal_beta = b
                                                    optimal_buy_ratio = br
                                                    optimal_sell_ratio = sr
                                                    optimal_risk_ratio = rr
                                                    optimal_buy_interval = bi
                                                    optimal_sell_interval = si
                                                    optimal_buy_column = i
                                                    optimal_sell_column = j
                                                # print("MP: ", f"{max_profit:.2f}", "P: ", f"{total_profit:.2f}", "(DCA:", f"{dca_total_profit:.2f}", ") T: ", total_trades, "(DCA: ", dca_total_trades, ") A: ", optimal_alpha, " B: ", optimal_beta, " BR: ", optimal_buy_ratio, " SR: ", optimal_sell_ratio, " RR: ", optimal_risk_ratio, " BI: ", optimal_buy_interval, " SI: ", optimal_sell_interval, " BC: ", optimal_buy_column, " SC: ", optimal_sell_column)
                                                print("%:", f"{opt_percentage:.2f}", "(M%:", f"{max_percentage:.2f}", ") P:", f"{total_profit:.2f}", "(MP:", f"{max_profit:.2f}", ") DCA:", f"{dca_total_profit:.2f}", "T:", total_trades, "(DCA:", dca_total_trades, ") DCA_INT:", dca_i, "MA:", ma, "A:", a, "B:", b, "BR:", br, "SR:", sr, "RR:", rr, "BI:", bi, "SI:", si, "BC:", i, "SC:", j)

                                                # OPTIMAL VALUES:
                                                # %: 2.25 (M%: 2.25 ) P: 37370.36 (MP: 37370.36 ) DCA: 36547.37 T: 375 (DCA: 362 ) DCA_INT: 7.0 MA: 0.0 A: 0.001 B: 0.1 BR: 1.0 SR: 0.0 RR: 0.1 BI: 1.0 SI: 1.0 BC: 3 SC: 2
else:
    f_opt = optimal_scenario

    # [alph,bet,buy_rat,sell_rat,risk_rat,buy_int,sell_int,week_inv] = x
    # buy_col = 0
    # sell_col = 1
    # mov_avg_int = 21
    dca_i = 1

    alpha_bounds = (0.01,1.0)
    beta_bounds = (0.01,0.1)
    buy_ratio_bounds = (0.01,0.1)
    sell_ratio_bounds = (0,0.1)
    risk_ratio_bounds = (0.1,1)
    buy_interval_bounds = (1,7)
    sell_interval_bounds = (1,7)
    ma_bounds = (0,21)
    the_bounds = (alpha_bounds,beta_bounds,buy_ratio_bounds,sell_ratio_bounds,risk_ratio_bounds,buy_interval_bounds,sell_interval_bounds,ma_bounds)
    pop_size = 100
    generations = 20
    dims = np.shape(the_bounds)[0]  # number of variables in x0
    
    dca_store = np.array([])
    profit_store = np.array([])
    x_store = np.array([])
    n_trade_store = np.array([])
    loop_iter = 0 

    # for i in range(df_columns-1):
    #     for j in range(df_columns-1):
    # df_columns = [close, open, high, low]
    i = 0
    j = 1
    
    print(f"\nRunning Genetic Algorithm")
    
    # x_star, f_star, x, n_gen = gf.genetic_algorithm(f_opt,gf.fit_func, bounds=the_bounds, pop_size=pop_size, generations=generations, dims=dims)
    x_star, f_star, x, n_gen = gf.particle_swarm(f_opt,bounds=the_bounds, pop_size=pop_size, generations=generations, dims=dims)
    # [rr,sr, br, si, bi, dca_i, b,a] = x_star
    [a,b,br,sr,rr,bi,si,ma] = x_star
    total_profit = f_star
    
    print("Applying Gradient to best points...")
    # take the best points (n_points) and use gradient approach to converge solution
    n_points = (int)(np.ceil(pop_size/10))

    x_star_n = np.zeros([n_points,dims])
    f_star_n = np.zeros(n_points)
    if(np.shape(x)[0] >=np.size(f_star_n)):
        for k in range(n_points):
            res_n = opt(f_opt,
                        x[k,:],
                        jac=lambda l: FD.Complex_Step(f_opt,l),       # need this for some reason. It doesn't like just adding the function handle.
                        bounds=the_bounds,
                        tol=1e-12)
            x_star_n[k,:] = res_n.x
            f_star_n[k] = res_n.fun
    
    total_profit = np.max(f_star_n)

    dca_total_profit, dca_total_trades = dca_scenario(prices, i, j, br, sr, rr, dca_i, weekly_investment)
    
    opt_percentage = 100*((total_profit - dca_total_profit) / dca_total_profit)
    max_percentage = opt_percentage
    total_trades = dca_total_trades
    if(loop_iter>1):
        dca_store = np.vstack([dca_store,dca_total_profit])
        profit_store = np.vstack([profit_store,total_profit])
        x_store = np.vstack([x_store,x_star_n[0,:]])
        n_trade_store = np.vstack([n_trade_store,total_trades])
    else:
        dca_store = dca_total_profit
        profit_store = total_profit
        x_store = x_star_n[0,:]
        n_trade_store = total_trades

    # print("MP: ", f"{max_profit:.2f}", "P: ", f"{total_profit:.2f}", "(DCA:", f"{dca_total_profit:.2f}", ") T: ", total_trades, "(DCA: ", dca_total_trades, ") A: ", optimal_alpha, " B: ", optimal_beta, " BR: ", optimal_buy_ratio, " SR: ", optimal_sell_ratio, " RR: ", optimal_risk_ratio, " BI: ", optimal_buy_interval, " SI: ", optimal_sell_interval, " BC: ", optimal_buy_column, " SC: ", optimal_sell_column)
    # print("%:", f"{opt_percentage:.2f}", "(M%:", f"{max_percentage:.2f}", ")\n P:", f"{total_profit:.2f}", "(MP:", f"{max_profit:.2f}", ")\n DCA:", f"{dca_total_profit:.2f}", "T:", total_trades, "(DCA:", dca_total_trades, ")\n DCA_INT:", dca_i, "MA:", ma, "\nA:", a, "B:", b, "\nBR:", br, "SR:", sr, "\nRR:", rr, "\nBI:", bi, "SI:", si, "\nBC:", i, "SC:", j)
    loop_iter = loop_iter + 1

    index_max = np.argmax(profit_store)
    if(np.size(profit_store)==1):
        x_star = x_store
        f_star = profit_store
        dca_profit = dca_store
    else:
        x_star = x_store[index_max,:]
        f_star = profit_store[index_max]
        dca_profit = dca_store[0]

    # Now that the optimization is done, we will print the output for the optimization for each Moving Average type used.
    print("Optimized Values:")
    print(f'alpha: {x_star[0]}')
    print(f'beta: {x_star[1]}')
    print(f'buy ratio: {x_star[2]}')
    print(f'sell ratio: {x_star[3]}')
    print(f'risk ratio: {x_star[4]}')
    print(f'buy interval: {x_star[5]}')
    print(f'sell interval: {x_star[6]}')
    print(f'moving average interval: {x_star[7]}')

    print(f"Max profit: {np.round(f_star,2)}")
    print(f"DCA profit: {np.round(dca_profit,2)}")