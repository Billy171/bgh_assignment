import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TradingStrategy:
    def __init__(self, model, do_null_tol, weight, portfolio_val):
        self.model = model
        self.do_null_tol = do_null_tol
        self.weight = weight
        self.portfolio_val = portfolio_val

    def predict(self, model, data):
        predictions = model.predict(data)
        return predictions

    def classify_trade(self, predictions):
        trade = pd.cut(predictions, [-np.inf, -0.02, 0.02, np.inf], labels=['SELL', 'DO NOTHING', 'BUY']).to_numpy()
        return trade

    def assign_weights(self, predictions):
        trade = self.classify_trade(predictions)
        pred_buy = np.where(trade == 'BUY', predictions, 0)
        if pred_buy.sum() == 0:
            return np.zeros(len(predictions))
        if self.weight == 'regular':
            return pred_buy/(pred_buy.sum())
        if self.weight == 'softmax':
            return np.exp(pred_buy)/np.exp(pred_buy).sum()
        if self.weight == 'uniform':
            buy_number = len(np.where(trade == 'BUY')[0])
            return np.where(pred_buy != 0, 1/buy_number, 0)

    def run_trade_sim(self, data, title):
        days = pd.unique(data['Date'])
        portfolio_value = self.portfolio_val
        profit_loss = [portfolio_value]

        for day in days:
            data_day = data[data['Date'] == day]
            returns = data_day['Target']
            x = data_day.drop('Date', axis=1)
            x = x.drop('Target', axis=1)
            #now generate predictions for the day's selected coins

            predictions = self.predict(self.model, x)

            #now generate portfolio weights
            weights = self.assign_weights(predictions)

            portfolio_weighted = portfolio_value*weights

            #get day returns
            total_return = sum(portfolio_weighted*(1+returns))

            #update portfolio value (after selling)
            portfolio_value = portfolio_value - portfolio_weighted.sum() + total_return

            profit_loss.append(portfolio_value)

        #plot P&L
        plt.plot(profit_loss)
        plt.title(title+' '+str(self.do_null_tol)+' '+self.weight)
        plt.xlabel('Day')
        plt.ylabel('$')
        plt.show()
        return profit_loss


