from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from modeling import make_pipeline

pipeline = make_pipeline("scal.joblib","model.joblib")

class MaStrategie(Strategy):

    def init(self):
        self.model = pipeline
        
    def next(self):
        jour_en_cours = self.data.df.iloc[-1:]
        
        volatilite = self.model.predict(jour_en_cours)
        
        if volatilite == 1:
            if jour_en_cours.Open.iloc[0] > jour_en_cours.Open_1.iloc[0]:
                self.buy()
            else: 
                self.sell()

def result(data,cash=10000,commission=0):
    bt = Backtest(data, MaStrategie,
                cash=cash, commission=commission,
                exclusive_orders=True)

    output = bt.run()
    fig = bt.plot(open_browser=False)

    return output,fig
    
