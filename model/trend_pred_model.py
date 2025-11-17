import pandas as pd
from model.mylenet import lenet_regression
from data.process_data import techFactorTransform, prepareTrainData


class TrendPredModel():
    def __init__(self, winlen, model_name='lenet'):
        self.winlen = winlen
        self.model_dir = f'./model/regression/{model_name}'
        self.model_path = self.model_dir + '/epoch_sel.h5'

    def train(self, data_path: str, split = 0.3):
        train_data = prepareTrainData(self.winlen, data_path, split)
        print("Train samples:", train_data.train_x.shape[0], " Val samples:", train_data.val_x.shape[0])
        model = lenet_regression(shape= train_data.train_x.shape[1:])
        model.train(train_data.train_x, 
                    train_data.train_y, 
                    train_data.val_x,
                    train_data.val_y,
                    bs=32,
                    lr=5e-5,
                    epochs=20,
                    path=self.model_dir)
    
    def predict(self, df: pd.DataFrame): 
        tsData = techFactorTransform(self.winlen, df, with_label=False)
        model = lenet_regression(shape= tsData.array_3d.shape[1:])
        model.load(self.model_path)
        df = pd.DataFrame({"date": tsData.dates, "prediction": model.predict(tsData.array_3d)[:,0]})
        df.set_index("date", inplace=True)
        return df
