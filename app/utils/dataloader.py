import pandas as pd
import numpy as np



class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        #ОБРОБКА ДАННИХ ТУТ!!!!   

        return self.dataset