import os
os.chdir("..")
from demand_models.ar import AutoRegression


class AutoRegressiveDemandPredictor:
    def __init__(self,
                 config_path,
                 steps,
                 seed,
                 days,
                 bins_size,
                 model_path,
                 load_model=False):

        """
        :param config_path: inventory.yaml file path
        :param steps: episode_max_steps in the inventory.yaml file
        :param days: number of days to sample for poisson for training
        :param bins_size: the bin size i.e. number of steps in each bin
        :param model_path: the path where the model is saved and loaded from
        :param load_model: load already saved model
        """

        self.model_path = model_path

        if load_model:
            self.ar = AutoRegression(model_path, config_path, seed, steps, days, bins_size)
            self.ar.load_models()
        else:
            self.ar = AutoRegression(model_path, config_path, seed, steps, days, bins_size)
            self.ar.sample_data()
            self.ar.prepare_data()
            self.ar.train_ar()
            self.ar.save_models()


    def predict(self, curr_data, pred_steps, item):
        """
        :param curr_data: the new data received
        :param pred_steps: number of predictions to be returned
        :param item: the product ID of curr_data and prediction data, according to inventory.yaml
        :return: predictions, numpy array
        """
        predictions = self.ar.predict_next_n(curr_data=curr_data, pred_steps=pred_steps, item=item)

        return predictions


if __name__ == "__main__":
    for i in range(10):
        predictor = AutoRegressiveDemandPredictor(config_path="./inventory.yaml", seed=i, steps=20, days=10, bins_size=1,
                                                  model_path="./models/ar_model_"+str(i))
        print("Model: ", i)
