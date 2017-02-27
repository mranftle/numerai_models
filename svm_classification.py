import pandas as pd
import numpy as np
from sklearn import svm
def main():
    print ("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target']
    X = training_data.drop('target', axis=1)
    t_id = prediction_data['t_id']
    x_prediction = prediction_data.drop('t_id', axis=1)


    print("Training model...")
    # generate model
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X,Y)

    y_prediction = model.predict(x_prediction)

    print y_prediction

if __name__ == '__main__':
    main()