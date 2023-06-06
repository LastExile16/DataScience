# improt all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
# from sklearn.base import BaseEstimator, TransformerMixin
# import os
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))



class Absenteeism_model:
    def __init__(self, model_file, scaler_file):
        with open(model_file, 'rb') as md, open(scaler_file, 'rb') as sc:
            self.reg = pickle.load(md)
            self.scaler = pickle.load(sc)
            self.data = None
    
    def load_and_clean_data(self, data_csv):

        # import the data
        df = pd.read_csv(data_csv)
        # store the data within the object for later use
        self.df_with_predicitons = df.copy()
        # drop the ID column
        df = df.drop(['ID'], axis=1)
        # to preserve the same structure of our previous code, we will add the target column here but with NaN
        df['Absenteeism Time in Hours'] = 'NaN'

        # create a seperate dataframe for dummy variables
        reason_columns = pd.get_dummies(df['Reasons for Absence'], drop_first=True)

        # split reason_columns into 4 types
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        # to avoid multicollinearity, drop the 'Reasons for Absence' column from df
        df.drop(['Reasons for Absence'], axis=1, inplace=True)

        # concatenate df and the 4 types of reason for absence
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4])

        # assign names to the 4 reason type columns
        # df.columns = ['Transportation Expense', 'Distance to Work', 'Age',
        #       'Daily Work Load Average', 'Body Mass Index', 'Education',
        #       'Children', 'Pets', 'Absenteeism Time in Hours', 'Month Value', 'Day of the Week', 
        #       'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',]
        df.rename(columns={'1': 'Reason_1', '2': 'Reason_2', '3': 'Reason_3', '4': 'Reason_4'}, inplace=True)
        
        # convert the 'Date' column into datetime
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format="%d/%m/%Y")

        # extract month and day of the week value from column Date
        df['Month Value'] = df['Date'].dt.month
        df['Day of the Week'] = df['Date'].dt.day_of_week
        df.drop('Date', axis=1, inplace=True)

         # reorder the columns
        df = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
                'Day of the Week', 'Transportation Expense', 'Distance to Work',
                'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                'Children', 'Pets', 'Absenteeism Time in Hours']]
        
        # Education to dummies
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        # replace the NaN values
        df = df.fillna(value=0)

        # drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
        
        # drop the variables we decide we don't need
        # I inlcuded 'Education' because in my version, 'Education' is also a useless parameter
        df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work', 'Education'],axis=1)
        
        # we have included this line of code if you want to call the 'preprocessed data'
        self.preprocessed_data = df.copy()

        # we need this line so we can use it in the next functions
        self.data = self.scaler.transform(df)

    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):  
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
    
    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    
    # predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

