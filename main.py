import os
import pandas as pd

def load_data():
    folder = os.getcwd()

    # file_path = os.path.join(folder, 'jijfsf','csv_data')
    file_path = r'C:\Users\karan\Desktop\Sleepyy\project\jijfhsf\csv_data'

    csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

    df = pd.read_csv(os.path.join(file_path, csv_files[0]))

    return df

df = load_data()
def what_to_predict():
    user_input = input(f'What do you want to predict? \n {df.columns.tolist()} \n')
    return user_input
# load_data()

# what_to_predict()