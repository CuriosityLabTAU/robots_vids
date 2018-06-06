import pandas as pd
import numpy as np
import seaborn as sns



def combine_dataframes(df1, df2, nf):
    '''
    :param df: data frames to combine
    :param nf: index of the features
    :return:
    '''
    usersID = df2.columns[nf:]
    features = df2.columns[:nf]
    tempd_df = df2.drop(features, axis=1)
    return pd.concat([df1, tempd_df], axis=1)

def feel_the_data(stats_df):
    '''
    see correlations in the data
    :return: pairplot of the dataframe
    '''

    sdf = stats_df.drop(stats_df.columns[[0, 1, 3, 4]], axis=1)
    sdf = sdf.drop(1, axis=0)
    sdf = sdf.transpose()
    cnames = sdf.loc['feature']
    sdf = sdf.drop('feature', axis=0)
    # sdf = sdf.drop('feature', axis=1)
    sdf = sdf.reset_index(drop=True)
    sns.pairplot(sdf)

if __name__ == "__main__":
    rDeployment = 'rbrlri'
    raw_df = pd.read_csv('data/raw_dataframe_'+rDeployment+'.csv')
    stats_df = pd.read_csv('data/stats_dataframe_'+rDeployment+'.csv')


    feel_the_data(stats_df)
    # a,drop(,axis=1)
    print('finished inferential analysis!')