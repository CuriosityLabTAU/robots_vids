import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os



def combine_dataframes(df1, df2):
    '''
    :param df: data frames to combine
    :param nf: index of the features
    :return:
    '''
    temp = pd.concat([df1, df2])
    temp = temp.drop(columns = temp.columns[0])
    return temp

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

def preference_plot(stats_df, column, option):
    '''
    plot preferenec
    :param stats_df:
    :param column: which column (ex. 'sub_scale')
    :param option: which option (ex. 'average')
    :return:
    '''

    b = stats_df[stats_df[column] == option]
    cnames = b.columns[[0,4,7,8,9]]
    # cnames = b.columns[[0,7,8,9,10]] # userID
    m = 2
    n = int(round(float(cnames.__len__())/m))
    print(m,n)
    fig, ax = plt.subplots(m,n)
    for i, c in enumerate(cnames):
        b.answers = b.answers.astype('float64')
        print([i,(i)/(n), i%n])
        sns.barplot(hue = c,x = 'rationality', y = 'answers', data = b, ax = ax[i/n, i%n])
    print('t')



if __name__ == "__main__":
    rDeployment = ['rrrlbh', 'rbhlrr', 'rbilrh', 'rbilrr', 'rbrlrh', 'rbrlri', 'rrhlbi', 'rrilbh', 'rrhlbr', 'rrilbr', 'rrrlbi', 'rbhlri']
    for i, rd in enumerate(rDeployment):
        raw_path   = 'data/raw_dataframe_' + rd + '.csv'
        stats_path = 'data/stats_dataframe_'+rd+'.csv'
        print(stats_path, os.path.isfile(raw_path), i)
        if (os.path.isfile(raw_path)) & (os.path.isfile(stats_path)):
            raw_df = pd.read_csv(raw_path)
            stats_df = pd.read_csv(stats_path)

            if 'crdf' in locals():
                crdf = combine_dataframes(crdf, raw_df)
                cdf = combine_dataframes(cdf, stats_df)
            else:
                crdf = raw_df
                cdf = stats_df

    # feel_the_data(stats_df)
    preference_plot(cdf, 'sub_scale', 'average')
    plt.show()



    print('finished inferential analysis!')