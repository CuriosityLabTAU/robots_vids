import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
plt.style.use('ggplot')
# plt.xkcd();

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
    # b1 = stats_df[(stats_df[column] == 'prefer') & (stats_df.feature == 'q_preference')]
    # b1.answers = b1.answers.replace(b1.answers.unique().max(), 2.)
    # b = b.append(b1)
    # b = b.reset_index(drop=True)
    # b = stats_df[(stats_df[column] == option) & (stats_df.feature == 'r_preference')]
    b.answers.replace(np.nan, 0)
    cnames = b.columns[[0,4,7,8,9]]
    m = 2
    n = int(round(float(cnames.__len__())/m))
    fig, ax = plt.subplots(m,n)
    for i, c in enumerate(cnames):
        b.answers = b.answers.astype('float64')
        sns.barplot(hue = c,x = 'rationality', y = 'answers', data = b, ax = ax[i/n, i%n])
        # sns.countplot(x = 'answers', hue='rationality', data=b, ax = ax[i/n, i%n])
        # sns.factorplot(hue = c,x = 'rationality', y = 'answers', data = b, ax = ax[i/n, i%n])

    return fig


def save_maxfig(fig, fig_name, ax=None, save_plotly=None, save_pkl = 1, transperent = False, frmt='png', resize=None):
    '''Save figure in high resultion'''
    # matplotlib.rcParams.update({'font.size': 40})
    fig.set_size_inches(12, 12)
    if resize != None:
        fig.set_size_inches(resize[0], resize[1])
    p_fname = 'figs_files/'
    if not os.path.exists(p_fname):
        os.makedirs(p_fname)

    fig_name+='.'+frmt
    plt.savefig(p_fname+fig_name, dpi=300, transperent=transperent, format=frmt)
    # print fig_name
    # print os.getcwd()+fig_name
    fig.set_size_inches(5, 5)
    # saving the figures files.

    if save_pkl == 1:
        p_fname = p_fname + fig_name + 'fig.pckl'
        # if ax != None:
        #     pickle.dump(ax, file(p_fname, 'wb'))
        # else:
        pickle.dump(fig, file(p_fname, 'wb'))

if __name__ == "__main__":
    rDeployment_rh = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']
    rDeployment = [rDeployment_ri, rDeployment_rh, rDeployment_ih, rDeployment_ih+rDeployment_rh+rDeployment_ri]

    for j, rDep in enumerate(rDeployment):
        for i, rd in enumerate(rDep):
            raw_path   = 'data/raw_dataframe_' + rd + '.csv'
            stats_path = 'data/stats_dataframe_' + rd + '.csv'
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
        fig = preference_plot(cdf, 'sub_scale', 'summary')

        # preference_plot(cdf, 'sub_scale', 'prefer')
        # preference_plot(cdf, 'sub_scale', 'choice')
        if j!=3:
            save_maxfig(fig, rDep[0][2]+rDep[0][5] + '_barplot_only_choices')
        else:
            save_maxfig(fig, 'tot_barplot_only_choices')
        del(crdf)
    # todo: wha is the total number of participants. how many were excluded? gender?


    plt.show()



    print('finished inferential analysis!')