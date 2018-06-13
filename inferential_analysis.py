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

def preference_plot(stats_df, column, option, fname, deployment=False):
    '''
    plot preferenec
    :param stats_df:
    :param column: which column (ex. 'sub_scale')
    :param option: which option (ex. 'average')
    :return:
    '''

    b = stats_df[stats_df[column] == option]
    b.answers.replace(np.nan, 0)
    cnames = ['education','robot','gender','side','sub_scale']
    m = 2
    n = int(round(float(cnames.__len__())/m))
    fig, ax = plt.subplots(m,n)
    for i, c in enumerate(cnames):
        b.answers = b.answers.astype('float64')
        # (b.groupby(c).size() / 3).tolist()
        # d = b.groupby(c)['rationality']
        # d.groups.keys()
        if deployment:
            sns.barplot(x = c, y = 'answers', data = b, ax = ax[i/n, i%n])
        else:
            sns.barplot(hue = c,x = 'rationality', y = 'answers', data = b, ax = ax[i/n, i%n])

        # sns.countplot(x = 'answers', hue='rationality', data=b, ax = ax[i/n, i%n])
        # sns.factorplot(hue = c,x = 'rationality', y = 'answers', data = b, ax = ax[i/n, i%n])
    fig.suptitle('n = '+str(stats_df.userID.unique().__len__()))
    save_maxfig(fig, fname)
    return fig

def see_the_data(cdf):
    '''
    Plot histogram of the parameters of the participants.
    :param cdf: dataframe of the data from robots vids questionnaire.
    '''
    print('Plot histogram of the parameters of the participants...')
    # todo: change ticks of gender andd age

    fig, ax = plt.subplots(1,3)
    users_data = cdf[(cdf.feature == 'BFI') & (cdf.sub_scale == 'S1') ]
    sns.distplot(pd.to_numeric(users_data.age), kde=False, ax=ax[0])
    pd.value_counts(users_data.gender)
    sns.countplot(users_data.gender, ax=ax[1])
    pd.value_counts(users_data.education)
    sns.countplot(users_data.education, ax=ax[2])
    save_maxfig(fig, 'participants_histogram')

def preference_per_question(pref_df):
    '''
    plot the preference index per question
    :param pref_df: preference data frames of all the questions
    :return:
    '''
    fig,ax = plt.subplots(1,1)

    # todo: for pilot only
    pref_df = pref_df.reset_index(drop=True)
    a = pref_df.loc[pref_df.question.isin(['89', '91', '95'])].rationality
    a = a.str.replace('irrational', 'abc')
    a = a.str.replace('rational', 'irrational')
    a = a.str.replace('abc', 'rational')
    pref_df.loc[pref_df.question.isin(['89', '91', '95']), 'rationality'] = a.tolist()
    pref_df.loc[pref_df.rationality.isin(['irrational_rational']), 'preference'] = -pref_df.loc[pref_df.rationality.isin(['irrational_rational']), 'preference']
    pref_df.loc[pref_df.rationality.isin(['irrational_rational']), 'rationality'] = 'rational_irrational'

    sns.pointplot(x='question', y='preference', hue='rationality', data=pref_df, legend_out=True,ax=ax)
    save_maxfig(fig, 'preference')

def pair_plot(stats_df, surveys):
    '''
    create dataframe in the shape, columns: ('NARS1,NARS2,...,GODSPEED2')
    :param stats_df:  stats dataframe
    :param surveys: which surveys we are intrested in (BFI,NARS,GODSPEED1,GODSPEED2)
    :return: pairplot
    '''
    fig,ax = plt.subplots(1,1)
    g = stats_df[(stats_df['feature'] == 'GODSPEED1') | (stats_df['feature'] == 'GODSPEED2') | (stats_df['feature'] == 'BFI') | (
            stats_df['feature'] == 'NARS')][['feature', 'sub_scale', 'answers']]
    g['feature'] = g[['feature', 'sub_scale']].sum(axis=1)
    g = g.drop('sub_scale', axis=1)

    d = {}
    for gg in g['feature'].unique():
        d[gg] = g[g['feature'] == gg].answers.tolist()
    dff = pd.DataFrame.from_dict(d)
    cnames = []
    for c in surveys:
        cnames += dff.columns[dff.columns.str.contains(c)].tolist()
    sns.pairplot(dff[cnames],ax = ax)
    save_maxfig(fig, 'pairplot_participant')

    # sns.pairplot(g, hue='feature')

def preference_cinsistency(users_pref_tot):
    '''
    show the dynamics of choice of all the users per question
    :param users_pref_tot:
    :return:
    '''
    users_pref_tot = users_pref_tot.replace('rational', 0)
    users_pref_tot = users_pref_tot.replace('half', 1)
    users_pref_tot = users_pref_tot.replace('irrational', 2)
    users_pref_tot = users_pref_tot.astype('int')

    fig,ax = plt.subplots(1,1)

    p = ax.pcolor(users_pref_tot, edgecolors='k', cmap='tab20b')

    y_labels = users_pref_tot.index.tolist()
    N = y_labels.__len__()
    ax.set_yticks(np.linspace(0.5, N - 0.5, N))
    ax.set_yticklabels(y_labels)

    fig.colorbar(p, ax=ax)

    save_maxfig(fig, 'users_preference_per_question')

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
    rDeployment_rh  = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih  = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri  = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']
    rDeployment_tot =  rDeployment_ih + rDeployment_rh + rDeployment_ri
    rDeployment     = {'rDeployment_tot':rDeployment_tot, 'rDeployment_ri':rDeployment_ri, 'rDeployment_rh': rDeployment_rh, 'rDeployment_ih':rDeployment_ih}
    # rDeployment     = {'rDeployment_tot':rDeployment_tot}
    # rDeployment     = {'rDeployment_tot':rDeployment_ri}

    for rDep in rDeployment:
        for i, rd in enumerate(rDeployment[rDep]):
            raw_path   = 'data/raw_dataframe_' + rd + '.csv'
            stats_path = 'data/stats_dataframe_' + rd + '.csv'
            if (os.path.isfile(raw_path)) & (os.path.isfile(stats_path)):
                raw_df = pd.read_csv(raw_path)
                stats_df = pd.read_csv(stats_path)

                if 'crdf' in locals():
                    # crdf = combine_dataframes(crdf, raw_df)
                    a = raw_df[raw_df.columns[5:]]
                    a.columns = (np.arange(a.columns.__len__()) + 1) + i * 100
                    crdf = pd.concat([crdf, a], axis=1)

                    # cdf = combine_dataframes(cdf, stats_df)
                else:
                    crdf = raw_df
                    # cdf = stats_df

        # filtering out trap question
        from reformat_data import trap_exclusion1, create_stats_df
        before = crdf[crdf.columns[:-5]].columns.__len__()
        crdf, users_after_exclusion = trap_exclusion1(crdf)
        # raw_df = response_time_exclusion(raw_df, users_after_exclusion)
        excluded = before - users_after_exclusion.__len__()
        print('exclude:', excluded,'out of', before)

        crdf = crdf.set_index(crdf[crdf.columns[0]])
        crdf = crdf[crdf.columns[1:]]

        cdf = create_stats_df(crdf,rDep)
        cdf.loc[:, 'gender'] = cdf.loc[:, 'gender'].replace({1.0: 'female', 2.0: 'male'})
        cdf.loc[:, 'education'] = cdf.loc[:, 'education'].replace({1.0: '<HS', 2.0: 'HS', 3.0:'<BA', 4.0:'BA', 5.0:'MA', 6.0:'professional',7.0:'PhD'})

        # feel_the_data(stats_df)
        # todo: add n to the bars in the graphs!!!
        fig = preference_plot(cdf, 'sub_scale', 'summary')
        save_maxfig(fig, rDep + '_barplot_only_choices')
        fig1 = preference_plot(cdf, 'sub_scale', 'summary', deployment = True)
        save_maxfig(fig1, rDep + '_summary')

        if rDep == 'rDeployment_tot':
            see_the_data(cdf)

        del(crdf)


    plt.show()



    print('finished inferential analysis!')