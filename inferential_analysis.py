import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap
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

    # # todo: for pilot only (reverse disjunction)
    pref_df = pref_df.reset_index(drop=True)
    # a = pref_df.loc[pref_df.question.isin(['89', '91', '95'])].rationality
    # a = a.str.replace('irrational', 'abc')
    # a = a.str.replace('rational', 'irrational')
    # a = a.str.replace('abc', 'rational')
    # pref_df.loc[pref_df.question.isin(['89', '91', '95']), 'rationality'] = a.tolist()
    # pref_df.loc[pref_df.rationality.isin(['irrational_rational']), 'preference'] = -pref_df.loc[pref_df.rationality.isin(['irrational_rational']), 'preference']
    # pref_df.loc[pref_df.rationality.isin(['irrational_rational']), 'rationality'] = 'rational_irrational'

    sns.pointplot(x='question', y='preference', hue='rationality', data=pref_df, legend_out=True,ax=ax)
    save_maxfig(fig, 'preference')

def pair_plot(stats_df, surveys):
    '''
    create dataframe in the shape, columns: ('NARS1,NARS2,...,GODSPEED2')
    :param stats_df:  stats dataframe
    :param surveys: which surveys we are intrested in (BFI,NARS,GODSPEED1,GODSPEED2)
    :return: g: dataframe organized by questionnaires subscales.
    '''
    fig,ax = plt.subplots(1,1)
    g = stats_df[(stats_df['feature'] == 'GODSPEED1') | (stats_df['feature'] == 'GODSPEED2') | (stats_df['feature'] == 'BFI') | (
            stats_df['feature'] == 'NARS')][['feature', 'sub_scale', 'answers','gender','education','age']]
    g['feature'] = g[['feature', 'sub_scale']].sum(axis=1)
    g = g.drop('sub_scale', axis=1)

    d = {}
    for gg in g['feature'].unique():
        d[gg] = g[g['feature'] == gg].answers.tolist()
    dff = pd.DataFrame.from_dict(d)
    cnames = []
    for c in surveys:
        cnames += dff.columns[dff.columns.str.contains(c)].tolist()
    sns.pairplot(dff[cnames])
    save_maxfig(fig, 'pairplot_participant')

    # sns.pairplot(g, hue='feature')

    return g

def questionnaires_boxplot(qdf,y,x,hue):
    '''
    Plotting boxplot of the questionnaires answers by sub scale by hue.
    :param qdf: questionnaires dataframe
    :param y: which axis to get the answers for.
    :param x: the answers.
    :param hue: what to differentiate the answers.
    :return:
    '''
    fig,ax = plt.subplots(1,1)
    sns.boxplot(data=qdf, y=y, x=x, hue=hue, ax=ax)
    save_maxfig(fig, 'questionnaires_boxplot')


def preference_cinsistency(users_pref_tot, sf, ignore = True):
    '''
    show the dynamics of choice of all the users per question
    :param users_pref_tot: pref dataframe of the robots that were chosen in each question for each participant.
    :param sf: dicitionary that contains stats_df of all the combinations.
    :param ignore: Is there questions to ignore.
    :return:
    '''
    '''
    
    :param users_pref_tot:
    :return:
    '''
    users_pref_tot = users_pref_tot.replace('rational', 0)
    users_pref_tot = users_pref_tot.replace('half', 1)
    users_pref_tot = users_pref_tot.replace('irrational', 2)
    users_pref_tot = users_pref_tot.astype('int')

    fig,ax = plt.subplots(1,1)

    # creating custom colormap & pcolor usin matplotlib
    # colors = [(.36, .65, .93), (.36, .93, .65), (.925, .365, .365)]
    # cmap_name = 'my_list'
    # cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
    # p = ax.pcolor(users_pref_tot, edgecolors='k', cmap=cm)
    # cbar = fig.colorbar(p, ax=ax, ticks=[0.33, 1, 1.66])

    cm = sns.color_palette('bone', 3)
    sns.heatmap(users_pref_tot, cmap=cm, linewidths=.5, ax = ax,
                cbar_kws={'ticks':[0.33, 1, 1.66]})

    ax.set_xticks(np.arange(1,users_pref_tot.columns.__len__(),5))
    ax.set_xticklabels(np.arange(1,users_pref_tot.columns.__len__(),5))

    # ,
    # 'labels':['Rational', 'Half', 'Irrational']

    y_labels = users_pref_tot.index.tolist()
    N = y_labels.__len__()
    l = np.linspace(0.5, N - 0.5, N)
    ax.set_yticks(l)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Participants')
    ax.set_ylabel('Question')

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    if ignore:
        ax.hlines((l[3],l[4],l[6]), *ax.get_xlim(), lw = 4., linestyle='-.')
        ax.annotate('ignore this question', xy=(0+.5, l[3]), annotation_clip=False, fontsize=12, bbox=bbox_props)
        ax.annotate('ignore this question', xy=(0+.5, l[4]), annotation_clip=False, fontsize=12, bbox=bbox_props)
        ax.annotate('ignore this question', xy=(0+.5, l[6]), annotation_clip=False, fontsize=12, bbox=bbox_props)

    xl = 0
    for i in sf:
        cu = sf[i].userID.unique().__len__() # current users count
        xlm = ax.get_ylim()
        ax.annotate(i[-2:], xy=(xl + float(cu)/2, xlm[1]-.2), annotation_clip=False, fontsize=14)
        xl += cu
        ax.vlines(xl, *xlm, lw= 6.)


    cbar = fig.axes[1]
    cbar.set_yticklabels(['Rational', 'Single Irrationality', 'Double Irrationality']) # colorbar text

    save_maxfig(fig, 'users_preference_per_question')

def creating_dataframe4manova(sf, users_pref_tot, numeric = True):
    '''
    Creating a dataframe for manova.
    Each participants donate 2 points for each question because of the godspeed (rationality)
    :param sf: stats dataframe
    :param users_pref_tot: preference in the questions.
    :return: data frame for manova
    '''

    upt = users_pref_tot.transpose().reset_index(drop=True)
    stats_df = sf['rDeployment_tt']
    g = stats_df[(stats_df['feature'] == 'GODSPEED1') | (stats_df['feature'] == 'GODSPEED2') | (
            stats_df['feature'] == 'BFI') | (
                         stats_df['feature'] == 'NARS')][
        ['feature', 'sub_scale', 'answers', 'gender', 'education', 'age', 'userID','robot','rationality', 'side']]
    g['f'] = g['feature'] + '_' + g['sub_scale']
    g['f1'] = g['feature'] + '_' + g['sub_scale'] + '_' + g['robot'] +'_'+g['rationality']
    g = g.drop(['feature', 'sub_scale'], axis=1)

    for s in g.f.unique():
        temp_col = g[g.f == s].answers.tolist()
        if 'manova_df' not in locals():
            manova_df = g[g.f == s][['gender','education', 'age', 'userID']]


        manova_df.insert(loc = manova_df.columns.__len__(), column=s , value = temp_col)

    # data frame for 1st robot
    manova_df = manova_df.reset_index(drop=True)
    manova_df1 = manova_df.sort_values(by='userID')
    manova_df = pd.concat((upt,manova_df1), axis=1)

    manova_df2 = manovadf_drop_support(manova_df, 'GODSPEED1')
    manova_df2 = manovadf_robot_support(manova_df2, 'GODSPEED2_S1', g)

    # data frame for 2nd robot
    manova_df = manovadf_drop_support(manova_df, 'GODSPEED2')
    manova_df = manovadf_robot_support(manova_df, 'GODSPEED1_S1', g)

    manova_df2 = manova_df2.rename(columns=dict(zip(manova_df2.columns[-5:], manova_df.columns[-5:])))

    manova_df = manova_df.append(manova_df2)

    manova_df = manova_df.reset_index(drop=True)

    for c in upt.columns:
        manova_df[c] = (manova_df[c] == manova_df.rationality).astype('int')
    if numeric:
        cls = {'red':0, 'blue':1}
        rat = {'rational':0, 'half':1, 'irrational':2}
        sides = {'left':0, 'right':1}
        genders = {'female':0, 'male':1}
        edu = {'<HS':1, 'HS':2, '<BA':3, 'BA':4, 'MA':5, 'professional':6, 'PhD':7}
        rdict = {'rationality':rat, 'color':cls, 'side':sides, 'gender':genders, 'education':edu}
        for c in manova_df.columns[[0, 1, 2, 15, 16]]:
            manova_df.loc[:, c] = manova_df.loc[:, c].replace(rdict[c])
    return manova_df

def manovadf_robot_support(manova_df, s, g):
    '''
    Inserting robot's parmeters
    :param manova_df:
    :param s: feature
    :param g: dataframe we take the information from.
    :return:
    '''
    temp_col_rationality = g[g.f == s].rationality.tolist()
    temp_col_color = g[g.f == s].robot.tolist()
    temp_col_side = g[g.f == s].side.tolist()
    manova_df.insert(loc=0, column='side', value=temp_col_side)
    manova_df.insert(loc=0, column='color', value=temp_col_color)
    manova_df.insert(loc=0, column='rationality', value=temp_col_rationality)
    return manova_df

def manovadf_drop_support(manova_df, s):
    '''
    drop columns with s
    :param manovadf:
    :param s:
    :return:
    '''

    a = pd.DataFrame(manova_df.columns).astype('str')
    manova_df1 = manova_df.copy()
    manova_df1 = manova_df1.drop(manova_df.columns[a.loc[:,0].str.contains(s)], axis=1)

    return manova_df1

def word_cloud(open_answers_tot):
    '''
    creating word clouds by rationality.
    :param open_answers_tot: data frame with all the full answers
    :return:
    '''
    from wordcloud import WordCloud


    rationalities = open_answers_tot.rationality.unique()
    fig, ax = plt.subplots(1,rationalities.__len__())
    wordclouds = []
    for i, rat in enumerate(rationalities):
        # fig, ax = plt.subplots(1, 1)
        txt = open_answers_tot.loc[open_answers_tot.rationality == rat, 'answer'].str.cat(sep=' ')
        wordcloud = WordCloud(max_font_size=60, min_font_size=12).generate(txt)
        ax[i].imshow(wordcloud, interpolation="bilinear")
        ax[i].set_title(rat)
        ax[i].axis("off")
        wordclouds.append(wordcloud)

    a = set(wordclouds[0].words_.keys()) # rational words
    b = set(wordclouds[1].words_.keys()) # irrational words
    c = a.intersection(b).__len__()      # how many words in both?

    save_maxfig(fig, 'rationalities_wrodclouds')

def save_maxfig(fig, fig_name, save_pkl = 1, transperent = False, frmt='png', resize=None):
    '''
    Save figure in high resolution
    :param fig: which figure to save
    :param fig_name: name of the figure
    :param save_pkl: save the figure's data to pickle?
    :param transperent: background
    :param frmt: file's format
    :param resize: give size (width, height)
    :return:
    '''

    # matplotlib.rcParams.update({'font.size': 40})
    fig.set_size_inches(12, 12)
    if resize != None:
        fig.set_size_inches(resize[0], resize[1])
    p_fname = 'figs_files/'
    if not os.path.exists(p_fname):
        os.makedirs(p_fname)

    fig_name+='.'+frmt
    plt.savefig(p_fname+fig_name, dpi=300, transperent=transperent, format=frmt)
    fig.set_size_inches(5, 5)

    if save_pkl == 1:
        p_fname = p_fname + fig_name + 'fig.pckl'
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