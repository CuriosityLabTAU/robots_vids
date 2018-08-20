import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap
# plt.style.use('ggplot')
from scipy.stats import mannwhitneyu

# plt.xkcd();
ft = 12 # annotate fontsize

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

def preference_plot(stats_df, column, option, fname, yy = 'answers', p='deployment'):
    '''
    plot preferenec
    :param stats_df:
    :param column: which column (ex. 'sub_scale')
    :param option: which option (ex. 'average')
    :return:
    '''
    m = 2
    cnames = ['education','robot','gender','side','sub_scale']
    if yy == 'answers':
        b = stats_df[stats_df[column] == option]
        b.answers.replace(np.nan, 0)
        n = int(np.ceil(float(cnames.__len__())/m))
        b.answers = b.answers.astype('float64')
    else:
        # for maniva_df
        cnames = ['education', 'gender', 'side', 'color']
        n = int(np.ceil(float(cnames.__len__()) / m))
        b = stats_df
    fig, ax = plt.subplots(m,n)
    for i, c in enumerate(cnames):
        axc = ax[int(i / n), int(i % n)]
        # (b.groupby(c).size() / 3).tolist()
        # d = b.groupby(c)['rationality']
        # d.groups.keys()
        if p == 'coutnplot':
            sns.countplot(hue = c, x='rationality', data = b, ax =axc)
        elif p == 'deployment':
            sns.barplot(x = c, y = 'answers', data = b, ax = axc)
        elif p == 'default':
            sns.barplot(hue = c,x = 'rationality', y = yy, data = b, ax = axc)

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

def creating_dataframe4manova(stats_df, users_pref_tot, numeric = True):
    '''
    Creating a dataframe for manova.
    Each participants donate 2 points for each question because of the godspeed (rationality)
    :param sf: stats dataframe
    :param users_pref_tot: preference in the questions.
    :return: data frame for manova
    '''

    upt = users_pref_tot.transpose().reset_index(drop=True)
    # stats_df = sf['rDeployment_tt']
    g = stats_df[(stats_df['feature'] == 'GODSPEED1') | (stats_df['feature'] == 'GODSPEED2')
                 | (stats_df['feature'] == 'BFI') | (stats_df['feature'] == 'NARS')]
    [['feature', 'sub_scale', 'answers', 'gender', 'education', 'age', 'userID','robot','rationality', 'side']]
    g['rationality'] = g['rationality'].astype('str')
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

    manova_df_small = manova_df.loc[:manova_df.shape[0]/2 - 1].copy()

    return manova_df, manova_df_small

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

def word_cloud(open_answers_tot, cloud = 1, inside = 0, number_of_words = 30):
    '''
    creating word clouds by rationality.
    :param open_answers_tot: data frame with all the full answers
    :param cloud: 1, plot cloud, 0 don't plot.
    :param inside: 1, not inside, 0 inside.
    :param number_of_words: N top number of words to analyze
    :return:
    '''
    from wordcloud import WordCloud
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid.inset_locator import inset_axes

    N = number_of_words # how many words to plot

    rationalities = open_answers_tot.rationality.unique()

    fig = plt.figure()
    gs = gridspec.GridSpec(4, rationalities.__len__())
    gs.update(hspace=0.5)

    wordclouds = []
    for i, rat in enumerate(rationalities):
        txt = open_answers_tot.loc[open_answers_tot.rationality == rat, 'answer'].str.cat(sep=' ')
        wordcloud = WordCloud(max_font_size=40, max_words=N).generate(txt)  # min_font_size=12,

        ax_wfreq = fig.add_subplot(gs[cloud*inside:4, i]) # freq axes
        if cloud != 0:
            # wordcloud
            if inside == 0:
                ax_wcloud = inset_axes(ax_wfreq,
                                       width='60%',  # width = 30% of parent_bbox
                                       height='60%',  # height : 1 inch
                                       loc=4)
            else:
                ax_wcloud = fig.add_subplot(gs[0, i])
                ax_wcloud.set_title(rat)
            ax_wcloud.imshow(wordcloud, cmap=plt.cm.gray, interpolation="bilinear")
            ax_wcloud.axis("off")


        # frequency plot
        words_names = wordcloud.words_.keys()
        words_count = wordcloud.words_.values()
        wc, wn = [], []
        for w in zip(words_names, words_count):
            wc.append(w[1])
            wn.append(w[0])

        wc.reverse()
        wn.reverse()

        ax_wfreq.set_ylabel('Top ' + str(N) + ' Words')
        ax_wfreq.set_xlabel('Frequency')
        ax_wfreq.set_title(rationalities[i])
        indexes = np.arange(N)
        width = .4
        ax_wfreq.barh(indexes, wc, width)
        ax_wfreq.set_yticks(indexes)
        ax_wfreq.set_yticklabels(wn)

        wordclouds.append(wordcloud)

    a = set(wordclouds[0].words_.keys()) # rational words
    b = set(wordclouds[1].words_.keys()) # irrational words
    c = a.intersection(b).__len__()      # how many words in both?

    save_maxfig(fig, 'rationalities_wrodclouds_freq')

    # if   open_answers_tot = pd.read_csv(df_dir + 'open_answers_dataframe_v1' + '.csv', index_col=0)
    # fig, ax = plt.subplots(1,1)
    # answers_counts = (open_answers_tot.groupby(['rationality'])['answer']
    #                      .value_counts(normalize=True)
    #                      .rename('percentage')
    #                      .mul(100)
    #                      .reset_index()
    #                      .sort_values('answer'))
    # sns.barplot(x="answer", y="percentage", hue="rationality", data=answers_counts, ax=ax)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

def stacked_plot(users_pref_tot,  rat_pref_df_tot, binomal_df, show_sig = True):
    '''
    Stacked barplot of choices per intuition questions,
    :param users_pref_tot: Dataframe of which robot each user chose in each question.
    :param rat_pref_df_tot: Dataframe rationalities preference.
    :param binomal_df: Dataframe of which rationalities are significantly different.
    :return:
    '''
    binomal_df = binomal_df.reset_index(drop=True)
    binomal_df.index = rat_pref_df_tot.question.unique().tolist()
    binomal_sig = binomal_df < .05
    fig, ax = plt.subplots(1,1)
    #
    # fig = plt.figure()
    # ax = fig.add_axes((.1, .4, .8, .5))

    qs = ['investments', 'analyst', 'jury', 'bartender', 'prefer']
    qs = users_pref_tot.index.tolist()
    nq = np.arange(0,qs.__len__())
    barWidth = .4
    yy1, yy2, yy3 = np.array([]), np.array([]), np.array([])
    deps = rat_pref_df_tot.deployment.unique()
    for i, q in enumerate(qs):
        temp = pd.value_counts(users_pref_tot.loc[q, :])
        y1, y2, y3 = temp['rational'], temp['half'], temp['irrational']
        yy1, yy2, yy3 = np.append(yy1, y1), np.append(yy2, y2), np.append(yy3, y3)
        tot = y1 + y2 + y3
        ax.annotate(str(np.round(float(y1) / tot * 100,1))+'%', xy=(i-barWidth/4 , y1 / 2), annotation_clip=False, fontsize=ft)
        ax.annotate(str(np.round(float(y2) / tot * 100,1))+'%', xy=(i-barWidth/4 , y1 + y2 / 2), annotation_clip=False, fontsize=ft)
        ax.annotate(str(np.round(float(y3) / tot * 100,1))+'%', xy=(i-barWidth/4 , y1 + y2 + y3 / 2), annotation_clip=False, fontsize=ft)


        if show_sig:
            y = (y1 + y2 / 2, y1 / 2, y1 + y2 / 2 - 8)
            x = i-barWidth/4
            for k, dep in enumerate(deps): # comparison on 3 groups
                dep1 = dep.split('_')[0][0] + dep.split('_')[1][0]
                print(k, dep, binomal_sig.loc[q,dep1])
                if binomal_sig.loc[q, dep1]:
                    # print(k, dep, '*'*k)
                    ax.annotate('*'*(k+1), xy=(x , y[k] - 8), annotation_clip=False, fontsize=ft)
            if binomal_sig.loc[q, 'rih']: # difference between rational and all the irrationaltogther
                ax.annotate('*'*4, xy=(x, y[1] - 16), annotation_clip=False, fontsize=ft)


            # significance *** plot
            t = rat_pref_df_tot[rat_pref_df_tot.question == q]

            # for k, dep in enumerate(deps): # based on the original two groups
            #     if t[t.deployment == dep].sig.values[0]:
            #         print(k, dep, '*'*k)
            #         ax.annotate('*'*(k+1), xy=(x + barWidth , y[k]), annotation_clip=False, fontsize=12)

    ax.bar(nq, yy1, width=barWidth, color = 'darkgrey', label = 'rational')
    ax.bar(nq, yy2, bottom=yy1, width=barWidth, color = 'slategrey', label = 'irrationalty type 1')
    ax.bar(nq, yy3, bottom=yy1 + yy2, width=barWidth, color = 'darkslategrey', label = 'irrationalty type 2')


    ax.vlines(i-.5, *ax.get_ylim(), lw = 4., linestyle='-.')
    ax.set_xticks(nq)
    ax.set_xticklabels(qs)
    ax.set_ylabel('Number of people')
    ax.set_xlabel('Question')
    box = ax.get_position() # get the axis position
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # shrink the axes for the legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # put legend outside the plot

    an = '* rational - type I\n** rational - type II\n*** type II - type I\n**** rational - (type II + type I)'
    fig.text(.78, .7, an, bbox={'facecolor':'lightgrey', 'alpha':0.5})

    save_maxfig(fig, 'stacked_choices_per_rationality', resize=(24,12))

def statistical_diff(df_dir):
    '''
    calculating statistical difference between the different questionnaires parameters
    :param df_dir: path to where the desired dataframes are
    :return: stats_diff.csv
    '''
    # loading dataframes for statistical analysis of all the couple combinations.


    mdfd = {}
    for f in os.listdir(df_dir):
        if ('mdf_' in f) & (len(f) == 10):
            mdfd[f.split('.')[0].split('_')[1]] = pd.read_csv(df_dir + f)


    st = pd.DataFrame({'deployment':[], 'measurement':[], 'statistics':[], 'p_value':[]})
    for key, m in mdfd.items():
        r1, r2 = m.rationality.unique()
        # for c in m.columns[m.columns.str.contains('GOD').tolist()]:
        for c in m.columns:
            # s, p = mannwhitneyu(m.query('rationality==str(r1)')[c], m.query('rationality==str(r2)')[c])
            # m.groupby('rationality') # learn how to use this for statistics.
            s, p = mannwhitneyu(m[m.rationality==r1][c], m[m.rationality==r2][c])
            st = st.append(pd.DataFrame(data = [[key, c, s, p]], columns = st.columns))
    st.to_csv(df_dir+'stats_diff_mannwhitney.csv')

    print(st)
    print('statistics saved to data/dataframes/stats_diff_mannwhitney.csv')

def summary_diff(sf, df_dir):
    '''
    calculating and plotting statistical difference between the summary
    :param stats_df: dictionary of statistical dataframe
    :return: summary_mannwhitney.csv and figure
    '''
    d = {} # preference dictionary
    categories = ['side', 'robot', 'rationality']
    st = pd.DataFrame({'deployment':[], 'category_by': [], 'statistics':[], 'p_value':[]})
    st = st[st.columns]
    for deployment, stats_df in sf.items():
        if deployment != 'rDeployment_tt':
            m = stats_df[stats_df.sub_scale == 'summary']
            m = m[m['feature'] == 'r_preference'] # todo: where the q_preference from?
            for categ in categories:
                psig = False
                r1, r2 = m[categ].unique()
                s, p = mannwhitneyu(m[m[categ] == r1]['answers'], m[m[categ] == r2]['answers'])
                st = st.append(pd.DataFrame(data=[[deployment[-2:], categ, p, s]], columns=st.columns))
                if (categ == 'rationality') and(p < 0.05):
                    psig = True

                if categ == 'rationality':
                    d[r1[0] + r2[0]] = m[m[categ] == r1]['answers'].reset_index(drop=True)
                    d[r2[0] + r1[0]] = m[m[categ] == r2]['answers'].reset_index(drop=True)


            # figure for the article
            fig, ax = plt.subplots(1,1)
            fig.canvas.set_window_title(deployment)
            sns.barplot(x='rationality', y='answers', data=m, ax=ax)
            if psig:
                cxt = ax.get_xticks()
                ax.hlines(ax.get_ylim()[1], cxt[0], cxt[1])
                ax.annotate('*', xy=(np.mean(cxt), ax.get_ylim()[1] + 0.005), annotation_clip=False, fontsize=14)
                # ax.hlines((l[3], l[4], l[6]), *ax.get_xlim(), lw=4., linestyle='-.')

            save_maxfig(fig, 'rationality_diff' + deployment[-2:])

    q_pref_df = pd.DataFrame.from_dict(d)

    b = pd.melt(q_pref_df, id_vars='group', value_vars=['hr', 'hi'], var_name='rationality', value_name='preference')
    b = b[~b['preference'].isna()]
    b['group'] = 1
    a = pd.melt(q_pref_df, id_vars='group', value_vars=['ir', 'ih'], var_name='rationality', value_name='preference')
    a['group'] = 2
    a = a[~a['preference'].isna()]

    q_pref_df1 = pd.concat([a, b])
    q_pref_df1['rationality'] = q_pref_df1['rationality'].str.replace('ih','i2')
    q_pref_df1['rationality'] = q_pref_df1['rationality'].str.replace('ir','i1')
    q_pref_df1['rationality'] = q_pref_df1['rationality'].str.replace('hi','i2')
    q_pref_df1['rationality'] = q_pref_df1['rationality'].str.replace('hr','i1')

    fig, ax = plt.subplots(1, 1)
    sns.barplot(data=q_pref_df1, x='group', y='preference', hue='rationality', ax=ax)
    # sns.barplot(data=q_pref_df, order=['hr', 'hi'], ax=ax)

    st.to_csv(df_dir+'summary_mannwhitney.csv')
    q_pref_df.to_csv(df_dir + '__q_pref_df.csv')
    print('statistics saved to data/dataframes/summary_mannwhitney.csv')
    print('questions preference saved to data/dataframes/__q_pref_df.csv')




def save_maxfig(fig, fig_name, transperent = False, frmt='png', resize=None):
    '''
    Save figure in high resolution
    :param fig: which figure to save
    :param fig_name: name of the figure
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