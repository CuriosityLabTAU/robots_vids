from reformat_data import *
from inferential_analysis import *
from paper_plots_analysis import *
import pickle
from matplotlib import style
import seaborn as sns


sns.set_context('paper')

def prepare_data(df_dir):
    rDeployment_rh = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']
    rDeployment_tt = []
    rDeployment = {'rDeployment_ri': rDeployment_ri, 'rDeployment_rh': rDeployment_rh, 'rDeployment_ih': rDeployment_ih,'rDeployment_tt': rDeployment_tt}
    # rDeployment = {'rDeployment_ri': rDeployment_ri}

    files = os.listdir('data/raw/')
    if not os.path.exists(df_dir):
        os.mkdir(df_dir)
    for f in files:
        path = 'data/raw/' + f
        raw_df, rDeployment1 = raw_data_extraction(path)
        rDeployment_tt += [rDeployment1.split('.')[0]]

    rDeployment['rDeployment_tt'] = rDeployment_tt

    for i, rDep in enumerate(rDeployment):
        print('reformatting '+rDep)
        fn = '_'+rDep[-2:]+'.csv'
        raw_df = comine_raw_data2dataframe(rDeployment[rDep])
        raw_df.to_csv(df_dir+'raw_dataframe'+fn)

        # cleaning trap question
        before = raw_df[raw_df.columns[:-5]].columns.__len__()
        raw_df, users_after_exclusion = trap_exclusion1(raw_df)
        # raw_df, excluded_users = response_time_exclusion(raw_df, users_after_exclusion)
        excluded = before - users_after_exclusion.__len__()
        print('trap question excluded:', excluded, 'out of', before)
        raw_df = raw_df.set_index(raw_df[raw_df.columns[0]])
        raw_df = raw_df.drop(raw_df.columns[0], axis=1)

        stats_df = create_stats_df(raw_df, fn)

        if rDep != 'rDeployment_tt':
            pref_df, users_pref, open_answers, rat_pref_df = prefernce_dataframe_index(raw_df) # reverse DON'T question answers inside this question
            mdf, ___ = creating_dataframe4manova(stats_df, users_pref)
            mdf.to_csv('data/dataframes/mdf_'+rDep[-2:]+'.csv')

            #  crating preference dataframe
            if 'pref_df_tot' in locals():
                pref_df_tot = pref_df_tot.append(pref_df)
                users_pref_tot = pd.concat([users_pref_tot, users_pref],axis=1)
                open_answers_tot = pd.concat([open_answers_tot, open_answers],axis=0)
                rat_pref_df_tot  = rat_pref_df_tot.append(rat_pref_df)
            else:
                pref_df_tot = pref_df.copy()
                users_pref_tot = users_pref.copy()
                open_answers_tot = open_answers.copy()
                rat_pref_df_tot = rat_pref_df.copy()

        if rDep == 'rDeployment_tt':
            manova_df, mdf_small = creating_dataframe4manova(stats_df, users_pref_tot)

    # saving dataframes of answers
    pref_df_tot.to_csv(df_dir+'__pref_dataframe'+'.csv')
    rat_pref_df_tot.to_csv(df_dir+'__rat_pref_dataframe'+'.csv')
    users_pref_tot.to_csv(df_dir+'__users_pref_dataframe'+'.csv')
    open_answers_tot = pd.DataFrame(dict(answer = open_answers_tot, rationality=users_pref_tot.loc['prefer', :].tolist()))
    open_answers_tot.index = users_pref_tot.columns
    open_answers_tot.to_csv(df_dir+'__open_answers_dataframe'+'.csv')
    manova_df.to_csv(df_dir + '__manova_df_dataframe.csv')
    mdf_small.to_csv(df_dir + '__mdf_small.csv')

    # pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    # pt = PowerTransformer(method='yeo-johnson', standardize=False)
    # mdf_normalized = pt.fit_transform(manova_df)
    # mdf_small_normalized = pt.fit_transform(mdf_small)
    # mdf_small['GODSPEED_S1_normalized'] = _yeo_johnson_transform(mdf_small['GODSPEED_S1'])
    # mdf_normalized.to_csv(df_dir + '__normalized_manova_df.csv')
    # mdf_small_normalized.to_csv(df_dir + '__normalized_small_manova_df.csv')

    gnbp_diff_corr(df_dir)

    pickle.dump(rDeployment, open(df_dir+'robot_deployments','wb'))

    r,c = users_pref_tot.shape
    temp = pd.DataFrame(data = np.zeros([c * r, 2]), columns = ['question','preference'])
    for i,row in enumerate(users_pref_tot.index):
        temp.loc[i * c : (i+1) * c - 1, 'preference'] = users_pref_tot.loc[row, :].tolist()
        temp.loc[i * c : (i+1) * c - 1, 'question']   = row
    temp.to_csv(df_dir + '__pref_df_long.csv')
    print('Done reformating the data.')

def fal_rate(users_pref_tot):
    '''
    return how many people choose each fallacy.
    :return: fallacy rate per type
    '''
    idx = {'Q5.1': 'conj', 'Q7.1': 'conj', 'Q10.1': 'disj', 'Q12.1': 'conj', 'Q14.1': 'disj', 'Q16.1': 'disj',
           'Q18.1': 'disj'}
    df = users_pref_tot.iloc[:7, :227].rename(index = idx)
    fal_rate = df.T.apply(pd.value_counts)

    a = fal_rate['conj'].sum(axis=1).rename(index = {'half':'single_conj', 'irrational':'double_conj'})
    b = fal_rate['disj'].sum(axis=1).rename(index = {'half':'single_disj', 'irrational':'double_disj'})
    a = pd.DataFrame(a).T
    b = pd.DataFrame(b).T

    fal_rate = pd.concat([a, b], ignore_index=True).fillna(0).sum(axis=0)
    fal_rate = np.round(fal_rate / fal_rate.sum() * 100, 2)
    print(fal_rate)
    return fal_rate

def comine_raw_data2dataframe(rDeployment):
    '''
    combines raw dataframe to total dataframe.
    :param rDeployment: dictionary of all the deployments
    :return:
    '''
    for i, rd in enumerate(rDeployment):
        raw_path = 'data/dataframes/raw_dataframe_' + rd + '.csv'
        if os.path.isfile(raw_path):
            raw_df = pd.read_csv(raw_path)

            if 'crdf' in locals():
                a = raw_df[raw_df.columns[5:]]
                a.columns = (np.arange(a.columns.__len__()) + 1) + i * 100
                crdf = pd.concat([crdf, a], axis=1)
            else:
                crdf = raw_df
                ix = crdf.index
    return crdf.loc[ix,:]


def rat3_df(rat_pref_df_tot, df_dir):
    '''
    creating dataframe for comparision between 3 rationalities for each question
    :param rat_pref_df_tot: dataframe containing the preference for each rationality.
    :return:rat3_df - dataframe with all preference across al deployments togther
            binomal_df - dataframe with all preference for each deployment with binom_test calculation
            chi_square_dict - dictionary containing the data frames for performing chi_square
    '''
    nu = rat_pref_df_tot.num_users.unique().sum()
    binomal_df = pd.DataFrame(data=[], columns = ['rh', 'ri', 'ih', 'rih'])
    rat_pref_df_tot = rat_pref_df_tot.reset_index(drop=True)
    qs = rat_pref_df_tot.question.unique()
    for q in qs:
        t = rat_pref_df_tot[rat_pref_df_tot.question == q]
        rat = t.rationality.unique()
        bin_temp = []
        for r in rat:
            a = t[t.rationality == r]
            temp = pd.DataFrame(a.loc[a.index[0]]).transpose()
            temp.preference = a.preference.sum()
            if 'rat3_df' in locals():
                rat3_df = rat3_df.append(temp)
            else:
                rat3_df = temp.copy()

        temp_rat = rat3_df[rat3_df.question == q]
        bin_temp.append(stats.binom_test(x=temp_rat.loc[temp_rat.rationality == 'rational', 'preference'],
                                         n=temp_rat.loc[(temp_rat.rationality == 'rational') | (temp_rat.rationality == 'half'), 'preference'].sum(), p=.5))
        bin_temp.append(stats.binom_test(x=temp_rat.loc[temp_rat.rationality == 'rational', 'preference'],
                                         n=temp_rat.loc[(temp_rat.rationality == 'rational') | (temp_rat.rationality == 'irrational'), 'preference'].sum(), p=.5))
        bin_temp.append(stats.binom_test(x=temp_rat.loc[temp_rat.rationality == 'irrational', 'preference'],
                                         n=temp_rat.loc[(temp_rat.rationality == 'irrational') | (temp_rat.rationality == 'half'), 'preference'].sum(), p=.5))
        bin_temp.append(stats.binom_test(x=temp_rat.loc[temp_rat.rationality == 'irrational', 'preference'],
                                         n=temp_rat.loc[:, 'preference'].sum(), p=.5))

        if 'binomal_df' in locals():
            binomal_df = binomal_df.append(pd.DataFrame(data=[bin_temp], columns = binomal_df.columns))
        else:
            binomal_df = pd.DataFrame(data=[bin_temp], columns=binomal_df.columns)
    binomal_df.index = qs
    chi_square_dict = {}
    for dep in rat_pref_df_tot.deployment.unique():
        a = rat_pref_df_tot[rat_pref_df_tot.deployment == dep]
        chi_square_df = a.pivot(index='rationality', columns='question', values='preference')
        chi_square_dict[dep] = chi_square_df

        s, p = stats.chisquare(chi_square_df)
        a = pd.DataFrame(data = [p, s], columns = chi_square_df.columns)
        a.index = ['chi_square_p_value', 'chi_square_statistics']
        chi_square_df = chi_square_df.append(a)
        chi_square_df.to_csv(df_dir+'df4chi_squared_'+dep+'.csv')

    chi_square_dict['rational_half'].index = chi_square_dict['rational_irrational'].index
    cs_r_all_i = chi_square_dict['rational_half'] + chi_square_dict['rational_irrational']
    cs_all, pv_all = stats.chisquare(cs_r_all_i) # chisquare of rational- all irrational (without irrational - half)
    a = pd.DataFrame(data=[pv_all, cs_all], columns=cs_r_all_i.columns)
    a.index = ['chi_square_p_value', 'chi_square_statistics']
    cs_r_all_i = cs_r_all_i.append(a)
    cs_r_all_i.to_csv(df_dir + 'df4chi_squared_' + 'r_all_irr' + '.csv')

    return rat3_df, binomal_df, chi_square_dict

def creating_new_style():
    import matplotlib
    import os
    from shutil import copy2
    style_path = matplotlib.get_configdir()
    style_path = os.path.join(style_path, 'stylelib')

    if not os.path.exists(style_path):
        os.makedirs(style_path)

    copy2('presentation.mplstyle', style_path)


if __name__ == "__main__":
    reformat, infer = True, False
    # reformat, infer = False, True
    creating_new_style()
    df_dir = 'data/dataframes/'

    if reformat:
        prepare_data(df_dir)
    else:
        rDeployment = pickle.load(open(df_dir + 'robot_deployments','rb'))

        # rDeployment_tot = rDeployment_ih+rDeployment_rh+rDeployment_ri
        # rDeployment_tot = rDeployment['rDeployment_ri']
        # rDeployment.pop('rDeployment_tt')
        # rDeployment['rDeployment_tt'] = rDeployment_tot

        rf, sf = {}, {}
        for rDep in rDeployment:
            fn = '_' + rDep[-2:] + '.csv'
            rf[rDep] = pd.read_csv(df_dir + 'raw_dataframe' + fn, index_col=0)
            sf[rDep] = pd.read_csv(df_dir + 'stats_dataframe' + fn, index_col=0)


        pref_df_tot          = pd.read_csv(df_dir + '__pref_dataframe' + '.csv', index_col=0)
        users_pref_tot       = pd.read_csv(df_dir + '__users_pref_dataframe' + '.csv', index_col=0)
        open_answers_tot     = pd.read_csv(df_dir + '__open_answers_dataframe' + '.csv', index_col=0)
        manova_df            = pd.read_csv(df_dir + '__manova_df_dataframe.csv', index_col = 0)
        manova_df_small      = pd.read_csv(df_dir + '__mdf_small.csv', index_col = 0)
        rat_pref_df_tot      = pd.read_csv(df_dir+'__rat_pref_dataframe.csv', index_col = 0)

        rat_pref_df_tot_3rat, binomal_df, chi_square_dict = rat3_df(rat_pref_df_tot, df_dir)
        binom_7questions = binom_test_pref(rat_pref_df_tot)

        rat_pref_df_tot_3rat.to_csv(df_dir+'__rat_pref_dataframe_3rat.csv')
        binomal_df.to_csv(df_dir+'__binomal_df.csv')
        binom_7questions.to_csv(df_dir+'__binom_7questions.csv')


    if infer:
        # style.use(['ggplot', 'presentation'])

        style.use(['ggplot'])

        # sf['rDeployment_tt'] = sf['rDeployment_ri'] # for only specific deployment

        # gnbp_diff_corr(df_dir, plot=True) # most of the plots for the article
        #
        # print(np.unique(np.asarray(users_pref_tot),return_counts=True))

        # manova_df, manova_df_small = creating_dataframe4manova(sf['rDeployment_tt'], users_pref_tot)
        # sf.pop('rDeployment_tt')
        # only analyzing the choices of all the questions asked after each videeo.

        # for gs in manova_df.keys()[manova_df.keys().str.contains('GODSPEED')]:
        #     manova_df.keys()[manova_df.keys().str.contains('GODSPEED')]
        #     preference_plot(manova_df, 'sub_scale', 'summary', yy = gs, fname='_barplot_only_choices_'+'GS'+gs[-2:])

        # sff = sf['rDeployment_tt']
        # for gs in pd.Series(sff.sub_scale.unique())[pd.Series(sff.sub_scale.unique()).str.contains('Q')]:
        #     # fig, ax = plt.subplots(1,1)
        #     # sns.countplot(hue='robot', x='rationality', data=sff[sff.sub_scale == gs], ax = ax)
        #     # save_maxfig(fig, 'pref'+gs.replace('.','_'))
        #     preference_plot(sff, 'sub_scale', gs, fname='countplot_only_choices_'+gs.replace('.','_'), p = 'countplot')

        # for _, stats_df in sf.items():
        #     preference_plot(stats_df, 'sub_scale', 'summary', fname='_barplot_only_choices_'+_[-2:], p='default')
            # preference_plot(stats_df, 'sub_scale', 'summary', fname='_summary_'+_[-2:], p='deployment') # summary plot
        #     qdf = pair_plot(stats_df, ['BFI','NARS'])
        #     questionnaires_boxplot(qdf, 'feature', 'answers', 'gender')

        # preference_cinsistency(users_pref_tot, sf, ignore = False)
        # preference_per_question(pref_df_tot)
        fal_rate(users_pref_tot)

        # statistical_diff(df_dir)
        # q_pref_df = summary_diff(sf, df_dir)

        # con_prob = conditional_probability(manova_df_small, 'bartender', 'prefer')

        # word_cloud(open_answers_tot, cloud=0)
        # stacked_plot(users_pref_tot, rat_pref_df_tot, binomal_df)
        plt.show()

    # if rDep == 'rDeployment_tt':
    #     see_the_data(stats_df) # todo: continue working on descriptive of the BFI, NARS and Godspeed

