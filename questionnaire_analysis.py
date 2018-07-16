from reformat_data import *
from inferential_analysis import *
import pickle
from matplotlib import style


def prepare_data(df_dir ):
    rDeployment_rh = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']
    rDeployment_tt = []
    rDeployment = {'rDeployment_ri': rDeployment_ri, 'rDeployment_rh': rDeployment_rh, 'rDeployment_ih': rDeployment_ih,'rDeployment_tt': rDeployment_tt}
    # rDeployment = {'rDeployment_ri': rDeployment_ri}

    # infer = False
    if reformat:
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

        pickle.dump(rDeployment, open(df_dir+'robot_deployments','wb'))

        r,c = users_pref_tot.shape
        temp = pd.DataFrame(data = np.zeros([c * r, 2]), columns = ['question','preference'])
        for i,row in enumerate(users_pref_tot.index):
            temp.loc[i * c : (i+1) * c - 1, 'preference'] = users_pref_tot.loc[row, :].tolist()
            temp.loc[i * c : (i+1) * c - 1, 'question']   = row
        temp.to_csv(df_dir + '__pref_df_long.csv')
        print('Done reformating the data.')

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


def rat3_df(rat_pref_df_tot):
    '''
    creating dataframe for comparision between 3 rationalities for each question
    :param rat_pref_df_tot: dataframe containing the preference for each rationality.
    :return:
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

    return rat3_df, binomal_df

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
    # reformat, infer = True, False
    reformat, infer = False, True

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

        rat_pref_df_tot_3rat, binomal_df = rat3_df(rat_pref_df_tot)
        rat_pref_df_tot_3rat.to_csv(df_dir+'__rat_pref_dataframe_3rat.csv')
        binomal_df.index = users_pref_tot.index
        binomal_df.to_csv(df_dir+'__binomal_df.csv')


    if infer:
        style.use(['ggplot', 'presentation'])
        # sf['rDeployment_tt'] = sf['rDeployment_ri'] # for only specific deployment

        print(np.unique(np.asarray(users_pref_tot),return_counts=True))

        # word_cloud(open_answers_tot)
        stacked_plot(users_pref_tot, rat_pref_df_tot, binomal_df, show_sig=False)

        # manova_df = creating_dataframe4manova(sf['rDeployment_tt'], users_pref_tot)
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

        for i in sf:
            stats_df = sf[i]
            # preference_plot(stats_df, 'sub_scale', 'summary', fname='_barplot_only_choices_'+i[-2:], p='default')
            # preference_plot(stats_df, 'sub_scale', 'summary', fname='_barplot_only_choices_'+i[-2:], p='default')
            # preference_plot(stats_df, 'sub_scale', 'summary', fname='_summary_'+i[-2:], p='deployment')
        #     qdf = pair_plot(stats_df, ['BFI','NARS'])
        #     questionnaires_boxplot(qdf, 'feature', 'answers', 'gender')
        # preference_cinsistency(users_pref_tot, sf, ignore = False)
        # preference_per_question(pref_df_tot)

    plt.show()

    #todo: word cloud for the open question
    # if rDep == 'rDeployment_tt':
    #     see_the_data(stats_df) # todo: contonue working on descriptive of the BFI, NARS and Godspeed

