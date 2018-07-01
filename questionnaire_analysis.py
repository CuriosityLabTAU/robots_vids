from reformat_data import *
from inferential_analysis import *
import pickle

def prepare_data(df_dir ):
    rDeployment_rh = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']
    rDeployment_tt = []
    rDeployment = {'rDeployment_ri': rDeployment_ri, 'rDeployment_rh': rDeployment_rh, 'rDeployment_ih': rDeployment_ih,'rDeployment_tt': rDeployment_tt}
    rDeployment = {'rDeployment_ri': rDeployment_ri}

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
                pref_df, users_pref, open_answers = prefernce_dataframe_index(raw_df)

                #  crating preference dataframe
                if 'pref_df_tot' in locals():
                    pref_df_tot = pref_df_tot.append(pref_df)
                    users_pref_tot = pd.concat([users_pref_tot, users_pref],axis=1)
                    open_answers_tot = pd.concat([open_answers_tot, open_answers],axis=1)
                else:
                    pref_df_tot = pref_df.copy()
                    users_pref_tot = users_pref.copy()
                    open_answers_tot = open_answers.copy()

                    # reverse DON'T question answers
        # pref_df_tot.loc[pref_df_tot['question'] == 'Q16.1', 'preference'] = 1 - pref_df_tot.loc[pref_df_tot['question'] == 'Q16.1', 'preference']

        # saving dataframes of answers
        pref_df_tot.to_csv(df_dir+'pref_dataframe'+'.csv')
        users_pref_tot.to_csv(df_dir+'users_pref_dataframe'+'.csv')
        open_answers_tot = pd.DataFrame(dict(answer = open_answers_tot, rationality=users_pref_tot.loc['prefer', :].tolist()))
        open_answers_tot.index = users_pref_tot.columns
        open_answers_tot.to_csv(df_dir+'open_answers_dataframe'+'.csv')
        pickle.dump(rDeployment, open(df_dir+'robot_deployments','wb'))
        print('Done reformating the data.')

def comine_raw_data2dataframe(rDeployment):
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
    crdf
    return crdf


if __name__ == "__main__":
    # reformat, infer = True, False
    reformat, infer = False, True

    df_dir = 'data/dataframes/'

    if reformat:
        prepare_data(df_dir)
    else:
        rDeployment = pickle.load(open(df_dir + 'robot_deployments','rb'))

        # rDeployment_tot = rDeployment_ih+rDeployment_rh+rDeployment_ri
        rDeployment_tot = rDeployment['rDeployment_ri']
        rDeployment.pop('rDeployment_tt')
        # rDeployment['rDeployment_tt'] = rDeployment_tot

        rf, sf = {}, {}
        for rDep in rDeployment:
            fn = '_' + rDep[-2:] + '.csv'
            rf[rDep] = pd.read_csv(df_dir + 'raw_dataframe' + fn, index_col=0)
            sf[rDep] = pd.read_csv(df_dir + 'stats_dataframe' + fn, index_col=0)

        pref_df_tot = pd.read_csv(df_dir + 'pref_dataframe' + '.csv', index_col=0)
        users_pref_tot = pd.read_csv(df_dir + 'users_pref_dataframe' + '.csv', index_col=0)
        open_answers_tot = pd.read_csv(df_dir + 'open_answers_dataframe' + '.csv', index_col=0)


    if infer:
        print(np.unique(np.asarray(users_pref_tot),return_counts=True))

        word_cloud(open_answers_tot)

        sf['rDeployment_tt'] = sf['rDeployment_ri']
        manova_df = creating_dataframe4manova(sf, users_pref_tot)
        manova_df.to_csv(df_dir + 'manova_df_dataframe.csv')
        sf.pop('rDeployment_tt')
        for i in sf:
            stats_df = sf[i]
            preference_plot(stats_df, 'sub_scale', 'summary', fname='_barplot_only_choices')
            preference_plot(stats_df, 'sub_scale', 'summary', fname='_summary', deployment=True)
            # qdf = pair_plot(stats_df, ['BFI','NARS'])
            # questionnaires_boxplot(qdf, 'feature', 'answers', 'gender')
        preference_cinsistency(users_pref_tot, sf, ignore = False)
        preference_per_question(pref_df_tot) # todo: i think there is something wrong with the order - rationality_irrationality

    plt.show()

    #todo: word cloud for the open question
    # if rDep == 'rDeployment_tt':
    #     see_the_data(stats_df) # todo: contonue working on descriptive of the BFI, NARS and Godspeed

