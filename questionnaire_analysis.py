from reformat_data import *
from inferential_analysis import *


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

def create_full_stats_df(raw_df, fn):
    '''
    creating stats dataframe from full raw dataframe
    :param raw_df: raw data,
    :param fn - string for file name
    :return:
    '''


    return stats_df, users_after_exclusion


if __name__ == "__main__":
    # reformat, infer = True, False
    reformat, infer = False, True

    rDeployment_rh = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']
    rDeployment = {'rDeployment_ri': rDeployment_ri, 'rDeployment_rh': rDeployment_rh, 'rDeployment_ih': rDeployment_ih}

    rDeployment_tot = []

    # infer = False
    df_dir = 'data/dataframes/'
    if reformat:
        files = os.listdir('data/raw/')
        if not os.path.exists(df_dir):
            os.mkdir(df_dir)
        for f in files:
            path = 'data/raw/' + f
            raw_df, rDeployment1 = raw_data_extraction(path)
            rDeployment_tot += [rDeployment1.split('.')[0]]

        # rDeployment['rDeployment_tot'] = rDeployment_tot

        for i, rDep in enumerate(rDeployment):
            fn = '_'+rDep[-2:]
            raw_df = comine_raw_data2dataframe(rDeployment[rDep])
            raw_df.to_csv(df_dir+'raw_dataframe'+fn)

            # cleaning trap question
            before = raw_df[raw_df.columns[:-5]].columns.__len__()
            raw_df, users_after_exclusion = trap_exclusion1(raw_df)
            # raw_df, excluded_users = response_time_exclusion(raw_df, users_after_exclusion)
            excluded = before - users_after_exclusion.__len__()
            print('exclude:', excluded, 'out of', before)
            raw_df = raw_df.set_index(raw_df[raw_df.columns[0]])
            raw_df = raw_df.drop(raw_df.columns[0], axis=1)

            stats_df = create_stats_df(raw_df, fn)

            pref_df, users_pref = prefernce_dataframe_index(raw_df)

            #  crating preference dataframe
            if 'pref_df_tot' in locals():
                pref_df_tot = pref_df_tot.append(pref_df)
                users_pref_tot = pd.concat([users_pref_tot, users_pref],axis=1)
            else:
                pref_df_tot = pref_df.copy()
                users_pref_tot = users_pref.copy()

        pref_df_tot.to_csv(df_dir+'pref_dataframe')
        users_pref_tot.to_csv(df_dir+'users_pref_dataframe')

    else:
        rf, sf = {}, {}
        for rDep in rDeployment:
            fn = '_'+rDep[-2:]
            rf[rDep] = pd.read_csv(df_dir+'raw_dataframe'+fn, index_col=0)
            sf[rDep] = pd.read_csv(df_dir+'stats_dataframe'+fn, index_col=0)

        pref_df_tot = pd.read_csv(df_dir+'pref_dataframe', index_col=0)
        users_pref_tot = pd.read_csv(df_dir+'users_pref_dataframe', index_col=0)

    if infer:
        for i in sf:
            stats_df = sf[i]
            # preference_plot(stats_df, 'sub_scale', 'summary', fname='_barplot_only_choices')
            # preference_plot(stats_df, 'sub_scale', 'summary', fname='_summary', deployment=True)
            # qdf = pair_plot(stats_df, ['BFI','NARS'])
            # questionnaires_boplot(qdf, 'feature', 'answers', 'gender')
        preference_cinsistency(users_pref_tot, sf)
        # preference_per_question(pref_df_tot)

    plt.show()

    #
    # if rDep == 'rDeployment_tot':
    #     see_the_data(stats_df) # todo: contonue working on descriptive of the BFI, NARS and Godspeed

    print('t')