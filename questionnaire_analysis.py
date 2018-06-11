from reformat_data import *
from inferential_analysis import *


def comine_raw_data2dataframe(rDeployment):
    for i, rd in enumerate(rDeployment):
        raw_path = 'data/raw_dataframe_' + rd + '.csv'
        if os.path.isfile(raw_path):
            raw_df = pd.read_csv(raw_path)

            if 'crdf' in locals():
                a = raw_df[raw_df.columns[5:]]
                a.columns = (np.arange(a.columns.__len__()) + 1) + i * 100
                crdf = pd.concat([crdf, a], axis=1)
            else:
                crdf = raw_df
    return crdf

def create_full_stats_df(raw_df, rDep):
    '''
    creating stats dataframe from full raw dataframe
    :param raw_df: raw data, dataframe
    :return:
    '''
    before = raw_df[raw_df.columns[:-5]].columns.__len__()
    raw_df, users_after_exclusion = trap_exclusion1(raw_df)
    # raw_df, excluded_users = response_time_exclusion(raw_df, users_after_exclusion)
    excluded = before - users_after_exclusion.__len__()
    print('exclude:', excluded, 'out of', before)

    raw_df = raw_df.set_index(raw_df[raw_df.columns[0]])
    raw_df = raw_df[raw_df.columns[1:]]

    stats_df = create_stats_df(raw_df, rDep)

    return stats_df, users_after_exclusion


if __name__ == "__main__":
    rDeployment_rh = ['rrrlbh', 'rbhlrr', 'rbrlrh', 'rrhlbr']
    rDeployment_ih = ['rbilrh', 'rrhlbi', 'rrilbh', 'rbhlri']
    rDeployment_ri = ['rbilrr', 'rbrlri', 'rrilbr', 'rrrlbi']

    reformat = True
    # reformat = False
    fn = '_combined.csv'
    rDeployment_tot = []

    infer = True
    # infer = False

    if reformat:
        files = os.listdir('data/raw/')
        for f in files:
            path = 'data/raw/' + f
            raw_df, rDeployment = raw_data_extraction(path)
            rDeployment_tot += [rDeployment.split('.')[0]]
        print('raw_df')
        # rDeployment = {'rDeployment_tot': rDeployment_tot}

        rDeployment = {'rDeployment_ri': rDeployment_ri,'rDeployment_rh': rDeployment_rh, 'rDeployment_ih': rDeployment_ih, 'rDeployment_tot': rDeployment_tot}

        for rDep in rDeployment:
            raw_df = comine_raw_data2dataframe(rDeployment[rDep])
            raw_df.to_csv('data/raw_dataframe'+fn)

            stats_df, users_after_exclusion = create_full_stats_df(raw_df, fn)

            preference_plot(stats_df, 'sub_scale', 'summary', fname='_barplot_only_choices')
            preference_plot(stats_df, 'sub_scale', 'summary', fname='_summary', deployment=True)
    else:
        raw_df = pd.read_csv('data/raw_dataframe'+fn, index_col=0)
        stats_df = pd.read_csv('data/stats_dataframe_'+fn, index_col=0)

    # if infer:
    #     preference_plot(stats_df, 'sub_scale', 'summary', fname = '_barplot_only_choices')
    #     preference_plot(stats_df, 'sub_scale', 'summary', fname = '_summary', deployment=True)

    plt.show()

    #
    # if rDep == 'rDeployment_tot':
    #     see_the_data(stats_df)

    print('t')