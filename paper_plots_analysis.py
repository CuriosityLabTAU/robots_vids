import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import pickle
# plt.style.use('ggplot')

from sklearn.preprocessing import PowerTransformer

from inferential_analysis import *
from reformat_data import bfi_reverse

sns.set_context('paper')

cnames_dict = {
    'NARS_S1': 'Interactions',
    'NARS_S2': 'Social_influence',
    'NARS_S3': 'Emotions',
    'BFI_S1': 'Extroversion',
    'BFI_S2': 'Agreeableness',
    'BFI_S3': 'Conscientiousness',
    'BFI_S4': 'Neuroticism',
    'BFI_S5': 'Openness',
    'GODSPEED1_S1': 'Anthropomorphism',
    'GODSPEED1_S2': 'Animacy',
    'GODSPEED1_S3': 'Likability',
    'GODSPEED1_S4': 'Intelligence',
    'GODSPEED1_S5': 'Safety',
    'preference_average': 'agreement'}

cnames_groups = {
    'Participant': ['Gender', 'Age', 'Education'],
    'Robot': ['Rationality', 'Color', 'Side'],
    'NARS': ['Interactions', 'Social_influence', 'Emotions'],
    'BFI': ['Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
    'GODSPEED': ['Anthropomorphism', 'Animacy', 'Likability', 'Intelligence', 'Safety'],
    'Roles': ['Investments', 'Analyst', 'Jury', 'Bartender'],
    'Choices': ['Agreement', 'Prefer']}

def create_df4paper(df_dir,  cnames_dict, cnames_groups, without12 = True):

    mdf = {}
    # find all dataframes of called mdf_*
    a = pd.Series(os.listdir(df_dir))
    a = np.array(a[a.str.contains('mdf')].tolist())
    a = a[~(a == '__mdf_small.csv')]

    for m in a:
        mdf[m.split('.')[0]] = pd.read_csv(df_dir + m, index_col=0)
        print()

    if without12:
        df4paper = mdf['mdf_rh'].append(mdf['mdf_ri'])
        df4paper.loc[df4paper['rationality'] != 0, 'rationality'] = 1
    else:
        df4paper = mdf['mdf_rh'].append(mdf['mdf_ri']).append(mdf['mdf_ih'])



    df4paper = df4paper.rename(index=str, columns = cnames_dict)
    df4paper.columns = df4paper.columns.str.capitalize()
    df4paper = df4paper.drop(df4paper.columns[list(df4paper.columns.str.contains('Q'))], axis = 1)

    df4paper_grouped = df4paper.groupby(['Rationality'])
    g1, g2 = list(df4paper_grouped.groups.keys())

    cn = cnames_groups['GODSPEED']
    d0 = df4paper_grouped.get_group(g1).reset_index(drop=True).sort_values(by = 'Userid') # rational group
    d2 = df4paper_grouped.get_group(g2).reset_index(drop=True).sort_values(by = 'Userid') # irrational group
    d1 = d0.copy()
    d1[cn] = d0[cn] - d2[cn]

    d1[cnames_groups['Choices'][0]] = d0[cnames_groups['Choices'][0]]

    df4paper_small = d1.copy()

    df4paper = df4paper.reset_index(drop=True)
    df4paper_small = df4paper_small.reset_index(drop=True)

    df4paper.to_csv(df_dir + 'df4paper.csv')
    df4paper_small.to_csv(df_dir + 'df4paper_small.csv')

    return df4paper, df4paper_small

def plot_likability_agreement(df4paper, df4paper_small, cnames_groups, save_dir):
    cnames = np.array(df4paper.columns)
    categories_columns = np.delete(cnames, [0, 1, 2, 8, 9, 10, 11])

    pair_plot = sns.pairplot(df4paper, y_vars=cnames_groups['GODSPEED'], x_vars=cnames_groups['Choices'], kind="reg")
    pair_plot.savefig(save_dir +'df4paper_pairplot')
    reg_names = cnames_groups['GODSPEED'] + cnames_groups['Choices']

    df4paper_small.loc[df4paper_small['Age'] == 'A43', 'Age'] = 43
    df4paper_small['Age'] = df4paper_small['Age'].astype('float32')
    df4paper_small.loc[:,['Education', 'Age']] = df4paper_small.loc[:,['Education', 'Age']].astype('float32')

    pair_plot = sns.pairplot(df4paper_small, y_vars=cnames_groups['GODSPEED'], x_vars=cnames_groups['Choices'], kind="reg")
    pair_plot.savefig(save_dir + 'df4paper_small_pairplot')

    n = 2
    for x in cnames_groups['Robot']:
        df4paper_grouped = df4paper.groupby(x)
        g1, g2 = list(df4paper_grouped.groups.keys())

        fig, ax = plt.subplots(5, n)
        # for i, cat in enumerate(categories_columns):
        for i, cat in enumerate(cnames_groups['GODSPEED'] + [cnames_groups['Choices'][0]]):
            cax = ax[int(i/n), i%n]
            agree_bar  = sns.barplot(y = cat, x = x, data = df4paper, ax=cax)

            y1 = df4paper_grouped.get_group(g1).sort_values(by='Userid')[cat]
            y2 = df4paper_grouped.get_group(g2).sort_values(by='Userid')[cat]

            # print(x, cat, g1, g2)

            paired = False
            test_value, p_value, ttest, test_typ = significance_plot_between2bars(y1, y2, cax, paired)
            if cat == 'Agreement':
                # n = y1.shape[0]
                # x = temp1.preference.sum()
                # bt = stats.binom_test(x=x, n=n, p=.5)
                paired = True
                ttest = False
                test_typ = 'wilcoxon'
                test_value, p_value = stats.wilcoxon(y1)
                g2 = g1
                cat = 'agreement with rational'

            temp_stats = pd.DataFrame.from_dict({'behavior1': [g1], 'behavior2': [g2], 'measurement': [cat],
                                                 'mean1': [y1.mean()], 'std1': [y1.std()], 'mean2': [y2.mean()], 'std2': [y2.std()],
                                                 'test_type': [test_typ], 'test_value': [test_value],'p_value': [p_value]})

            if 'df_diff_test' not in locals():
                df_diff_test = temp_stats.copy()
            else:
                df_diff_test = pd.concat([df_diff_test, temp_stats], axis=0)

        df_diff_test = df_diff_test.reset_index(drop=True)
        save_table(df_diff_test, save_dir, 'df_diff_test_'+x)
        save_maxfig(fig, 'diff_' + x, p_fname=save_dir)
        # fig2plotly(fig, save_dir + 'diff_' + x)

        del(df_diff_test)

    pvalues, df_corr = calculate_corr_with_pvalues(df4paper_small, method = 'reg')
    save_table(df_corr, save_dir, 'df_corr')

    corr_godspeed = df_corr.loc[cnames_groups['NARS']+cnames_groups['BFI'] + [cnames_groups['Choices'][0]], cnames_groups['GODSPEED']].copy()
    corr_bfi      = df_corr.loc[cnames_groups['NARS']+ [cnames_groups['Choices'][0]], cnames_groups['BFI']].copy()
    corr_nars     = df_corr.loc[[cnames_groups['Choices'][0]], cnames_groups['NARS']].copy()
    save_table(corr_godspeed, save_dir, 'df_corr_godspeed')
    save_table(corr_bfi, save_dir, 'df_corr_bfi')
    save_table(corr_nars, save_dir, 'df_corr_nars')

    return df4paper_small

def calculae_chisquare(df4paper_small, cnames_groups, save_dir):
    df4chi = {}
    for col in cnames_groups['Roles'] + [cnames_groups['Choices'][1]]:
        df4chi[col] = pd.value_counts(df4paper_small[col])

    df4chi = pd.DataFrame.from_dict(df4chi)

    s, p = stats.chisquare(df4chi)
    a = pd.DataFrame(data=[p, s], columns=df4chi.columns)
    a.index = ['chi_square_p_value', 'chi_square_statistics']
    chi_square_df = df4chi.append(a)
    save_table(chi_square_df, save_dir, 'df4chi_squared')
    return df4chi

def significance_plot_between2bars(y1,y2, cax, paired = True):
    s, p, ttest, typ = ttest_or_mannwhitney(y1, y2, paired)

    if (p < 0.05):
        stars = pvalue_stars(p)
        cxt = cax.get_xticks()
        cax.hlines(cax.get_ylim()[1], cxt[0], cxt[1])
        if ttest:
            cax.annotate(stars, xy=(np.mean(cxt), cax.get_ylim()[1] + 0.001), annotation_clip=False, fontsize=14)
        else:
            cax.annotate(stars, xy=(np.mean(cxt), cax.get_ylim()[1] + 0.001), annotation_clip=False, fontsize=14)

    return s, p, ttest, typ

def fig2plotly(fig, path):
    import plotly.tools as tls
    import plotly
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly.offline.plot(plotly_fig, path, auto_open=False)

def save_table(df, df_dir,file_name, csv = True, Latex = True):
    if csv:
        df.to_csv(df_dir + file_name + '.csv')

    if Latex:
        df = df.round(3)
        with open(df_dir + file_name+ '.tex', 'w') as tf:
            tf.write(df.to_latex())

def calculate_corr_with_pvalues(df, method = 'pearsonr'):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')

    rho = df.corr()

    for r in df.columns:
        for c in df.columns:
            if method == 'pearsonr':
                pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            elif method == 'spearman':
                pvalues[r][c] = round(stats.spearmanr(df[r], df[c])[1], 4)
            elif method == 'reg':
                slope, intercept, rho[r][c], pvalues[r][c], std_err = stats.linregress(x=df[r],y= df[c])

    rho = rho.round(2)
    pval = pvalues
    # create three masks
    r1 = rho.applymap(lambda x: '{}*'.format(x))
    r2 = rho.applymap(lambda x: '{}**'.format(x))
    r3 = rho.applymap(lambda x: '{}***'.format(x))
    # apply them where appropriate
    rho = rho.mask(pval <= 0.05, r1)
    rho = rho.mask(pval <= 0.01, r2)
    rho = rho.mask(pval <= 0.001, r3)

    return pvalues, rho

def binom_(df):
    pass
    # n = temp['num_users'][0] * temp1.shape[0]
    # x = temp1.preference.sum()

    x = df.Agreement
    x *= 7
    n = df.shape[0] * 7

    bt = stats.binom_test(x=x, n=n, p=.5)
    print('binom test = ', bt)

def transform4mancova(df, save_dir, file_name, ncolumns = None):
    '''

    :param df: manovav_df - or any data frames that you want to transform to a normal distribution.
    :param ncolumns: Which columns to normalize
    :return:
    '''
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    if ncolumns == None:
        ncolumns = df.columns[df.columns.str.contains('NARS') | df.columns.str.contains('BFI') | df.columns.str.contains('GODSPEED')]
    df[ncolumns] = pd.DataFrame(data = pt.fit_transform(df[ncolumns]), columns = ncolumns)
    df.to_csv(save_dir + 'normalized_' + file_name + '.csv')

def cronbach_analysis(df, save_dir, *args, **kwargs):
    '''
    Cronbach analysis of all sub scales.
    :param df:
    :param save_dir:
    :param args:
    :param kwargs:
    :return:
    '''

    df = df.fillna(df.mean())

    BFI_rev = {'S1':[0,1,0,0,1,0,1,0], 'S2':[1,0,1,0,0,1,0,1,0], 'S3':[0,1,0,1,1,0,0,0,1],'S4':[0,1,0,0,1,0,1,0],'S5':[0,0,0,0,0,0,1,0,1,0]}

    questionnaires = {
        'BFI' : {
            'S1': {'meaning': 'Extraversion',
                   'indices': [1,6,11,16,21,26,31,36]},
            'S2': {'meaning': 'Agreeableness',
                   'indices': [2,7,12,17,22,27,32,37,42]},
            'S3': {'meaning': 'Conscientiousness',
                   'indices': [3,8,13,18,23,28,33,38,43]},
            'S4': {'meaning': 'Neuroticism',
                   'indices': [4,9,14,19,24,29,34,39]},
            'S5': {'meaning': 'Openness',
                   'indices': [5,10,15,20,25,30,35,40,41,44]},
        },
        'NARS' : {
            'S1': {'meaning': 'Interactions',
                   'indices': [4,7,8,9,10,12]},
            'S2': {'meaning': 'Social',
                   'indices': [1,2,11,13,14]},
            'S3': {'meaning': 'Emotions',
                   'indices': [3,5,6]},
        },
        'Godspeed' : {
            'S1': {'meaning': 'Anthropomorphism',
                   'indices': np.arange(1,6)},
            'S2': {'meaning':'Animacy',
                   'indices': np.arange(6, 12)},
            'S3': {'meaning': 'Likeability',
                   'indices': np.arange(12, 17)},
            'S4': {'meaning': 'Perceived Intelligence',
                   'indices': np.arange(17, 22)},
            'S5': {'meaning': 'Perceived Safety',
                   'indices': np.arange(22, 25)},
        }
    }

    for n in questionnaires['Godspeed']:
        questionnaires['Godspeed'][n]['indices'] = map(str, questionnaires['Godspeed'][n]['indices'])

    bfi_cols      = df.columns[df.columns.str.contains('BFI')]
    nars_cols     = df.columns[df.columns.str.contains('NARS')]
    godspeed_cols = df.columns[df.columns.str.contains('GODSPEED')]

    ### reversinfg the columns
    rev_cols = []
    for k, v in questionnaires['BFI'].items():
        rev = np.array(BFI_rev[k], dtype=bool)
        v = np.array(v['indices'])
        sub_scales2rev = v[rev]
        for s in sub_scales2rev:
            rev_cols += list(bfi_cols[bfi_cols.str.contains('_'+str(s))])

    for s in questionnaires['NARS']['S3']['indices']:
        rev_cols += list(nars_cols[nars_cols.str.contains('_' + str(s))])

    rev_cols += list(godspeed_cols[godspeed_cols.str.contains('_' + str(24))])

    df[rev_cols] = df[rev_cols].applymap(bfi_reverse)

    df_cronbach_alpha = pd.DataFrame.from_dict({'sub_scale': [], 'cronbach_alpha': []})
    for name, questionnaire in questionnaires.items():
        df_cronbach_alpha = subscales(questionnaire, df, name, df_cronbach_alpha)

    df.to_csv(save_dir + 'raw_dataframe_scales_cronbach.csv')
    df_cronbach_alpha = df_cronbach_alpha.reset_index(drop = True)
    df_cronbach_alpha.to_csv(save_dir + ' df_cronbach_alpha.csv')
    print()

def subscales(d, df, name, df_cronbach_alpha):
    '''
    calculating cronbach alpha for all the sub scales
    :param d: dict for BFI/ NARS/ Godspeed
    :param df: dataframe with raw data in columns
    :param name: which questionnaire
    :param df_cronbach_alpha: dataframe for saving the results.
    :return:
    '''
    for s, v in d.items():
        sub_scale = []
        idx = v['indices']
        for i in idx:
            if name != 'Godspeed':
                sub_scale += [name + '_' + str(i)]
            else:
                sub_scale += ['GODSPEED1_' + str(i), 'GODSPEED2_' + str(i)]

        ca = CronbachAlpha(df[sub_scale].T)
        # df[sub_scale].isnull().values.any()
        ca_df = pd.DataFrame.from_dict({'sub_scale':[name+s], 'cronbach_alpha':[ca]})

        df_cronbach_alpha = df_cronbach_alpha.append(ca_df)

    return df_cronbach_alpha


def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))

def nars_low_high(df4paper, cnames_groups, save_dir, method = 'median', q = 3):
    '''
    check godspeed ranking by high low nars.
    :param df4paper: data frame that I work with
    :param cnames_groups: dictionary with columns names organized by subject
    :param save_dir: where to save the data to.
    :param method: how to divide the data.  median/ mean/ qcut
    :param q: if method == 'qcut', how many quartiles to divide the data to.
    :return:
    '''
    df0 = df4paper.copy()
    if method =='median':
        df0[cnames_groups['NARS']] = np.array(df0[cnames_groups['NARS']] - df0[cnames_groups['NARS']].median() > 0, dtype = int)
    if method == 'mean':
        df0[cnames_groups['NARS']] = np.array(df0[cnames_groups['NARS']] - df0[cnames_groups['NARS']].mean() > 0, dtype=int)
    if method == 'mode':
        df0[cnames_groups['NARS']] = np.array(df0[cnames_groups['NARS']] - df0[cnames_groups['NARS']].mode() > 0, dtype=int)
    if method == 'qcut':
        for cnars in cnames_groups['NARS']:
            a = pd.qcut(x=df0[cnars], q=q)
            idx_high = a[a == a.unique()[-1]].index
            idx_low  = a[a != a.unique()[-1]].index
            df0.loc[idx_high, cnars] = 1
            df0.loc[idx_low,  cnars] = 0
    n = 2
    # for x in cnames_groups['Robot']:
    for x in ['Rationality']:
        for nars in cnames_groups['NARS']:
            for nars_cat in df0[nars].unique():
                df = df0[df0[nars] == nars_cat]
                df4paper_grouped = df.groupby(x)
                g1, g2 = list(df4paper_grouped.groups.keys())

                fig, ax = plt.subplots(5, n)
                # for i, cat in enumerate(categories_columns):
                for i, cat in enumerate(cnames_groups['GODSPEED'] + cnames_groups['Choices']):
                    cax = ax[int(i/n), i%n]
                    agree_bar  = sns.barplot(y = cat, x = x, data = df, ax=cax)

                    y1 = df4paper_grouped.get_group(g1).sort_values(by='Userid')[cat]
                    y2 = df4paper_grouped.get_group(g2).sort_values(by='Userid')[cat]

                    # print(x, cat, g1, g2)

                    paired = False
                    test_value, p_value, ttest, test_typ = significance_plot_between2bars(y1, y2, cax, paired)
                    if cat == 'Agreement':
                        # n = y1.shape[0]
                        # x = temp1.preference.sum()
                        # bt = stats.binom_test(x=x, n=n, p=.5)
                        paired = True
                        ttest = False
                        test_typ = 'wilcoxon'
                        test_value, p_value = stats.wilcoxon(y1)
                        g2 = g1
                        cat = 'agreement with rational'

                    temp_stats = pd.DataFrame.from_dict({'behavior1': [g1], 'behavior2': [g2], 'measurement': [cat],
                                                         'mean1': [y1.mean()], 'std1': [y1.std()], 'mean2': [y2.mean()], 'std2': [y2.std()],
                                                         'test_type': [test_typ], 'test_value': [test_value],'p_value': [p_value]})

                    if 'df_diff_test' not in locals():
                        df_diff_test = temp_stats.copy()
                    else:
                        df_diff_test = pd.concat([df_diff_test, temp_stats], axis=0)

                df_diff_test = df_diff_test.reset_index(drop=True)
                save_table(df_diff_test, save_dir, 'df_diff_test_'+x +'_'+ nars +'_' + str(nars_cat))
                save_maxfig(fig, 'diff_' +x +'_'+ nars +'_' + str(nars_cat), p_fname=save_dir)
                # fig2plotly(fig, save_dir + 'diff_' + x)

                del(df_diff_test)




def main():
    reformat_data = True
    # reformat_data = False
    df_dir = 'data/dataframes/'
    save_dir = 'data/paper/'
    # manova_df = pd.read_csv(df_dir + '__manova_df_dataframe.csv', index_col=0)
    # manova_df_small = pd.read_csv(df_dir + '__mdf_small.csv', index_col=0)
    raw_questionnaires_answers = pd.read_csv(df_dir + 'raw_dataframe_scales_cronbach.csv')
    cronbach_analysis(raw_questionnaires_answers, save_dir)

    if reformat_data:
        df4paper, df4paper_small = create_df4paper(df_dir, cnames_dict, cnames_groups, without12=True)
    else:
        df4paper       = pd.read_csv(df_dir + 'df4paper.csv', index_col=0)
        df4paper_small = pd.read_csv(df_dir + 'df4paper_small.csv', index_col=0)

    # # transform the data into normal distribution.
    # transform4mancova(manova_df, save_dir, 'manova_df')
    # transform4mancova(manova_df_small, save_dir, 'manova_df_small')
    # transform4mancova(df4paper, save_dir, 'df4paper', ncolumns = cnames_groups['NARS'] + cnames_groups['GODSPEED'] + cnames_groups['BFI'])
    # transform4mancova(df4paper_small, save_dir, 'df4paper_small', ncolumns = cnames_groups['NARS'] + cnames_groups['GODSPEED'] + cnames_groups['BFI'])

    # pref_df_tot = pd.read_csv(df_dir + '__pref_dataframe' + '.csv', index_col=0)
    # users_pref_tot = pd.read_csv(df_dir + '__users_pref_dataframe' + '.csv', index_col=0)
    # open_answers_tot = pd.read_csv(df_dir + '__open_answers_dataframe' + '.csv', index_col=0)
    # rat_pref_df_tot = pd.read_csv(df_dir + '__rat_pref_dataframe.csv', index_col=0)


    # df4chi = calculae_chisquare(df4paper_small, cnames_groups, save_dir)
    # plot_likability_agreement(df4paper, df4paper_small, cnames_groups, save_dir)
    # nars_low_high(df4paper, cnames_groups, save_dir, method = 'qcut', q = 7)

    # a = manova_df[manova_df.columns[manova_df.columns.str.contains('GODSPEED')]]
    # b = manova_df_small[manova_df_small.columns[manova_df_small.columns.str.contains('NARS')]]
    # c = manova_df_small[manova_df_small.columns[manova_df_small.columns.str.contains('BFI')]]
    # CronbachAlpha(a.T)
    # CronbachAlpha(b.T)
    # CronbachAlpha(c.T)

    # # binom_(df4paper_small)
    #
    plt.show()

if __name__ == '__main__':
    main()