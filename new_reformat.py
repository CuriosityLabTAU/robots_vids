# data libraries
import pandas as pd
import numpy as np
import os
# statistics libraries
from scipy import stats
import statsmodels.formula.api as smf
# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# questions organizer
questions = {'BFI': 'Q2.1',
             'NARS': 'Q3.1',
             'Godspeed_Red': 'Q20.2#1',
             'Godspeed_Blue': 'Q20.2#2',
             'Gender': 'Q1.3',
             'Age' : 'Q1.4',
             'Education' : 'Q1.5',
             'trap': 'Q8.1',
             'dont': 'Q16.1',
             'qns' : ['Q5.1','Q7.1','Q10.1','Q12.1','Q14.1', 'Q18.1'],
             'fallacies':{'conj': ['Q5.1','Q7.1','Q12.1'],
                          'disj': ['Q10.1','Q14.1','Q16.1', 'Q18.1']},
             'rules': ['investments', 'jury', 'analyst', 'bartender']}

cnames_groups = {
    'Participant': ['Gender', 'Age', 'Education'],
    'Robot': ['Rationality', 'Color', 'Side'],
    'NARS': ['Interactions', 'Social_Influence', 'Emotions'],
    'BFI': ['Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
    'GODSPEED': ['Anthropomorphism', 'Animacy', 'Likability', 'Intelligence', 'Safety'],
    'Roles': ['Investments', 'Analyst', 'Jury', 'Bartender'],
    'Choices': ['agree2rational', 'prefer']}

d = {'colors': {'b': 'Blue', 'r': 'Red'}, 'side': {'l': 'left', 'r': 'right'},
     'rationality': {'r': 'rational', 'h': 'half', 'i': 'irrational'}}

bfi_options = {'Disagree strongly' : 1, 'Disagree a little' : 2, 'Neither agree nor disagree':3,
           'Agree a little':4, 'Agree strongly':5}

nars_options = {'Strongly disagree' : 1, 'Disagree' : 2, 'Undecided':3,
           'Agree':4, 'Strongly agree':5}

godspeed_options = {'Answer 1' : 1, 'Answer 2' : 2, 'Answer 3':3,
           'Answer 4':4, 'Answer 5':5}


def prepare_data(df_dir):
    # raw_dir = 'data/raw/'
    raw_dir = 'data/raw_text/'
    files = os.listdir(raw_dir)
    if not os.path.exists(df_dir):
        os.mkdir(df_dir)
    for f in files:
        path = raw_dir + f
        df = data_processing(path)

        if 'raw_df' in locals():
            raw_df = raw_df.append(df)
            raw_df = raw_df.reset_index(drop=True)
        else:
            raw_df = df.copy()

    fal_rate = fallacy_rate(raw_df)
    ### replacing all the halfs so that there will be only ir/rational rationalities
    raw_df = raw_df.replace({'half': 'irrational'})
    raw_df = raw_df.replace({'rational': 0, 'irrational': 1})
    raw_df.to_csv(df_dir + 'new_raw_df.csv')

    return raw_df

def data_processing(path):

    print('processing: ' + path)
    comb = path.split('_')[-1].split('.')[0]

    ### color and rationality dictionary.
    robots_comb = {d['colors'][comb[1]]: d['rationality'][comb[2]],
                   d['colors'][comb[4]]: d['rationality'][comb[5]]}

    rat_comb = {d['rationality'][comb[2]] : {'color' : d['colors'][comb[1]],
                                             'side'  : d['side'][comb[0]]},
                d['rationality'][comb[5]]: {'color': d['colors'][comb[4]],
                                            'side': d['side'][comb[3]]}}

    irr = list(set(rat_comb.keys()) - set(['rational']))[0]

    # This way there is only rational/irrational in robots rationality
    rat_comb['irrational'] = rat_comb.pop(irr)

    ### reversed combination of color and rationality for the dont question
    rev_robots_comb = {d['colors'][comb[1]]: d['rationality'][comb[5]],
                       d['colors'][comb[4]]: d['rationality'][comb[2]]}

    raw_df = pd.read_csv(path)

    ### Find specific questions columns
    agree_questions = raw_df.columns[raw_df.iloc[0, :].str.contains('Which robot do you agree with?')]
    dont_question   = raw_df.columns[raw_df.iloc[0, :].str.contains('DON\'T')]
    rules_question  = raw_df.columns[raw_df.iloc[0, :].str.contains('- Which robot')]
    prefer_question = raw_df.columns[raw_df.iloc[0, :].str.contains('Which robot would you prefer?')]
    trap_question   = raw_df.columns[raw_df.iloc[0, :].str.contains('What does Liz enjoy doing?')]

    df_choices = pd.DataFrame()
    ### replacing color with raationality
    ### todo: there was no 'DON'T question in rrrlbi - wired!!!
    if agree_questions.__len__() == 7:
        df_choices[agree_questions] = raw_df[agree_questions].replace(robots_comb)
    else:
        df_choices[questions['qns']]   = raw_df[agree_questions].replace(robots_comb)
        df_choices[questions['dont']]  = raw_df[dont_question].replace(rev_robots_comb)

    df_choices[questions['rules']] = raw_df[rules_question].replace(robots_comb)
    df_choices['color']            = raw_df[prefer_question]
    df_choices['prefer']           = raw_df[prefer_question].replace(robots_comb)

    ### get users id
    df_choices['id']           = raw_df['survey_code']

    ### adding the demographic information to the dataframe
    age_col       = raw_df.columns[raw_df.iloc[0, :].str.contains('What is your age?')]
    gender_col    = raw_df.columns[raw_df.iloc[0, :].str.contains('To which gender identity do you most identify?')]
    education_col = raw_df.columns[raw_df.iloc[0, :].str.contains('What is the highest degree')]

    df_choices['age'] = raw_df.loc[:, age_col]
    df_choices['gender'] = raw_df.loc[:, gender_col]
    df_choices['education'] = raw_df.loc[:, education_col]

    ### Add robots color and side by rationality
    df_choices['iSide'] = rat_comb['irrational']['side']
    df_choices['iColor'] = rat_comb['irrational']['color']
    df_choices['rSide'] = rat_comb['rational']['side']
    df_choices['rColor'] = rat_comb['rational']['color']

    ### trap answers
    df_choices['trap']         = raw_df[trap_question]

    ### remove to first rows which are Qualtrics data identifiers.
    df_choices = df_choices.iloc[2:,:].sort_values('id')

    ### remove users that havn't completed the questionnaire
    df_choices = df_choices.dropna()

    ### remove users that answered the trap question wrong
    df_choices = df_choices[df_choices['trap'].isin(['Cooking', 'Baking'])]

    ### ir/rational agree total
    # df_choices['irrational_agree'] = df_choices[list(agree_questions.append(dont_question))].replace({'rational': 0, 'irrational': 1, 'half': 1})\
    #                              .astype('int').sum(axis=1) / 7
    df_choices['irrational_agree'] = df_choices[questions['qns'] + [questions['dont']]]\
                                         .replace({'rational': 0, 'irrational': 1, 'half': 1}) \
                                         .astype('int').sum(axis=1) / 7
    df_choices['rational_agree'] = 1 - df_choices['irrational_agree']
    df_choices['agree2rational'] = (df_choices['rational_agree'] - df_choices['irrational_agree'])

    ###
    df_choices = df_choices.reset_index(drop=True)

    ### getting the columns names for all the questionnaires
    for col in raw_df.columns:
        cq = raw_df.loc[1, col].split(':')[1].split('_')[0][1:]
        cq_cols = raw_df.iloc[1, :][raw_df.iloc[1, :].str.contains(cq)]
        len_cq = cq_cols.__len__()
        if len_cq == 44:
            bfi_cols = np.array(cq_cols.index)
        if len_cq == 14:
            nars_cols = np.array(cq_cols.index)
        if len_cq == 24:
            if raw_df.loc[0, col].__contains__('red'):
                red_godspeed = np.array(cq_cols.index)
            else:
                blue_godspeed = np.array(cq_cols.index)

    ### list of users
    users = list(df_choices.id.unique())

    ### questionnaires data
    df_bfi      = BFI_data(raw_df, bfi_cols, users).drop('survey_code', axis = 1)
    df_nars     = NARS_data(raw_df, nars_cols, users).drop('survey_code', axis = 1)
    df_godspeed = Godspeed_data(raw_df, robots_comb, red_godspeed, blue_godspeed, users).drop('survey_code', axis = 1)

    ### add the questionnaires data into the dataframe
    df_choices = pd.concat((df_choices, df_bfi, df_nars, df_godspeed), axis=1)

    return df_choices


def questionnaire_df(raw_df, questionnaire_cols, options, users):
    df = raw_df.loc[2:, np.append(questionnaire_cols,'survey_code')]\
        .dropna()
    df = df[df.survey_code.isin(users)]\
        .sort_values('survey_code')\
        .replace(options)\
        .astype(int)
    return df

def trap_exclusion1(raw_df):
    '''
    Exclude users which answered the trap quesion wrong
    :param raw_df: raw data dataframe
    :return: raw_Df
    '''
    # trap question - exclude users
    a = raw_df[raw_df.columns[5:]][raw_df.question == 'trap_question']
    all_users = set(raw_df.columns[5:])
    trap_value = ['4','1']
    users_after_exclusion = set(a[(a == trap_value[0]) | (a == trap_value[1])].dropna(axis=1).columns)
    raw_df = raw_df.drop(all_users - users_after_exclusion, axis=1)
    return raw_df, users_after_exclusion

def BFI_data(raw_df, bfi_cols, users):

    df = questionnaire_df(raw_df, bfi_cols, bfi_options,users)
    df = df[df.survey_code.isin(users)]
    df = df.reset_index(drop=True)


    BFI_sub_meaning = {'S1':'Extroversion', 'S2':'Agreeableness', 'S3':'Conscientiousness', 'S4':'Neuroticism', 'S5':'Openness'}
    BFI_rev = {'S1':[0,1,0,0,1,0,1,0], 'S2':[1,0,1,0,0,1,0,1,0], 'S3':[0,1,0,1,1,0,0,0,1],'S4':[0,1,0,0,1,0,1,0],'S5':[0,0,0,0,0,0,1,0,1,0]}
    BFI_sub = {'S1':[1,6,11,16,21,26,31,36], 'S2':[2,7,12,17,22,27,32,37,42], 'S3':[3,8,13,18,23,28,33,38,43],'S4':[4,9,14,19,24,29,34,39],'S5':[5,10,15,20,25,30,35,40,41,44]}

    for s in BFI_sub.keys():

        ### columns to revverse the choices 1 <--> 5
        rev_cols = bfi_cols[np.array(BFI_sub[s]) - 1][np.array(BFI_rev[s], dtype='bool')]
        df[rev_cols] = df[rev_cols].applymap(reverse_choices)

        ### calculating the average for this sub scale
        df[BFI_sub_meaning[s]] = df[bfi_cols[np.array(BFI_sub[s]) - 1]].mean(axis = 1)


    return df[list(BFI_sub_meaning.values()) + ['survey_code']]


def NARS_data(raw_df, nars_cols, users):
    df = questionnaire_df(raw_df, nars_cols, nars_options, users)

    NARS_sub_meaning = {'S1': 'Interactions', 'S2':'Social_Influence', 'S3': 'Emotions'}
    NARS_sub = {'S1': [4,7,8,9,10,12], 'S2':[1,2,11,13,14], 'S3': [3,5,6]}

    for s in NARS_sub.keys():

        ### columns to reverse the choices 1 <--> 5
        if s == 'S3':
            df[NARS_sub[s]] = df[nars_cols[NARS_sub[s]]].applymap(reverse_choices)

        ### calculating the average for this sub scale
        df[NARS_sub_meaning[s]] = df[nars_cols[np.array(NARS_sub[s]) - 1]].sum(axis=1)

    df = df.reset_index(drop=True)

    return df[list(NARS_sub_meaning.values()) + ['survey_code']]


def Godspeed_data(raw_df, robots_comb, red_godspeed, blue_godspeed, users):
    df_dict = {}
    df_dict[robots_comb['Red']]  = questionnaire_df(raw_df, red_godspeed,  godspeed_options, users)
    df_dict[robots_comb['Blue']] = questionnaire_df(raw_df, blue_godspeed, godspeed_options, users)

    irr = list(set(df_dict.keys()) - set(['rational']))[0]

    # This way there is only rational/irrational rankings for godspeed
    df_dict['irrational'] = df_dict.pop(irr)
    df1 = pd.DataFrame()
    for rat, df in df_dict.items():
        goodspeed_cols = np.array(df.columns[:-1])
        Godspeed_sub_meaning = {'S1': 'Anthropomorphism', 'S2': 'Animacy', 'S3': 'Likability', 'S4': 'Intelligence','S5': 'Safety'}
        Godspeed_sub = {'S1': np.arange(1,6), 'S2': np.arange(6,12),'S3': np.arange(12,17), 'S4': np.arange(17,22),'S5': np.arange(22,25)}

        for s in Godspeed_sub.keys():
            ### calculating the average for this sub scale
            df1[rat[0] + Godspeed_sub_meaning[s]] = df[goodspeed_cols[np.array(Godspeed_sub[s]) - 1]].mean(axis=1)

    df1['survey_code'] = df['survey_code']
    for s, ms in Godspeed_sub_meaning.items():
        df1[ms+'2rational'] = df1['r'+ms] - df1['i'+ms]

    df1 = df1.reset_index(drop=True)

    return df1


def reverse_choices(v):
    '''
    reverse value for BFI analysis
    :param v: value to reverse
    :return: reversed value
    '''
    revrsed_values = [5.,4.,3.,2.,1.]
    return revrsed_values[int(v)-1]

def ttest_or_mannwhitney(y1,y2, paired = False ,return_dic = False):
    '''
    Check if y1 and y2 stand the assumptions for ttest and if not preform mannwhitney
    :param y1: 1st sample
    :param y2: 2nd sample
    :return: mean1, sem1, mean2, sem2m, s, pvalue, ttest - True/False
    '''
    ttest = False

    # assumptions for t-test
    # https://pythonfordatascience.org/independent-t-test-python/#t_test-assumptions
    ns1, np1 = stats.shapiro(y1)  # test normality of the data
    ns2, np2 = stats.shapiro(y2)  # test noramlity of the data
    ls, lp = stats.levene(y1, y2)  # test that the variance behave the same
    if (lp > .05) & (np1 > .05) & (np2 > .05):
        ttest = True
        if paired:
            typ = 'paired_ttest'
            s, p = stats.ttest_rel(y1, y2)
        else:
            typ = 'ttest'
            s, p = stats.ttest_ind(y1, y2)
    else:
        if paired:
            typ = 'wilcoxon'
            s, p = stats.wilcoxon(y1, y2)
        else:
            typ = 'mannwhitneyu'
            s, p = stats.mannwhitneyu(y1, y2)

    if ~return_dic:
        return y1.mean(), stats.sem(y1), y2.mean(), stats.sem(y2), s, p, ttest, typ
    else:
        return {
            'y1_isnormal': np1,
            'y2_isnormal': np2,
            'variance_test': np2,
            'y1_mean':y1.mean(),
            'y1_sem':stats.sem(y1),
            'y2_mean':y2.mean(),
            'y2_sem':stats.sem(y2),
            'test_type': typ,
            'test_score': s,
            'p_value': p
        }

def calculate_corr_with_pvalues(df, method = 'pearsonr', questionnaires = True, save_dir = None):
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

    if questionnaires:
        ### correlation tables seperated per questionnaire.
        corr_godspeed = rho.loc[cnames_groups['NARS']+cnames_groups['BFI'] + cnames_groups['Choices'],[x + '2rational' for x in cnames_groups['GODSPEED']]].copy()
        corr_bfi      = rho.loc[cnames_groups['NARS']+ cnames_groups['Choices'], cnames_groups['BFI']].copy()
        corr_nars     = rho.loc[cnames_groups['Choices'], cnames_groups['NARS']].copy()

        save_table(rho, save_dir, '00correlations', csv=True, Latex=True)
        save_table(corr_godspeed, save_dir, '00df_corr_godspeed')
        save_table(corr_bfi, save_dir, '00df_corr_bfi')
        save_table(corr_nars, save_dir, '00df_corr_nars')

    return pvalues, rho


def fallacy_rate(raw_df):
    '''
    return how many people choose each fallacy.
    :return: fallacy rate per type
    '''
    idx = {'Q5.1': 'conj', 'Q7.1': 'conj', 'Q10.1': 'disj', 'Q12.1': 'conj', 'Q14.1': 'disj', 'Q16.1': 'disj',
           'Q18.1': 'disj'}
    df = raw_df[questions['qns'] + [questions['dont']]]\
        .rename(columns = idx)
    fal_rate = df.apply(pd.value_counts)

    a = fal_rate['conj'].sum(axis=1).rename(index = {'half':'single_conj', 'irrational':'double_conj'})
    b = fal_rate['disj'].sum(axis=1).rename(index = {'half':'single_disj', 'irrational':'double_disj'})
    a = pd.DataFrame(a).T
    b = pd.DataFrame(b).T

    fal_rate = pd.concat([a, b], ignore_index=True).fillna(0).sum(axis=0)
    fal_rate = np.round(fal_rate / fal_rate.sum() * 100, 2)
    print(fal_rate)
    return fal_rate

def diff_test(raw_df, save_dir):

    for cat in cnames_groups['GODSPEED'] + [cnames_groups['Choices'][0]]:

        if cat == 'agree2rational':
            y1 = raw_df['agree2rational']
            y1m, y1s = y1.mean(), y1.std()
            y2m, y2s = 0, 0
            paired = True
            ttest = False
            test_typ = 'wilcoxon'
            s, p = stats.wilcoxon(y1)
        else:
            y1 = raw_df['r' + cat]
            y2 = raw_df['i' + cat]
            paired = False
            y1m, y1s, y2m, y2s, s, p, ttest, test_typ = ttest_or_mannwhitney(y1, y2, paired)

        temp_stats = pd.DataFrame.from_dict({'measurement': cat,
                                             'rational (mean+sem)':   '%.2f + %.2f' %(y1m, y1s),
                                             'irrational (mean+sem)': '%.2f + %.2f' %(y2m, y2s),
                                             'test_type': [test_typ], 'test_value': [s], 'p_value': [p]})
        if 'df_diff_test' not in locals():
            df_diff_test = temp_stats.copy()
        else:
            df_diff_test = pd.concat([df_diff_test, temp_stats], axis=0)
    save_table(df_diff_test, save_dir, '00diff', csv=True, Latex=True)
    return df_diff_test

def multi_linear_regression(raw_df, save_dir):
    formula0 ='Likability2rational ~ agree2rational'

    # likeability ~ agree + prefer
    formula =  '%s + prefer' % formula0
    mlr = smf.ols(formula, data=raw_df).fit()
    print(mlr.summary(), file=open(save_dir + "00likeability_agree_prefer.txt", "a"))

    # likability ~ agree + NARS
    formula = '%s' % formula0
    for s in cnames_groups['NARS']:
        formula += ' + ' + s
    mlr = smf.ols(formula, data=raw_df).fit()
    print(mlr.summary() , file=open(save_dir + "00likeability_agree_nars.txt", "a"))

    # likability ~ agree + NARS + BFI
    for s in cnames_groups['BFI']:
        formula += ' + ' + s
        mlr = smf.ols(formula, data=raw_df).fit()
    print(mlr.summary(), file=open(save_dir + "00likeability_agree_nars_BFI.txt", "a"))

    # likability ~ agree + NARS + BFI
    formula += ' + prefer'
    mlr = smf.ols(formula, data=raw_df).fit()
    print(mlr.summary(), file=open(save_dir + "00likeability_agree_prefer_nars_BFI.txt", "a"))

    # likability ~ agree + BFI
    formula = '%s' % formula0
    for s in cnames_groups['BFI']:
        formula += ' + ' + s
        mlr = smf.ols(formula, data=raw_df).fit()
    print(mlr.summary(), file=open(save_dir + "00likeability_agree_BFI.txt", "a"))

def new_df4goren(raw_df, save_dir):
    dft = raw_df[['prefer', 'rColor', 'iColor', 'rational_agree', 'irrational_agree']]

    dft0 = dft[dft.prefer == 0]
    dft0['rationality'] = '1'
    dft0['prefer'] = 0
    dft0['agree'] = dft0.pop('irrational_agree')
    dft0['color'] = dft0.iColor
    dft0 = dft0.drop(['rational_agree', 'rColor', 'iColor'], axis = 1)
    dft0 = dft0.reset_index(drop=True)

    dft1 = dft[dft.prefer == 0]
    dft1['rationality'] = '0'
    dft1['prefer'] = 1
    dft1['agree'] = dft1.pop('rational_agree')
    dft1['color'] = dft1.rColor
    dft1 = dft1.drop(['irrational_agree', 'rColor', 'iColor'], axis = 1)
    dft1 = dft1.reset_index(drop=True)

    dft2 = dft[dft.prefer == 1]
    dft2['rationality'] = '0'
    dft2['prefer'] = 0
    dft2['agree'] = dft2.pop('rational_agree')
    dft2['color'] = dft2.rColor
    dft2 = dft2.drop(['irrational_agree', 'rColor', 'iColor'], axis = 1)
    dft2 = dft2.reset_index(drop=True)

    dft3 = dft[dft.prefer == 1]
    dft3['rationality'] = '1'
    dft3['prefer'] = 1
    dft3['agree'] = dft3.pop('irrational_agree')
    dft3['color'] = dft3.iColor
    dft3 = dft3.drop(['rational_agree', 'rColor', 'iColor'], axis = 1)
    dft3 = dft3.reset_index(drop=True)

    dft = pd.concat((dft0, dft1, dft2, dft3), axis=0)

    save_table(dft, save_dir, '00df4mlr', Latex=False)

    return dft

def response_times_diff(save_dir):
    cols = ['Q' + str(x) for x in range(7)]
    rts_ranking = pd.read_csv(save_dir + '00rts_ranking.csv', index_col=0).reset_index(drop = True)
    rts_ranking.columns = cols
    rts_choosing = pd.read_csv(save_dir + '00rts_choosing.csv', index_col=0).reset_index(drop = True)
    rts_choosing.columns = cols

    rts_diff = pd.DataFrame({'meanRanking':[], 'semRanking':[], 'meanChoosing':[], 'semChoosing':[], 't':[], 'p_value':[], 'test_type':[]})
    for i in rts_ranking:
        y1 = rts_ranking[i]
        y2 = rts_choosing[i]
        y1m, y1s, y2m, y2s, s, p, _, typ = ttest_or_mannwhitney(y1, y2)
        temp = pd.DataFrame({'meanRanking': [y1m], 'semRanking': [y1s], 'meanChoosing': [y2m], 'semChoosing': [y2s], 't': [s], 'p_value': [p],'test_type': [typ]})
        rts_diff = rts_diff.append(temp)
    rts_diff['question'] = cols
    rts_diff = rts_diff.reset_index(drop=True)

    rts_diff.to_csv(save_dir + '00rt.csv')
    return rts_diff


def save_table(df, df_dir,file_name, csv = True, Latex = True):
    if csv:
        df.to_csv(df_dir + file_name + '.csv') # drop = True

    if Latex:
        df = df.round(3)
        with open(df_dir + file_name+ '.tex', 'w') as tf:
            tf.write(df.to_latex())
def main():
    # process_data, analysis = True, False
    process_data, analysis = False, True
    df_dir = 'data/dataframes/'
    save_dir = 'data/paper/'

    if process_data:
        raw_df = prepare_data(df_dir)
    else:
        raw_df = pd.read_csv(df_dir + '/new_raw_df.csv', index_col=0)

    if analysis:
        pv, rho = calculate_corr_with_pvalues(raw_df, method='pearsonr' , save_dir=save_dir)
        df_diff = diff_test(raw_df, save_dir)
        save_table(raw_df.describe(), save_dir, '00descriptive')

        # multi_linear_regression(raw_df, save_dir)

        df4goren = new_df4goren(raw_df, save_dir)

    response_times_diff(save_dir)

    print()

if __name__ == "__main__":
    main()