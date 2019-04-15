import numpy as np
import pandas as pd

def BFI_data(df, bfi_cols):

    BFI_sub_meaning = {'S1':'Extroversion', 'S2':'Agreeableness', 'S3':'Conscientiousness', 'S4':'Neuroticism', 'S5':'Openness'}
    BFI_rev = {'S1':[0,1,0,0,1,0,1,0], 'S2':[1,0,1,0,0,1,0,1,0], 'S3':[0,1,0,1,1,0,0,0,1],'S4':[0,1,0,0,1,0,1,0],'S5':[0,0,0,0,0,0,1,0,1,0]}
    BFI_sub = {'S1':[1,6,11,16,21,26,31,36], 'S2':[2,7,12,17,22,27,32,37,42], 'S3':[3,8,13,18,23,28,33,38,43],'S4':[4,9,14,19,24,29,34,39],'S5':[5,10,15,20,25,30,35,40,41,44]}

    for s in BFI_sub.keys():

        ### columns to revverse the choices 1 <--> 5
        rev_cols = bfi_cols[np.array(BFI_sub[s]) - 1][np.array(BFI_rev[s], dtype='bool')]
        df[rev_cols] = df[rev_cols].applymap(reverse_choices)

        ### calculating the average for this sub scale
        df[BFI_sub_meaning[s]] = df[bfi_cols[np.array(BFI_sub[s]) - 1]].mean(axis = 1)


    return df[list(BFI_sub_meaning.values())]

def reverse_choices(v):
    '''
    reverse value for BFI analysis
    :param v: value to reverse
    :return: reversed value
    '''
    revrsed_values = [5.,4.,3.,2.,1.]
    return revrsed_values[int(v)-1]

def fallacy_rate(raw_df, fallacy, cq):
    '''
    :param raw_df: dataframe with raw data
    :param fallacy: conj/ disj
    :param cq: question number 'Q#_'
    '''

    col = raw_df.columns[raw_df.columns.str.contains(cq + '3')][0]
    temp_df = pd.DataFrame(raw_df[col])
    temp_df[cq+'rate'] = 0
    if fallacy == 'conj':
        temp_df.loc[temp_df[col] == '2', cq + 'rate'] = 1
        temp_df.loc[temp_df[col] == '1', cq + 'rate'] = 2
    elif fallacy == 'disj':
        temp_df.loc[temp_df[col] == '2', cq + 'rate'] = 3
        temp_df.loc[temp_df[col] == '3', cq + 'rate'] = 4

    return temp_df[cq + 'rate']

def total_fallacy_rate(raw_df, questions, save_dir = 'data/paper/'):
    for fal, qs in questions.items():
        for qn in qs:
            cq = 'Q' + str(qn) + '_'
            print(cq)
            raw_df[cq + 'rate'] = fallacy_rate(raw_df, fal, cq)

    rate_df = raw_df[raw_df.columns[raw_df.columns.str.contains('rate')]]
    rate_df.to_csv('raw_df_ranking.csv')

    qns_conj = ['Q' + str(x) + '_rate' for x in questions['conj']]
    qns_disj = ['Q' + str(x) + '_rate' for x in questions['disj']]

    b = pd.DataFrame()
    for q in qns_conj + qns_disj:
        temp = rate_df[q].value_counts() / raw_df.shape[0]
        b = pd.concat((b, temp), axis=1)
    b.columns = qns_conj + qns_disj
    conj_rate = b[qns_conj].dropna()
    conj_rate.to_csv('01free_rankings_conj.csv')
    conj_rate = conj_rate.mean(axis = 1)
    disj_rate = b[qns_disj].dropna()
    disj_rate.to_csv('01free_rankings_disj.csv')
    disj_rate = disj_rate.mean(axis = 1)

    print(conj_rate)
    print(disj_rate)

    fal_rate = rate_df.stack().value_counts()
    fal_rate = 100 * fal_rate / fal_rate.sum()

    print(fal_rate)
    return fal_rate, conj_rate, disj_rate

def trap_question(raw_df, qn, option):
    qn = 'Q'+str(qn)
    users_idx = raw_df[qn] != str(option)
    users2remove = raw_df.loc[users_idx, 'survey_code']

    print('%d participants failed the trap question'%(len(users2remove)))
    return raw_df.index[users_idx], users2remove

def response_times(raw_df):
    rt_questions = np.array(questions['conj'] + questions['disj']) + 1
    rt_questions = ['Q' + str(x) + '_Page Submit' for x in rt_questions]
    responses_times = raw_df[rt_questions].astype('float')
    return responses_times


# old quesionnaire guide
# questions = {'conj': [6, 8, 12],
#              'disj': [10, 14, 16, 18],
#              'trap': 21,
#              'fpath': 'Emma_ranking_options.csv'}

### new questionnaire
questions = {'conj': [9, 11, 18, 22],
             'disj': [14, 16, 20],
             'trap': 13,
             'fpath': 'Emma_ranking_options_v4.csv',
             'BFI': 7,
             'BFI_sub': ['Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
             'Gender': 'Q3',
             'Age' : 'Q4',
             'Education' : 'Q5',}
# load the data.
raw_df = pd.read_csv(questions['fpath'])

#check trap question
rows2remove, users2remove = trap_question(raw_df, questions['trap'], 3)
raw_df = raw_df.drop(rows2remove, axis=0)

### demographics
for idx in ['Gender', 'Age', 'Education']:
    raw_df = raw_df.rename({questions[idx]:idx}, axis = 1)

print(raw_df.shape[0])

### BFI answers
bfi_cols = raw_df.columns[raw_df.columns.str.contains(str(questions['BFI']) + '_')]
df_bfi = BFI_data(raw_df, bfi_cols)

raw_df = pd.concat((raw_df, df_bfi), axis=1)

raw_df.to_csv('00ranking_options.csv')

cs = raw_df.columns ### df columns
cnames = ['Q' + str(x) +'_' for x in questions['conj'] + questions['disj']] ### the columns I'm intersted in.
cnames = [i for e in cnames for i in cs if e in i]
cnames = [x for x in cnames if 'rate' not in x]

### choose onlty the columns of answers for the questions and drop unfinished answers.
df4goren = raw_df[cnames].dropna()
df4goren.to_csv('00ranking_options4goren.csv')

### calculate fallacy rates
fal_rate, conj_rate, disj_rate = total_fallacy_rate(raw_df, dict(filter(lambda i:i[0] in ['conj', 'disj'], questions.items())))
#
rts = response_times(raw_df)
print(rts.mean(axis = 0))
rts.to_csv('00rts_ranking.csv')

print()