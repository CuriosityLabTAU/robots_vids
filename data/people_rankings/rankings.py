import numpy as np
import pandas as pd

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
questions = {'conj': [6, 8, 12],
             'disj': [10, 14, 16, 18],
             'trap': 21,
             'fpath': 'Emma_ranking_options.csv'}

### new questionnaire
questions = {'conj': [9, 11, 18, 22],
             'disj': [14, 16, 20],
             'trap': 13,
             'fpath': 'Emma_ranking_options_v1.csv',
             'BFI': 7,
             'Gender': 'Q3',
             'Age' : 'Q4',
             'Education' : 'Q5',}
# load the data.
raw_df = pd.read_csv(questions['fpath'])

#check trap question
rows2remove, users2remove = trap_question(raw_df, questions['trap'], 3)
raw_df = raw_df.drop(rows2remove, axis=0)

for idx in ['Gender', 'Age', 'Education']:
    raw_df = raw_df.rename({questions[idx]:idx}, axis = 1)

print(raw_df.shape[0])
fal_rate, conj_rate, disj_rate = total_fallacy_rate(raw_df, dict(filter(lambda i:i[0] in ['conj', 'disj'], questions.items())))
#
rts = response_times(raw_df)
print(rts.mean(axis = 0))
rts.to_csv('00rts_ranking.csv')

print()