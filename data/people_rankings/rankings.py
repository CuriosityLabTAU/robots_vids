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

def total_fallacy_rate(raw_df, questions):
    for fal, qs in questions.items():
        for qn in qs:
            cq = 'Q' + str(qn) + '_'
            raw_df[cq + 'rate'] = fallacy_rate(raw_df, fal, cq)

    rate_df = raw_df[raw_df.columns[raw_df.columns.str.contains('rate')]]
    # rate_df = rate_df.drop([0, 1], axis=0)
    fal_rate = rate_df.stack().value_counts()
    fal_rate = 100 * fal_rate / fal_rate.sum()

    print('''
    %.2f %% of the people were rational
    %.2f %% of the people were irrational
    single conjunction rate is %.2f %%
    double conjunction rate is %.2f %%
    single disjunction rate is %.2f %%
    double disjunction rate is %.2f %%
    ''' % (fal_rate[0], fal_rate[1] + fal_rate[2] + fal_rate[3] + fal_rate[4], fal_rate[1], fal_rate[2], fal_rate[3],
           fal_rate[4]))
    return fal_rate

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


# questions fallacies
questions = {'conj': [6, 8, 12],
             'disj': [10, 14, 16, 18]}

# load the data.
raw_df = pd.read_csv('Emma_ranking_options.csv')

#check trap question
rows2remove, users2remove = trap_question(raw_df, 21, 3)
raw_df = raw_df.drop(rows2remove, axis=0)

print(raw_df.shape[0])
fal_rate = total_fallacy_rate(raw_df, questions)
rts = response_times(raw_df)
print(rts.mean(axis = 0))

rts.to_csv('00rts_ranking.csv')

print()