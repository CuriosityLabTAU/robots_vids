import numpy as np
import pandas as pd

raw_df = pd.read_csv('Emma_ranking_options.csv')

questions = {'conj': [6, 8, 12],
             'disj': [10, 14, 16, 18]}


def fallacy_rate(raw_df, fallacy, cq):
    '''
    :param raw_df: dataframe with raw data
    :param fallacy: conj/ disj
    :param cq: question number 'Q#_'
    '''

    temp_df = raw_df[raw_df.columns[raw_df.columns.str.contains(cq)]].copy()
    col = temp_df.columns[temp_df.columns.str.contains(cq + '3')][0]
    temp_df[cq+'rate'] = 0
    if fallacy == 'conj':
        temp_df[cq + 'rate'][temp_df[col] == '2'] = 1
        temp_df[cq + 'rate'][temp_df[col] == '1'] = 2
    elif fallacy == 'disj':
        temp_df[cq + 'rate'][temp_df[col] == '2'] = 3
        temp_df[cq + 'rate'][temp_df[col] == '3'] = 4

    return temp_df[cq + 'rate']

for fal, qs in questions.items():
    for qn in qs:
        cq = 'Q' + str(qn) + '_'
        raw_df[cq + 'rate'] = fallacy_rate(raw_df, fal, cq)

rate_df = raw_df[raw_df.columns[raw_df.columns.str.contains('rate')]]
rate_df = rate_df.drop([0,1], axis = 0)
fal_rate = rate_df.stack().value_counts()
fal_rate = 100 * fal_rate / fal_rate.sum()

print('''
%.2f %% of the people were rational
%.2f %% of the people were irrational
single conjunction rate is %.2f %%
double conjunction rate is %.2f %%
single disjunction rate is %.2f %%
double disjunction rate is %.2f %%
''' % (fal_rate[0], fal_rate[1] + fal_rate[2] + fal_rate[3] + fal_rate[4],fal_rate[1], fal_rate[2],fal_rate[3], fal_rate[4]))
print()