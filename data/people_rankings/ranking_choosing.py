import pandas as pd
import numpy as np

import os,sys,inspect
from new_reformat import ttest_or_mannwhitney

# questions organizer
questions = {'BFI': 7,
             'BFI_sub': ['Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
             'Gender': 'Q3',
             'Age' : 'Q4',
             'Education' : 'Q5',
             'trap': 13,
             'dont': 20,
             'fallacies':{'conj': [9, 11, 16],
                          'disj': [14, 18, 20, 22]}}

qns_conj = questions['fallacies']['conj']
qns_conj = list(np.array(qns_conj, dtype='str'))
qns_conj = ['Q' + qn for qn in qns_conj] # list of choosing questions

qns_disj = questions['fallacies']['disj']
qns_disj = list(np.array(qns_disj, dtype='str'))
qns_disj = ['Q' + qn for qn in qns_disj] # list of choosing questions

qns = qns_conj + qns_disj

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


def load_data(df2load, qns_conj, qns_disj, rat):
    '''
    load data.
    remove users hat fell in trap question.
    right choices rational/ irrational instead of 1,2.
    :param df2load: what file to load.
    :param questions: questions organizer.
    :param rat: what rationalities to replace with 1,2.
    :return: loaded dataframe.
    '''
    raw_df = pd.read_csv(df2load)
    print(raw_df.shape[0])
    rows2remove, users2remove = trap_question(raw_df, 13, 3)
    raw_df = raw_df.drop(rows2remove, axis=0)

    raw_df = dont_reverse(raw_df, 'Q'+ str(questions['trap'])) # reverse dont question
    
    raw_df[qns_conj] = raw_df[qns_conj].replace(rat_fal(rat, 'conj')) # replace 1, 2 to rationalities
    raw_df[qns_disj] = raw_df[qns_disj].replace(rat_fal(rat, 'disj')) # replace 1, 2 to rationalities

    raw_df = raw_df.reset_index(drop=True)

    return raw_df

def rat_fal(rat, fal):
    temp = rat.copy()
    for k, v in temp.items():
        if v != 'r':
            temp[k] = fal[0] + v
    return temp

def dont_reverse(df, col):
    df[col] = df[col].replace('1','3')
    df[col] = df[col].replace('2','1')
    df[col] = df[col].replace('3','2')
    return df

def trap_question(raw_df, qn, option):
    qn = 'Q'+str(qn)
    users_idx = raw_df[qn] != str(option)
    users2remove = raw_df.loc[users_idx, 'survey_code']

    print('%d participants failed the trap question'%(len(users2remove) - 2))
    return raw_df.index[users_idx], users2remove

def fallacy_rate(raw_df, qns, qns_conj, qns_disj, save_dir = 'data/paper/', fal_type = 'single'):
    fal_rate = raw_df[qns].apply(pd.value_counts).sum(axis=1)
    fal_rate = np.round(fal_rate/fal_rate.sum() * 100,2)
    fal_rate = fal_rate.rename(index = {'r': 'rational',
                                        'ch': 'single_conj',
                                        'ci': 'double_conj',
                                        'dh': 'single_disj',
                                        'di': 'double_disj'})

    b = pd.DataFrame()
    for q in qns:
        temp = raw_df[q].value_counts() / raw_df.shape[0]
        b = pd.concat((b, temp), axis=1)
    b.columns = qns

    conj_rate = b[qns_conj].dropna()
    conj_rate.to_csv('01rankings_choosing_conj' + fal_type + '.csv')
    conj_rate = conj_rate.mean(axis = 1)
    disj_rate = b[qns_disj].dropna()
    disj_rate.to_csv('01rankings_choosing_disj' + fal_type + '.csv')
    disj_rate = disj_rate.mean(axis = 1)

    return conj_rate, disj_rate


def compare_fallacy_rates(fal_type, rate_comparison_df):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir) + '\\paper\\'

    # load the data from the ranking choosing
    df1 = pd.read_csv('01rankings_choosing_' + fal_type + '.csv', index_col=0)
    # load the data from the robots choosing
    df2 = pd.read_csv(parentdir + '01robots_choosing_' + fal_type + '.csv', index_col=0)

    # rename the columns and indices to be the same
    df2.columns = df1.columns
    df2.index = df1.index

    # compare them for rational / irrational
    for ind, row in df1.iterrows():
        d = ttest_or_mannwhitney(row, df2.loc[ind,:], return_dic=True)
        d['fal_typ'] = [fal_type]
        d['sd'] = [ind] # single or double
        rate_comparison_df = rate_comparison_df.append(pd.DataFrame.from_dict(d))
    return rate_comparison_df


def response_times(raw_df):
    rt_questions = np.array(questions['fallacies']['conj'] + questions['fallacies']['disj']) + 1
    rt_questions = ['Q' + str(x) + '_Page Submit' for x in rt_questions]
    responses_times = raw_df[rt_questions].astype('float')
    return responses_times

def main():
    # load the data.
    for comb in ['lirr', 'lhrr', 'lrri', 'lrrh']:
        # 1 : left, 2: right
        rat = {'1': comb[1], '2': comb[-1]}
        sides = {'1': comb[0], '2': comb[-2]}
        df2load = 'Emma_ranking_' + comb + '.csv'
        try:
            df = load_data(df2load, qns_conj, qns_disj, rat)
            r_side = sides[list(rat.keys())[list(rat.values()).index('r')]]
            df['rational_position'] = r_side
            if r_side == 'r':
                df['irrational_position'] = 'l'
            else:
                df['irrational_position'] = 'r'
            a, b = comb[1], comb[-1]

            if (a == 'r' and b == 'i') or (a == 'i' and b == 'r'):
                if 'raw_df_ri' in locals():
                    raw_df_ri = raw_df_ri.append(df)
                    raw_df_ri = raw_df_ri.reset_index(drop=True)
                else:
                    raw_df_ri = df.copy()

            if (a == 'r' and b == 'h') or (a == 'h' and b == 'r'):
                if 'raw_df_rh' in locals():
                    raw_df_rh = raw_df_rh.append(df)
                    raw_df_rh = raw_df_rh.reset_index(drop=True)
                else:
                    raw_df_rh = df.copy()
            # if 'raw_df_all' in locals():
            #     raw_df_all = raw_df_all.append(raw_df)
            #     raw_df_all = raw_df_all.reset_index(drop=True)
            # else:
            #     raw_df_all = raw_df.copy()
        except:
            pass

    raw_df_all = raw_df_ri.append(raw_df_rh)
    for idx in ['Gender', 'Age', 'Education']:
        raw_df_all = raw_df_all.rename({questions[idx]: idx}, axis=1)
    raw_df_all.to_csv('raw_df_choose_ranking.csv')

    ### BFI answers
    bfi_cols = raw_df_all.columns[raw_df_all.columns.str.contains(str(questions['BFI']) + '_')]
    df_bfi = BFI_data(raw_df_all, bfi_cols)

    raw_df_all = pd.concat((raw_df_all, df_bfi), axis=1)

    # DataFrame for Goren
    rdfr = raw_df_all.copy()
    rdfr.pop('irrational_position')
    rdfr = rdfr.rename({'rational_position': 'side'}, axis='columns')
    rdfr['rationality'] = 'rational'
    rdfr[qns][rdfr[qns] != 'r'] = 0
    rdfr[qns][rdfr[qns] == 'r'] = 1
    rdfr = rdfr[qns + ['side', 'rationality', 'Gender', 'Age', 'Education'] + questions['BFI_sub']]

    rdfi = raw_df_all.copy()
    rdfi.pop('rational_position')
    rdfi = rdfi.rename({'irrational_position': 'side'}, axis='columns')
    rdfi['rationality'] = 'irrational'
    rdfi[qns][rdfi[qns] != 'r'] = 1
    rdfi[qns][rdfi[qns] == 'r'] = 0
    rdfi = rdfi[qns + ['side', 'rationality', 'Gender', 'Age', 'Education'] + questions['BFI_sub']]

    df4goren = rdfr.append(rdfi)
    df4goren['side'] = df4goren['side'].replace({'l':'left', 'r':'right'})
    df4goren.to_csv('df4goren.csv')

    conj_rate, disj_rate = fallacy_rate(raw_df_rh, qns, qns_conj, qns_disj)
    print('rational - single fallacy:')
    print(conj_rate)
    print(disj_rate)

    rate_comparison_df = pd.DataFrame()
    rate_comparison_df = compare_fallacy_rates('conjsingle', rate_comparison_df)
    rate_comparison_df = compare_fallacy_rates('disjsingle', rate_comparison_df)

    conj_rate, disj_rate = fallacy_rate(raw_df_ri, qns, qns_conj, qns_disj, fal_type='double')
    print('rational - double fallacy:')
    print(conj_rate)
    print(disj_rate)

    rate_comparison_df = compare_fallacy_rates('conjdouble', rate_comparison_df)
    rate_comparison_df = compare_fallacy_rates('disjdouble', rate_comparison_df)

    rate_comparison_df = rate_comparison_df.reset_index(drop=True)
    rate_comparison_df.to_csv('01ranking_choosing_comparison')
    print('-------- rates comparison ---------')
    print(rate_comparison_df)

    rts = response_times(raw_df_all)
    print('-------- response times ---------')
    print(rts.mean(axis=0))

    rts.to_csv('00rts_choosing.csv')


if __name__ == "__main__":
    main()