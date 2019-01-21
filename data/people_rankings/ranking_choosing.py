import pandas as pd
import numpy as np

# questions organizer
questions = {'BFI': 7,
             'Gender': 3,
             'Age' : 4,
             'Education' : 5,
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

def fallacy_rate(raw_df, qns):
    nu = raw_df.shape[1]
    fal_rate = raw_df[qns].apply(pd.value_counts).sum(axis=1)
    fal_rate = np.round(fal_rate/fal_rate.sum() * 100,2)
    fal_rate = fal_rate.rename(index = {'r': 'rational',
                                        'ch': 'single_conj',
                                        'ci': 'double_conj',
                                        'dh': 'single_disj',
                                        'di': 'double_disj'})
    return fal_rate
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
        df2load = 'Emma_ranking_' + comb + '.csv'
        try:
            raw_df = load_data(df2load, qns_conj, qns_disj, rat)
            if 'raw_df_all' in locals():
                raw_df_all = raw_df_all.append(raw_df)
                raw_df_all = raw_df_all.reset_index(drop=True)
            else:
                raw_df_all = raw_df.copy()
        except:
            pass

    print(raw_df_all.shape[0])
    fal_rate = fallacy_rate(raw_df_all, qns)
    print(fal_rate)
    rts = response_times(raw_df)
    print(rts.mean(axis=0))

    rts.to_csv('00rts_choosing.csv')


if __name__ == "__main__":
    main()