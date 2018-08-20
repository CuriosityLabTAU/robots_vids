import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from scipy import stats


def raw_data_extraction(path):
    '''
    Extracting raw data
    :param path: the path of the file
    :return: raw_df: dataframe containing the raw data
    '''
    raw_df = pd.read_csv(path)
    raw_df = raw_df.transpose()
    raw_df = raw_df.rename(index = str, columns={0: 'full_text', 1: 'dict_text'})
    raw_df.insert(0, 'option', '')
    raw_df.insert(0, 'question', '')

    # get the robots deployment in specific questionnaire
    rDeployment = path.split('_')[3]
    d = {'colors':{'b':'blue','r':'red'}, 'side':{'l':'left','r':'right'}, 'rationality':{'r':'rational','h':'half','i':'irrational'}}
    robots_deployment = {d['colors'][rDeployment[1]]: [d['side'][rDeployment[0]], d['rationality'][rDeployment[2]]],
                         d['colors'][rDeployment[4]]: [d['side'][rDeployment[3]], d['rationality'][rDeployment[5]]]}

    for i in raw_df.index:
        a = raw_df.loc[i, 'dict_text'].split(':')
        try:
            raw_df.loc[i, 'question'] = a[1].split('_')[0].replace('"', '')  # question
            try:
                raw_df.loc[i, 'option'] = int(a[1].split('_')[1].replace('}', '').replace('"', ''))  # question options
            except:
                raw_df.loc[i, 'option'] = a[1].split('_')[1]
            #     if equal page - timing
        except:
            b = a[1].split('_')[0].replace('"', '')  # question
            raw_df.loc[i, 'question'] = b.replace('}', '')

    # cleaning unnecessary rows.
    raw_df = raw_df.drop(raw_df.option[raw_df.option == 'FIRST'].index)
    raw_df = raw_df.drop(raw_df.option[raw_df.option == 'LAST'].index)
    raw_df = raw_df.drop(raw_df.option[raw_df.option == 'CLICK'].index)

    # find the questions response time
    raw_df.loc[raw_df[raw_df.option == 'PAGE'].index, 'option'] = 'response_time' # renaming response time
    raw_df.loc[raw_df[raw_df.full_text == 'Response ID'].index, 'question'] = 'ID' # renaming response time

    #  change the question name, so that the question name of the response time is the same of the question it's timing.
    questions = raw_df.question.unique()
    for q in raw_df[raw_df.option == 'response_time'].question:
        raw_df.question[raw_df.question == q] = questions[np.where(questions == q)[0] - 1][0]

    # Find the rows of the different questionnaires.
    i = 1
    for q in questions:
        if raw_df[raw_df.question == q].__len__() == 45:
            raw_df.question[raw_df.question == q] = 'BFI'
        if raw_df[raw_df.question == q].__len__() == 15:
            raw_df.question[raw_df.question == q] = 'NARS'
        if (raw_df[raw_df.question == q].__len__() == 24) | (raw_df[raw_df.question == q].__len__() == 25):
            raw_df.question[raw_df.question == q] = 'GODSPEED' + str(i)
            i+=1

    # Find the rows of the demographic questions.
    raw_df.loc[raw_df.full_text == 'What is your age?', 'question'] = 'age'
    raw_df.loc[raw_df.full_text == 'To which gender identity do you most identify?', 'question'] = 'gender'
    # raw_df.loc[raw_df.full_text == 'What is the highest degree or level of school you have completed? (If you are currently enrolled in school, please indicate the highest degree you have received.)', 'question'] = 'education'
    raw_df[raw_df.full_text.str.contains('What is your age?')].index.tolist()[0]
    raw_df.loc[raw_df[raw_df.full_text.str.contains('school')].index.tolist()[0], 'question'] = 'education'

    raw_df.loc[raw_df.full_text == 'What does Liz enjoy doing?', 'question'] = 'trap_question'

    nu = raw_df.columns[4:].__len__() # number of participants
    raw_df = raw_df.append(pd.DataFrame(data=[['red_robot', 'deployment', 'side', ''] + [robots_deployment['red'][0]] * nu], columns = raw_df.columns))
    raw_df = raw_df.append(pd.DataFrame(data=[['red_robot', 'deployment', 'rationality', ''] + [robots_deployment['red'][1]] * nu], columns = raw_df.columns))
    raw_df = raw_df.append(pd.DataFrame(data=[['blue_robot', 'deployment', 'side', ''] + [robots_deployment['blue'][0]] * nu], columns = raw_df.columns))
    raw_df = raw_df.append(pd.DataFrame(data=[['blue_robot', 'deployment', 'rationality', ''] + [robots_deployment['blue'][1]] * nu], columns = raw_df.columns))

    # cleaning users that didn't answer all the questions
    a = raw_df.loc[raw_df.question == 'locationLatitude', :]
    df_temp = a[a.columns[5:]]
    df_temp.columns = df_temp.columns.astype('int')
    filter_nan_users = pd.isna(df_temp).any(axis=0).tolist()
    if filter_nan_users != False:
        unanswered_users = df_temp.T[filter_nan_users].index.tolist()
        raw_df = raw_df.drop(unanswered_users, axis=1)
        print('unanswered users ', unanswered_users.__len__())

    # cleaning users that didn't complete to fill the questionnaire.
    a = raw_df[raw_df['full_text'].str.contains('talkative?')]
    b = pd.isnull(a).any()
    empty_users = b[b].index.tolist()
    print('empty users',empty_users.__len__())
    raw_df = raw_df.drop(empty_users, axis=1)

    raw_df.to_csv('data/dataframes/raw_dataframe_'+rDeployment)     # saving the data frame
    return raw_df, rDeployment

def trap_exclusion(raw_df):
    '''
    Exclude users which answered the trap quesion wrong
    :param raw_df: raw data dataframe
    :return: raw_Df
    '''
    # trap question - exclude users
    a = raw_df[raw_df.columns[:-5]][raw_df.question == 'trap_question']
    all_users = set(raw_df.columns[:-5])
    trap_value = '4'
    users_after_exclusion = set(a[a == trap_value].dropna(axis=1).columns)
    raw_df = raw_df.drop(all_users - users_after_exclusion, axis=1)
    return raw_df, users_after_exclusion

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

def response_time_exclusion(raw_df, users_after_exclusion):
    '''
    Removing users which had total response time bigger (smaller) than mean+-3std
    :param raw_df:
    :return:
    '''
    rt = {}
    rt['total'] = raw_df.loc[raw_df[raw_df.option == 'response_time'][raw_df.columns[4:].tolist()].index, users_after_exclusion].astype('float').sum()
    rt['std'] = rt['total'].std()
    rt['mean'] = rt['total'].mean()

    users_to_exclude = rt['total'][rt['total'] > rt['mean'] + 3 * rt['std']].index
    raw_df = raw_df.drop(users_to_exclude, axis = 1)

    return raw_df, users_to_exclude

def create_stats_df(raw_df, fn):
    '''
    Creating statistical dataframe for inferential analysis.
    :param raw_df: dataframe containing the raw data
    :param fn:  string of the setup for file name
    :return: stats_df: dataframe with for inferential analysis.
    '''
    # raw_df = raw_df.replace(np.nan, 1000)
    usersID = raw_df[raw_df.question == 'ID'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1)
    usersAGE = raw_df[raw_df.question == 'age'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).loc[raw_df[raw_df.full_text.str.contains('What is your age?')].index.tolist()[0]].tolist()
    usersGENDER = raw_df[raw_df.question == 'gender'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).loc[raw_df[raw_df.full_text.str.contains('gender')].index.tolist()[0]].astype(float).tolist()
    usersEDUCATION = raw_df[raw_df.question == 'education'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).loc[raw_df[raw_df.full_text.str.contains('school')].index.tolist()[0]].astype(float).tolist()
    cnames = ['robot','feature', 'sub_scale', 'meaning'] + usersID.transpose()['ResponseId'].tolist()

    stats_df = pd.DataFrame(columns = cnames) # Inferential dataframe

    stats_df = stats_df.append(pd.DataFrame(data = [['','gender','','']+usersGENDER], columns=cnames))
    stats_df = stats_df.append(pd.DataFrame(data = [['','age','','']+usersAGE], columns=cnames))
    stats_df = stats_df.append(pd.DataFrame(data = [['','education','','']+usersEDUCATION], columns=cnames))

    stats_df = NARS_data(stats_df, raw_df)
    stats_df = BFI_data(stats_df, raw_df)
    stats_df = GODSPEED_data(stats_df, raw_df, 'red')
    stats_df = GODSPEED_data(stats_df, raw_df, 'blue')

    stats_df.robot[stats_df.robot == ''] = 'participant'
    stats_df = preference_data(stats_df, raw_df)
    if not(stats_df.empty):
        stats_df = questions(stats_df, raw_df)

        # Insert what was the robot deployment
        a = raw_df[(raw_df.question == 'red_robot') | (raw_df.question == 'blue_robot')]
        a.columns = stats_df.columns
        stats_df = stats_df.append(a)

        # preference summary
        t = stats_df[stats_df.sub_scale == 'summary']
        stats_df = stats_df.append(pd.DataFrame(data=[['red', 'preference', 'average', '']+t[t.robot == 'red'][t.columns[4:]].mean().tolist()], columns = stats_df.columns))
        stats_df = stats_df.append(pd.DataFrame(data=[['red', 'preference', 'std', '']+t[t.robot == 'red'][t.columns[4:]].std().tolist()], columns = stats_df.columns))
        stats_df = stats_df.append(pd.DataFrame(data=[['blue', 'preference', 'average', '']+t[t.robot == 'blue'][t.columns[4:]].mean().tolist()], columns = stats_df.columns))
        stats_df = stats_df.append(pd.DataFrame(data=[['blue', 'preference', 'std', '']+t[t.robot == 'blue'][t.columns[4:]].std().tolist()], columns = stats_df.columns))

        stats_df = stats_df.reset_index(drop=True)

        stats_df = stats_df_reformat(stats_df)

        stats_df.loc[:, 'gender'] = stats_df.loc[:, 'gender'].replace({1.0: 'female', 2.0: 'male'})
        stats_df.loc[:, 'education'] = stats_df.loc[:, 'education'].replace(
            {1.0: '<HS', 2.0: 'HS', 3.0: '<BA', 4.0: 'Associate degree', 5.0: 'BA', 6.0: 'MA', 7.0: 'professional', 8.0: 'PhD'})


        stats_df = stats_df.reset_index(drop=True)
        stats_df.answers = stats_df.answers.fillna(0)

        stats_df.to_csv('data/dataframes/stats_dataframe'+ fn)     # saving the data frame

        return stats_df

def long_answers(stats_df):
    '''
    Adding were was the answer that the user cose
    :param stats_df: statistical datframe,
    :return:
    '''

    # todo: long answers
    stats_df['long_answer'] = ''
    stats_df[stats_df.feature == 'q_pref'].answers = stats_df[stats_df.feature == 'q_pref'].rationality
    # stats_df = stats_df.drop(users2exclude, axis=1)
    # df = df.drop(df.loc['ResponseId', df.loc['ResponseId'].isin(users2exclude)].index, axis=1)

    stats_df[stats_df.sub_scale == 'Q16.1'].rationality = stats_df[stats_df.sub_scale == 'Q16.1'].rationality.replace('irrational', 'two')
    stats_df[stats_df.sub_scale == 'Q16.1'].rationality = stats_df[stats_df.sub_scale == 'Q16.1'].rationality.replace('rational', 'irrational')
    stats_df[stats_df.sub_scale == 'Q16.1'].rationality = stats_df[stats_df.sub_scale == 'Q16.1'].rationality.replace('two', 'rational')


    conj = stats_df[(stats_df.sub_scale == 'Q5.1') | (stats_df.sub_scale == 'Q7.1') | (stats_df.sub_scale == 'Q10.1') | (stats_df.sub_scale == 'Q16.1')]
    stats_df[(stats_df.sub_scale == 'Q5.1') | (stats_df.sub_scale == 'Q7.1') | (stats_df.sub_scale == 'Q10.1') | (
                stats_df.sub_scale == 'Q16.1')][conj.rationality == 'irrational'].long_answer = 1
    stats_df[(stats_df.sub_scale == 'Q5.1') | (stats_df.sub_scale == 'Q7.1') | (stats_df.sub_scale == 'Q10.1') | (
                stats_df.sub_scale == 'Q16.1')][conj.rationality == 'half'].long_answer = 2
    stats_df[(stats_df.sub_scale == 'Q5.1') | (stats_df.sub_scale == 'Q7.1') | (stats_df.sub_scale == 'Q10.1') | (
                stats_df.sub_scale == 'Q16.1')][conj.rationality == 'rational'].long_answer = 3
    disj = stats_df[(stats_df.sub_scale == 'Q12.1') | (stats_df.sub_scale == 'Q14.1') | (stats_df.sub_scale == 'Q18.1')]
    stats_df[(stats_df.sub_scale == 'Q12.1') | (stats_df.sub_scale == 'Q14.1') | (stats_df.sub_scale == 'Q18.1')][disj.rationality == 'irrational'].long_answer = 3
    stats_df[(stats_df.sub_scale == 'Q12.1') | (stats_df.sub_scale == 'Q14.1') | (stats_df.sub_scale == 'Q18.1')][disj.rationality == 'half'].long_answer = 2
    stats_df[(stats_df.sub_scale == 'Q12.1') | (stats_df.sub_scale == 'Q14.1') | (stats_df.sub_scale == 'Q18.1')][disj.rationality == 'rational'].long_answer = 1

    return stats_df

def NARS_data(stats_df, df):
    # todo FOR NARS: www.statisticshowto.com/cronbachs-alpha-spss/
    '''
    Calculating NARS for all users
    http://uhra.herts.ac.uk/bitstream/handle/2299/9641/SyrdalDDautenhahn.pdf?sequence=1
    :param stats_df: dataframe for inferential statistics.
    :param df: raw dataframe
    :return:
    '''
    NARS_sub_meaning = {'S1': 'Situations and Interactions with Robots', 'S2':'Social Influence of Robots', 'S3': 'Emotions in Interaction with Robots'}
    NARS_sub = {'S1': [4,7,8,9,10,12], 'S2':[1,2,11,13,14], 'S3': [3,5,6]}
    for n in NARS_sub:
        NARS_sub[n] = map(str,NARS_sub[n])
    NARS = pd.DataFrame.from_dict([NARS_sub_meaning,NARS_sub])
    NARS = NARS.rename({0:'meaning',1:'average'}, axis='index')
    for s in NARS:
        temps = df[df.question == 'NARS'][df['option'][df.question == 'NARS'].isin(NARS[s]['average'])].drop(
            ['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float') # choose only the users answers
        temps = min_max_norm(temps) # Normalization
        ms = pd.DataFrame(data=[['','NARS', s, NARS[s]['meaning']] + temps.mean(axis=0).tolist()], columns=stats_df.columns)
        stats_df = stats_df.append(ms)
        pass
    return stats_df

def BFI_data(stats_df, df):
    '''
        Calculating BFI for all users
        https://www.ocf.berkeley.edu/~johnlab/pdfs/2008chapter.pdf
        :param stats_df: dataframe for inferential statistics.
        :param df: raw dataframe
        :return:
        '''
    BFI_sub_meaning = {'S1':'Extraversion', 'S2':'Agreeableness', 'S3':'Conscientiousness', 'S4':'Neuroticism', 'S5':'Openness'}
    BFI_rev = {'S1':[0,1,0,0,1,0,1,0], 'S2':[1,0,1,0,0,1,0,1,0], 'S3':[0,1,0,1,1,0,0,0,1],'S4':[0,1,0,0,1,0,1,0],'S5':[0,0,0,0,0,0,1,0,1,0]}
    BFI_sub = {'S1':[1,6,11,16,21,26,31,36], 'S2':[2,7,12,17,22,27,32,37,42], 'S3':[3,8,13,18,23,28,33,38,43],'S4':[4,9,14,19,24,29,34,39],'S5':[5,10,15,20,25,30,35,40,41,44]}
    for n in BFI_sub:
        BFI_sub[n] = map(str,BFI_sub[n])
    BFI = pd.DataFrame.from_dict([BFI_sub_meaning, BFI_sub])
    BFI = BFI.rename({0:'meaning',1:'average'}, axis='index')

    for s in BFI:
        a = df['option'][df.question == 'BFI'][:-1].astype('int')
        temps = df[df.question == 'BFI'][:-1][a.isin(BFI[s]['average'])].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')  # choose only the users answers
        rev = np.array(BFI_rev[s], dtype=bool)
        temps.loc[temps.index[rev]] = temps.loc[temps.index[rev]].applymap(bfi_revrse)
        # temps = min_max_norm(temps) # Normalization todo: fix this
        temps / temps.max()
        ms = pd.DataFrame(data=[['','BFI', s, BFI[s]['meaning']] + temps.mean(axis=0).tolist()], columns=stats_df.columns)
        stats_df = stats_df.append(ms)
    return stats_df

def bfi_revrse(v):
    '''
    reverse value for BFI analysis
    :param v: value to reverse
    :return: reversed value
    '''
    revrsed_values = [5.,4.,3.,2.,1.]
    return revrsed_values[int(v)-1]

def GODSPEED_data(stats_df, df, robot):
    '''
        Calculating Goddspeed for all users
        https://link.springer.com/content/pdf/10.1007%2Fs12369-008-0001-3.pdf
    :param stats_df: dataframe for inferential statistics.
    :param df: raw dataframe
    :param robot: which robot 'red', 'blue'
    :return:
    '''
    Godspeed_sub_meaning = {'S1':'Anthropomorphism', 'S2':'Animacy', 'S3':'Likeability', 'S4':'Perceived Intelligence', 'S5':'Perceived Safety'}
    Godspeed_sub = {'S1':np.arange(0,5), 'S2':np.arange(5,11), 'S3':np.arange(11,16),'S4':np.arange(16,21),'S5':np.arange(21,24)}
    for n in Godspeed_sub:
        Godspeed_sub[n] = map(str,Godspeed_sub[n])
    Godspeed = pd.DataFrame.from_dict([Godspeed_sub_meaning, Godspeed_sub])
    Godspeed = Godspeed.rename({0:'meaning',1:'average'}, axis='index')
    for s in Godspeed:
        if robot == 'red':
            q = 'GODSPEED1'
        else:
            q = 'GODSPEED2'
        temps = df[df.question == q][df['option'][df.question == q].isin(Godspeed[s]['average'])].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')  # choose only the users answers
        temps = temps.fillna(0)
        temps = min_max_norm(temps) # Normalization
        ms = pd.DataFrame(data=[[robot,q, s, Godspeed[s]['meaning']] + temps.mean(axis=0).tolist()], columns=stats_df.columns)
        stats_df = stats_df.append(ms)
    return stats_df

def preference_data(stats_df, df):
    '''
    Which robot the user preferred based on the questions during the questonnaire.
    :param stats_df: dataframe for inferential statistics.
    :param df: raw dataframe
    :return: stats_df, raw_df - updated, users2exclude - users that were excluded because they didn't answer all preference questions.
    '''
    temps = df[df.full_text.str.contains('you agree with?')].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')

    temps_qs = temps.copy()
    qs = temps_qs.index
    temps_qs.columns = stats_df.columns[4:]
    temps_qs = temps_qs.reindex(columns=stats_df.columns)
    temps_qs.feature = 'q_preference'
    temps_qs.sub_scale = qs
    temps_qs.meaning = ''

    if temps.empty:
        stats_df = temps
    else:
        temps = temps.apply(pd.value_counts)/float(temps.__len__())
        temps = temps.fillna(0)
        temps.columns = stats_df.columns[4:]
        temps = temps.reindex(columns=stats_df.columns)
        temps.feature = 'r_preference'
        temps.sub_scale = 'summary'
        temps.meaning = 'Count (Normalized) participant chose this robot'
        # add diff()

        # users2exclude = temps.sum()[temps.sum() != 1].index[4:]
        # print(str(users2exclude.__len__())+' users were excluded because they did not answer all the preference questions.')

        # temps.loc[temps.index[0], :][4:] - temps.loc[temps.index[1], :][4:]
        # temps['3.0',:] = temps.sum()

        if temps.shape[0] == 2:
            temps.robot = ['red', 'blue']
        else:
            if temps.index[0] == '1.':
                temps.robot = 'red'
            elif temps.index[0] == '2.':
                temps.robot = 'blue'
        stats_df = stats_df.append(temps)

        stats_df = stats_df.append(temps_qs)

    return stats_df #, df , users2exclude

def prefernce_dataframe_index(raw_df):
    '''
    crating dataframe with the preference index per question
    :param raw_df: dataframe of raw data
    :return: pref_df, users_pref
    '''
    temps = raw_df[raw_df.full_text.str.contains('agree with?')].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')
    qs = ['investments', 'analyst', 'jury', 'bartender', 'prefer']
    for q in qs:
        if q == 'prefer':
            a = raw_df[(raw_df.full_text.str.contains(q))].drop(['question', 'option', 'full_text', 'dict_text'],axis=1)
            preference = a.loc[a.index[0],:]
            open_answers = a.loc[a.index[1],:]
            temps1 = temps1.append(preference.astype('float'))
        else:
            if 'temps1' in locals():
                temps1 = temps1.append(raw_df[(raw_df.full_text.str.contains(q))].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float'))
            else:
                temps1 = raw_df[(raw_df.full_text.str.contains(q))].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')


    temps1 = temps1.replace(1, 2.)
    temps1 = temps1.replace(4, 1.)
    temps1 = temps1.replace(5, 2.)
    temps1.index = qs
    temps = temps.append(temps1)

    blue_rationality = raw_df[(raw_df.question == 'blue_robot') & (raw_df.full_text == 'rationality')]
    red_rationality = raw_df[(raw_df.question == 'red_robot') & (raw_df.full_text == 'rationality')]
    # reversing the DON'T question
    temps.loc['Q16.1', :] = temps.loc['Q16.1',:].replace(1.0,3.0).replace(2.0,1.0)
    temps.loc['Q16.1', :] = temps.loc['Q16.1',:].replace(3.0,2.0)
    for c in temps.columns:
        temps[c] = temps[c].replace(1.0, red_rationality[c].tolist()[0])
        temps[c] = temps[c].replace(2.0, blue_rationality[c].tolist()[0])

    for r in temps.index:
        pref_ix = pd.value_counts(temps.loc[r, :]).diff(-1) / pd.value_counts(temps.loc[r, :]).sum()
        if pref_ix.isna()[0]:
            pref_ix = pd.value_counts(temps.loc[r, :]) / pd.value_counts(temps.loc[r, :]).sum()
        if 'pref_df' in locals():
            temp_df = pd.DataFrame(data=[[r, '_'.join(pref_ix.index.tolist()), pref_ix[0]]],
                                   columns=['question', 'rationality', 'preference'])
            pref_df = pref_df.append(temp_df)
        else:
            pref_df = pd.DataFrame(data=[[r, '_'.join(pref_ix.index.tolist()), pref_ix[0]]],
                                   columns=['question', 'rationality', 'preference'])
        x = pd.value_counts(temps.loc[r, :])

        for i in range(x.__len__()):
            if 'rat_pref_df' in locals():
                rat_pref_df = rat_pref_df.append(pd.DataFrame(data = [[r, x.keys()[i], x[x.keys()[i]]]], columns=['question', 'rationality', 'preference']))
            else:
                rat_pref_df = pd.DataFrame(data = [[r, x.keys()[i], x[x.keys()[i]]]], columns=['question', 'rationality', 'preference'])

    rat_pref_df['zscore'], rat_pref_df['zprob'], rat_pref_df['binomal'], rat_pref_df['num_users'], rat_pref_df['sig'] = '', '', '', '', ''
    rat_pref_df['deployment'] = rat_pref_df['rationality'].tolist()[0] + '_' + rat_pref_df['rationality'].tolist()[1]
    rat_pref_df.zscore  = stats.mstats.zscore(rat_pref_df.preference)
    rat_pref_df.zprob   = stats.norm.cdf(rat_pref_df.preference)

    rat_pref_df.num_users = temps.shape[1]
    rat_pref_df = rat_pref_df.reset_index(drop=True)

    for i in rat_pref_df.index:
        rat_pref_df.loc[i, 'binomal'] = stats.binom_test(x=rat_pref_df.loc[i, 'preference'], n=rat_pref_df.loc[i, 'num_users'], p=.5)
        rat_pref_df.loc[i, 'sig'] = rat_pref_df.loc[i, 'binomal'] < 0.05

    pref_df = pref_df.reset_index(drop=True)
    pref_df.loc[pref_df.rationality != pref_df.rationality[0], 'preference'] = -pref_df.loc[pref_df.rationality != pref_df.rationality[0], 'preference']
    rat = pd.value_counts(pref_df.rationality).index[0]
    pref_df.rationality = rat
    users_pref = temps.copy()
    users_pref.columns = \
        raw_df[(raw_df.question == 'ID')].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).loc[
            'ResponseId']

    return pref_df, users_pref, open_answers, rat_pref_df
    
def questions(stats_df, raw_df):
    '''
    Analyzing our questions.
    :param stats_df: dataframe for inferential statistics.
    :param raw_df: raw dataframe
    :return:
    '''

    # count how many times in our question, each robot was chosen.
    # temps = stats_df.loc[stats_df[stats_df.feature == 'q_preference'].index,stats_df.columns[4:]].apply(pd.value_counts)/5.
    temps1 = stats_df.loc[stats_df[stats_df.feature == 'q_preference'].index,stats_df.columns[4:]].apply(pd.value_counts)/7. # normalization by number of questions

    p = 'Click to write the question text - '
    qp = ['Which robot would you prefer?', 'Which robot would you trust to manage your investments portfolio?',
          'Which robot do you think will serve as a good jury member?',
          'Which robot do you think will be a better analyst?', 'Which robot would you like as a bartender?']
    features = []
    for q in qp:
        try:
            temps = temps.append(raw_df[raw_df.full_text.str.contains(q)].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype(
                    'float'))
        except:
            temps = raw_df[raw_df.full_text.str.contains(q)].drop(['question', 'option', 'full_text', 'dict_text'],
                                                           axis=1).astype('float')
        features += [q.split(' ')[-1].replace('?', '')]

    # temps = temps.replace(1, 'blue')
    # temps = temps.replace(4, 'red')
    temps = temps.replace(1, 2.)
    temps = temps.replace(4, 1.)
    temps.columns = stats_df.columns[4:]
    temps = temps.reindex(columns=stats_df.columns)
    temps.meaning = 'choice'
    temps.sub_scale = features
    temps.feature = 'q_preference'
    stats_df = stats_df.append(temps)

    # todo: check if I flipped the question - I think I did

    meaning = 'Count (Normalized) participant chose this robot'

    # todo: uncomment???
    # try:
    #     stats_df = stats_df.append(pd.DataFrame(data=[['blue', 'q_preference', 'summary', meaning]+temps1.loc[4.].tolist()], columns = stats_df.columns))
    # except:
    #     pass
    # try:
    #     stats_df = stats_df.append(pd.DataFrame(data=[['red', 'q_preference', 'summary', meaning]+temps1.loc[1.].tolist()], columns = stats_df.columns))
    # except:
    #     pass

    return stats_df


def stats_df_reformat(stats_df):
    '''
    Reformat to much easier dataframe to work with.
    '''
    for i, user in enumerate(stats_df.columns[4:]):
        temp = stats_df[stats_df.columns[:4]]
        # temp['answers'] = stats_df[user]
        temp['answers'] = stats_df[user]
        temp['userID']      = user
        temp['age']         = stats_df[stats_df.feature == 'age'][user].tolist()[0]
        temp['gender']         = stats_df[stats_df.feature == 'gender'][user].tolist()[0]
        temp['education']   = stats_df[stats_df.feature == 'education'][user].tolist()[0]
        temp.insert(0,'side',np.nan)
        temp.insert(0,'rationality',np.nan)
        temp.loc[temp.robot == 'red', 'side']         = stats_df[user][(stats_df.robot == 'red_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'side')].tolist()[0]
        temp.loc[stats_df.robot == 'red', 'rationality']  = stats_df[user][(stats_df.robot == 'red_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'rationality')].tolist()[0]
        temp.loc[stats_df.robot == 'blue', 'side']        = stats_df[user][(stats_df.robot == 'blue_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'side')].tolist()[0]
        temp.loc[stats_df.robot == 'blue', 'rationality'] = stats_df[user][(stats_df.robot == 'blue_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'rationality')].tolist()[0]

        temp['rationality'] = temp['rationality'].astype('category')
        temp['side'] = temp['side'].astype('category')
        temp['gender'] = temp['gender'].astype('category')
        temp['education'] = temp['education'].astype('category')

        temp = temp.drop(temp[(temp.feature=='age') | (temp.feature=='education') | (temp.feature=='gender') | (temp.robot=='red_robot') | (temp.robot=='blue_robot')].index.tolist())

        temp.loc[(temp.feature == 'q_preference') & ((temp.answers == 1.) | (temp.answers == 5.)), 'robot'] = 'red'
        temp.loc[(temp.feature == 'q_preference') & (temp.answers == 2.), 'robot'] = 'blue'
        temp.loc[temp[temp.robot == 'red'].index.tolist(), 'side'] = temp[(temp.robot == 'red')].side.unique()[0]
        temp.loc[temp[temp.robot == 'red'].index.tolist(), 'rationality'] = temp[(temp.robot == 'red')].rationality.unique()[0]
        temp.loc[temp[temp.robot == 'red'].index.tolist(), 'rationality'] = temp[(temp.robot == 'red')].rationality.unique()[0]
        temp.loc[temp[temp.robot == 'blue'].index.tolist(), 'side'] = temp[(temp.robot == 'blue')].side.unique()[0]
        temp.loc[temp[temp.robot == 'blue'].index.tolist(), 'rationality'] = temp[(temp.robot == 'blue')].rationality.unique()[0]

        if i == 0:
            sdf = temp
        else:
            sdf = sdf.append(temp)

        # sdf.loc[((sdf.feature == 'r_preference') | (sdf.feature == 'q_preference')) & (sdf.answers == 1.), 'robot'] = 'red'
        # sdf.loc[((sdf.feature == 'r_preference') | (sdf.feature == 'q_preference')) & (sdf.answers == 2.), 'robot'] = 'blue'
        # sdf.loc[sdf[sdf.robot == 'red'].index.tolist(), 'side'] = sdf[(sdf.robot == 'red')].side.unique()[0]
        # sdf.loc[sdf[sdf.robot == 'red'].index.tolist(), 'rationality'] = sdf[(sdf.robot == 'red')].rationality.unique()[0]
        # sdf.loc[sdf[sdf.robot == 'red'].index.tolist(), 'rationality'] = sdf[(sdf.robot == 'red')].rationality.unique()[0]
        # sdf.loc[sdf[sdf.robot == 'blue'].index.tolist(), 'side'] = sdf[(sdf.robot == 'blue')].side.unique()[0]
        # sdf.loc[sdf[sdf.robot == 'blue'].index.tolist(), 'rationality'] = sdf[(sdf.robot == 'blue')].rationality.unique()[0]
 
    return sdf

def min_max_norm(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    df_normalized.columns = df.columns
    df_normalized.index = df.index
    return df_normalized

if __name__ == "__main__":

    files = os.listdir('data/raw/')
    for f in files:
        path = 'data/raw/'+f
        print(path)
        # (f == 'Emma_questionnaire_video_rrrlbh.csv') | (f == 'Emma_questionnaire_video_rbrlrh.csv') | (f == 'Emma_questionnaire_video_rrhlbi.csv'):
        # if  (f == 'Emma_questionnaire_video_rrrlbh.csv'):
        #     continue
        raw_df, rDeployment = raw_data_extraction(path)
        # stats_df = create_stats_df(raw_df, rDeployment)
        create_stats_df(raw_df, rDeployment)

    print('raw_df and stats_df were created!')

