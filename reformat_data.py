import pandas as pd
import numpy as np
import os

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

    raw_df, users_after_exclusion = trap_exclusion(raw_df)
    raw_df = response_time_exclusion(raw_df, users_after_exclusion)

    raw_df.to_csv('data/raw_dataframe_'+rDeployment)     # saving the data frame
    return raw_df, rDeployment

def trap_exclusion(raw_df):
    '''
    Exclude users which answered the trap quesion wrong
    :param raw_df: raw data dataframe
    :return: raw_Df
    '''
    # trap question - exclude users
    a = raw_df[raw_df.columns[5:]][raw_df.question == 'trap_question']
    all_users = set(raw_df.columns[4:])
    trap_value = '4'
    users_after_exclusion = set(a[a == trap_value].dropna(axis=1).columns)
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

    return raw_df

def create_stats_df(raw_df, rDeployment):
    '''
    Creating statistical dataframe for inferential analysis.
    :param raw_df: dataframe containing the raw data
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

    stats_df.to_csv('data/stats_dataframe_'+rDeployment)     # saving the data frame

    return stats_df


# todo FOR NARS: www.statisticshowto.com/cronbachs-alpha-spss/
def NARS_data(stats_df, df):
    '''
    Calculating NARS for all users
    http://uhra.herts.ac.uk/bitstream/handle/2299/9641/SyrdalDDautenhahn.pdf?sequence=1
    :param stats_df: dataframe for inferential statistics.
    :param df: raw dataframe
    :return:
    '''
    NARS_sub_meaning = {'S1': 'Situations and Interactions with Robots', 'S2':'Social Influence of Robots', 'S3': 'Emotions in Interaction with Robots'}
    NARS_sub = {'S1': [4,7,8,9,10,12], 'S2':[1,2,11,13,14], 'S3': [3,5,6]}
    NARS = pd.DataFrame.from_dict([NARS_sub_meaning,NARS_sub])
    NARS = NARS.rename({0:'meaning',1:'average'}, axis='index')
    for s in NARS:
        temps = df[df.question == 'NARS'][df['option'][df.question == 'NARS'].isin(NARS[s]['average'])].drop(
            ['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float') # choose only the users answers
        temps = temps/5. # Normalization
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
    BFI = pd.DataFrame.from_dict([BFI_sub_meaning, BFI_sub])
    BFI = BFI.rename({0:'meaning',1:'average'}, axis='index')

    for s in BFI:
        temps = df[df.question == 'BFI'][df['option'][df.question == 'BFI'].isin(BFI[s]['average'])].drop(
            ['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')  # choose only the users answers
        rev = np.array(BFI_rev[s], dtype=bool)
        temps.loc[temps.index[rev]] = temps.loc[temps.index[rev]].applymap(bfi_revrse)
        temps = temps / 5.  # Normalization
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
    Godspeed = pd.DataFrame.from_dict([Godspeed_sub_meaning, Godspeed_sub])
    Godspeed = Godspeed.rename({0:'meaning',1:'average'}, axis='index')
    for s in Godspeed:
        if robot == 'red':
            q = 'GODSPEED1'
        else:
            q = 'GODSPEED2'
        temps = df[df.question == q][df['option'][df.question == q].isin(Godspeed[s]['average'])].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')  # choose only the users answers
        temps = temps/5. # Normalization
        ms = pd.DataFrame(data=[[robot,'Godspeed', s, Godspeed[s]['meaning']] + temps.mean(axis=0).tolist()], columns=stats_df.columns)
        stats_df = stats_df.append(ms)
    return stats_df

def preference_data(stats_df, df):
    '''
    Which robot the user preferred based on the questions during.
    :param stats_df: dataframe for inferential statistics.
    :param df: raw dataframe
    :return: [red preference, blue preference]
    '''
    temps = df[df.full_text == 'Which robot do you agree with?'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype('float')
    temps = temps.apply(pd.value_counts)/7.
    temps.columns = stats_df.columns[4:]
    temps = temps.reindex(columns=stats_df.columns)
    temps.feature = 'r_preference'
    temps.sub_scale = 'summary'
    temps.meaning = 'Count (Normalized) participant chose this robot'
    temps.robot = ['red', 'blue']
    stats_df = stats_df.append(temps)
    return stats_df


def questions(stats_df, raw_df):
    '''
    Analyzing our questions.
    :param stats_df: dataframe for inferential statistics.
    :param raw_df: raw dataframe
    :return:
    '''
    p = 'Click to write the question text - '
    qp = ['Which robot would you prefer?', 'Which robot would you trust to manage your investments portfolio?',
          'Which robot do you think will serve as a good jury member?',
          'Which robot do you think will be a better analyst?', 'Which robot would you like as a bartender?']
    features = []
    for q in qp:
        try:
            temps = temps.append(
                raw_df[raw_df.full_text == p + q].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).astype(
                    'float'))
        except:
            temps = raw_df[raw_df.full_text == p + q].drop(['question', 'option', 'full_text', 'dict_text'],
                                                           axis=1).astype('float')
        features += [q.split(' ')[-1].replace('?', '')]

    temps.replace(1, 'blue')
    temps.replace(4, 'red')
    temps.columns = stats_df.columns[4:]
    temps = temps.reindex(columns=stats_df.columns)
    temps.meaning = qp
    temps.sub_scale = features
    temps.feature = 'q_preference'
    stats_df = stats_df.append(temps)


    # count how many times in our question, each robot was chosen.
    temps = stats_df.loc[stats_df[stats_df.feature == 'q_preference'].index,stats_df.columns[4:]].apply(pd.value_counts)/5.

    meaning = 'Count (Normalized) participant chose this robot'
    # stats_df = stats_df.append(
    #     pd.DataFrame(data=[['blue', 'q_preference', 'summary', meaning] + temps.loc['blue'].tolist()],
    #                  columns=stats_df.columns))
    # stats_df = stats_df.append(
    #     pd.DataFrame(data=[['red', 'q_preference', 'summary', meaning] + temps.loc['red'].tolist()],
    #                  columns=stats_df.columns))
    try:
        stats_df = stats_df.append(pd.DataFrame(data=[['blue', 'q_preference', 'summary', meaning]+temps.loc[1.].tolist()], columns = stats_df.columns))
    except:
        pass
    try:
        stats_df = stats_df.append(pd.DataFrame(data=[['red', 'q_preference', 'summary', meaning]+temps.loc[4.].tolist()], columns = stats_df.columns))
    except:
        pass

    return stats_df


def stats_df_reformat(stats_df):
    '''
    Reformat to much easier dataframe to work with.
    '''
    for i, user in enumerate(stats_df.columns[4:]):
        temp = stats_df[stats_df.columns[:4]]
        temp['answers'] = stats_df[user]
        temp['userID']      = user
        temp['age']         = stats_df[stats_df.feature == 'age'][user].tolist()[0]
        temp['gender']         = stats_df[stats_df.feature == 'gender'][user].tolist()[0]
        temp['education']   = stats_df[stats_df.feature == 'education'][user].tolist()[0]
        # temp['rationality'] = stats_df[stats_df.feature == 'rationality'][user].tolist()[0]
        temp['side'] = ''
        temp['rationality'] = ''
        temp.loc[temp.robot == 'red', 'side']         = stats_df[user][(stats_df.robot == 'red_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'side')].tolist()[0]
        temp.loc[stats_df.robot == 'red', 'rationality']  = stats_df[user][(stats_df.robot == 'red_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'rationality')].tolist()[0]
        temp.loc[stats_df.robot == 'blue', 'side']        = stats_df[user][(stats_df.robot == 'blue_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'side')].tolist()[0]
        temp.loc[stats_df.robot == 'blue', 'rationality'] = stats_df[user][(stats_df.robot == 'blue_robot') & (stats_df.feature == 'deployment') & (stats_df.sub_scale == 'rationality')].tolist()[0]

        temp['rationality'] = temp['rationality'].astype('category')
        temp['side'] = temp['side'].astype('category')
        temp['gender'] = temp['gender'].astype('category')
        temp['education'] = temp['education'].astype('category')

        temp = temp.drop(temp[(temp.feature=='age') | (temp.feature=='education') | (temp.feature=='gender') | (temp.robot=='red_robot') | (temp.robot=='blue_robot')].index.tolist())


        if i == 0:
            sdf = temp
        else:
            sdf = sdf.append(temp)
    return sdf

if __name__ == "__main__":

    files = os.listdir('data/raw/')
    for f in files:
        path = 'data/raw/'+f
        print(path)
        if (f == 'Emma_questionnaire_video_rrrlbh.csv') | (f == 'Emma_questionnaire_video_rbrlrh.csv') | (f == 'Emma_questionnaire_video_rrhlbi.csv'):
            continue
        raw_df, rDeployment = raw_data_extraction(path)
        # raw_df, rDeployment = raw_data_extraction('data/Emma_questionnaire_video_rbilrr_June_4_2018.csv')
        stats_df = create_stats_df(raw_df, rDeployment)

    print('raw_df and stats_df were created!')

    # todo: response time, trap question
