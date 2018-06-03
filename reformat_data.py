import pandas as pd
import numpy as np

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

    # need to know what the question numbers of BFI, NARS and Godspeed for the reformatting_guide.csv

    for i in raw_df.index:
        # todo: get this!!! how to choose the right row and split it!
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
        if raw_df[raw_df.question == q].__len__() == 44: #45 rt
            raw_df.question[raw_df.question == q] = 'BFI'
        if raw_df[raw_df.question == q].__len__() == 14: #15 rt
            raw_df.question[raw_df.question == q] = 'NARS'
        if (raw_df[raw_df.question == q].__len__() == 24) | (raw_df[raw_df.question == q].__len__() == 25):
            raw_df.question[raw_df.question == q] = 'GODSPEED' + str(i)
            i+=1

    # Find the rows of the demographic questions.
    raw_df.loc[raw_df.full_text == 'What is your age?' , 'question'] = 'age'
    raw_df.loc[raw_df.full_text == 'To which gender identity do you most identify?' , 'question'] = 'gender'
    return raw_df


def create_stats_df(raw_df):
    '''
    Creating statistical dataframe for inferential analysis.
    :param raw_df: dataframe containing the raw data
    :return: stats_df: dataframe with for inferential analysis.
    '''
    usersID = raw_df[raw_df.question == 'ID'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1)
    # usersAGE = df[df.question == 'age'].drop(['question','option','full_text','dict_text'], axis=1).loc['Q1.4'].astype(float).tolist() # todo: uncomment on real age
    usersAGE = raw_df[raw_df.question == 'age'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).loc['Q1.4'].tolist()
    usersGENDER = raw_df[raw_df.question == 'gender'].drop(['question', 'option', 'full_text', 'dict_text'], axis=1).loc['Q1.3'].astype(float).tolist()
    cnames = ['robot','feature', 'sub_scale', 'meaning'] + usersID.transpose()['ResponseId'].tolist()

    stats_df = pd.DataFrame(columns = cnames) # Inferential dataframe

    stats_df = stats_df.append(pd.DataFrame(data = [['','gender','','']+usersGENDER], columns=cnames))
    stats_df = stats_df.append(pd.DataFrame(data = [['','age','','']+usersAGE], columns=cnames))

    stats_df = NARS_data(stats_df, raw_df)
    stats_df = BFI_data(stats_df, raw_df)
    stats_df = GODSPEED_data(stats_df, raw_df, 'red')
    stats_df = GODSPEED_data(stats_df, raw_df, 'blue')
    stats_df = preference_data(stats_df, raw_df)
    stats_df = questions(stats_df, raw_df)

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
    temps = temps.apply(pd.value_counts)/.7
    temps.columns = stats_df.columns[4:]
    temps = temps.reindex(columns=stats_df.columns)
    temps.feature = 'q_preference'
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

    temps.columns = stats_df.columns[4:]
    temps = temps.reindex(columns=stats_df.columns)
    temps.meaning = qp
    temps.feature = features
    stats_df = stats_df.append(temps)
    return stats_df


if __name__ == "__main__":
    # todo: check red/ blue values!!!!
    raw_df = raw_data_extraction('Emma_questionnaire_video_rbrlri_June_2_2018.csv')
    stats_df = create_stats_df(raw_df)
    print('end')