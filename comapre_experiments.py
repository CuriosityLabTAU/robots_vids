import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from new_reformat import *
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from sklearn import preprocessing
# from pyvttbl import Anova1way


sns.set_context('paper')
plt.style.use('bmh')

right_order = [0,1,3,2,4,5,6]

questions_df = {}

### loading the data for free ranking
raw_df_ranking = pd.read_csv('data/people_rankings/raw_df_ranking.csv', index_col=0)
qns_ranking = ['Q6_rate', 'Q8_rate', 'Q12_rate', 'Q10_rate', 'Q14_rate', 'Q16_rate','Q18_rate']

raw_df_ranking[raw_df_ranking != 0] = 1
br = raw_df_ranking.sum(axis = 0) / raw_df_ranking.shape[0]
br = br[np.array(br.index)[right_order]]
br = br.reset_index(drop=True)
questions_df['ranking'] = br

### loading the data for choosing between rankings
raw_df_choosing = pd.read_csv('data/people_rankings/raw_df_choose_ranking.csv', index_col=0)
qns_choosing = ['Q9', 'Q11', 'Q16', 'Q14', 'Q18', 'Q20', 'Q22']

a = raw_df_choosing[qns_choosing].copy()
a[a != 'r'] = 1
a[a == 'r'] = 0
ar = a.sum(axis = 0) / a.shape[0]
ar = ar[np.array(ar.index)[right_order]]
ar = ar.reset_index(drop=True)
questions_df['chosing'] = ar

### loading the data for choosing between robots
raw_df_robots = pd.read_csv('data/dataframes/new_raw_df.csv', index_col=0)
qns_robots = ['Q5.1','Q7.1','Q12.1', 'Q10.1','Q14.1','Q16.1', 'Q18.1']

c = raw_df_robots[qns_robots].copy()
cr = c.sum(axis = 0) / c.shape[0]
cr = cr[np.array(cr.index)[right_order]]
cr = cr.reset_index(drop=True)
questions_df['robots'] = cr


### creating the dataframe of the rankings per questions per experiment
questions_df = pd.DataFrame.from_dict(questions_df)
questions_df = 1 - questions_df # for rational preference
questions_df['q'] = list(questions_df.index)
save_table(questions_df, 'data/paper/', '01fal_rate_per_question_per_experiment')

questions_df.to_csv('data/paper/00questions_df.csv')

### test if percentage per question is significantly
df_anova_qn = pd.DataFrame()
for qn, [q1, q2, q3] in enumerate(zip(qns_choosing, qns_ranking, qns_robots)):
    df_anova = pd.concat((pd.DataFrame(data = np.array([np.array(a[q1]), np.ones(a.shape[0])]).T, columns = ['choice', 'experiment']),
        pd.DataFrame(data = np.array([np.array(raw_df_ranking[q2]), np.zeros(raw_df_ranking.shape[0])]).T,
                 columns = ['choice', 'experiment']),
        pd.DataFrame(data = np.array([np.array(c[q3]), 2*np.ones(c.shape[0])]).T,
                 columns = ['choice', 'experiment'])), axis = 0)
    df_anova['choice'] = 1 - df_anova['choice']
    df_anova['qn'] = qn

    df_anova['choice'] = pd.to_numeric(df_anova['choice'])
    print('####### question: '+ q1 +' #######')
    F, p = stats.f_oneway(df_anova['choice'][df_anova['experiment'] == 0], df_anova['choice'][df_anova['experiment'] == 1],
                   df_anova['choice'][df_anova['experiment'] == 2])
    print('ANOVA: %.2f, %.4f' %(F,p))
    if p < .05:
        res = pairwise_tukeyhsd(df_anova['choice'], df_anova['experiment'])
        print(res)
    df_anova_qn = pd.concat((df_anova_qn, df_anova), axis = 0)

    sns.barplot(data = df_anova_qn, x = 'qn', y = 'choice', hue = 'experiment')

questions_df_multilinear = pd.melt(questions_df, id_vars=['q'], value_vars=questions_df.columns[:-1],
                                   var_name='experiment', value_name='percentage')
questions_df_multilinear.to_csv('data/paper/00questions_df_multilinear.csv')

### plotting the regression lines for all the frequency
figure, ax = plt.subplots(1,1)
for x in questions_df.columns:
    if x != 'q':
        sns.regplot(x='q', y=x, data=questions_df, label = x, ax=ax)
ax.set_ylabel('Prefer towards rational')
plt.legend()

### check the correlation of the preference to each of the questions
d = raw_df_robots[qns_robots].copy()
d = d[d.columns[right_order]]
d['prefer'] = raw_df_robots['prefer']
d1,d2 = calculate_corr_with_pvalues(d, questionnaires = False)

# In order to compare slopes look at this two URLs:
# 1. http://blog.minitab.com/blog/adventures-in-statistics-2/how-to-compare-regression-lines-between-different-models
# 2. http://www.csub.edu/~psmith3/teaching/310-12.pdf

### Multi linear regression for the robots experiment.
df = raw_df_robots.copy()
formula = 'rLikability ~ prefer * rational_agree'
mlr = smf.ols(formula, data=df).fit()
print(mlr.summary())

# calculate_corr_with_pvalues(raw_df_robots[['rLikability', 'prefer', 'rational_agree']], questionnaires = False)

##$ logistic regression
# # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# min_max_scaler = preprocessing.MinMaxScaler()
# df['rLikability'] = min_max_scaler.fit_transform(df['rLikability'])
# mlr = smf.logit(formula, data=df).fit()
# print(mlr.summary())


plt.show()


