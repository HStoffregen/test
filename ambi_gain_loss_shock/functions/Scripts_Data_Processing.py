import re
import datetime
import pandas as pd
import numpy as np
import glob
import pickle
from IPython.core.debugger import Tracer
import boto
import sys

def push_results_to_s3(fname,fpath,bucket_folder='cdm'):
    '''
    Example: push_results_to_s3('../notebook_htmls/Notebook_Exploring_New_Data_v1.html',
    'Notebook_Exploring_New_Data_v1.html')

    Requires S3 key in user's home directory in a folder ~/.aws

    '''

    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    s3_connection = boto.connect_s3()
    bucket = s3_connection.get_bucket('bishopbucket')
    key = boto.s3.key.Key(bucket, 'proj_'+bucket_folder+'/Results/'+fname)
    key.set_contents_from_filename(fpath,
        cb=percent_cb, num_cb=10)
    s3_connection.close()
    print('')
    print('https://s3-us-west-2.amazonaws.com/bishopbucket/proj_'+bucket_folder+'/Results/'+fname)



def get_date_from_filename(f):

    try:
        match = re.search(r'\d{4}_\d{2}_\d{2}',f)
        date = datetime.datetime.strptime(match.group(), '%Y_%m_%d').date()
    except:
        try:
            match = re.search(r'\d{4}_\d{2}_\d{1}',f)
            date = datetime.datetime.strptime(match.group(), '%Y_%m_%d').date()
        except:
            try:
                match = re.search(r'\d{4}_\d{1}_\d{2}',f)
                date = datetime.datetime.strptime(match.group(), '%Y_%m_%d').date()
            except:
                match = re.search(r'\d{4}_\d{1}_\d{1}',f)
                date = datetime.datetime.strptime(match.group(), '%Y_%m_%d').date()
    return(date)

def get_gain_loss_from_filename(f):
    if 'gain' in f:
    	task = 'gain'
    if 'loss' in f:
	    task = 'loss'
    if 'combined' in f:
        task='combined'
    return(task)


# get data function
def get_trial_table(filename,combined=False):
    # read the table #
    trial_table = pd.read_csv(filename)

    #from IPython.core.debugger import Tracer    #Tracer()()

    # remove practice trials
    if combined==False:
    	trial_table = trial_table.loc[trial_table['practice']==False,]
    else:
    	if 'false' in trial_table['practice'].unique():
    		trial_table = trial_table.loc[trial_table['practice']=='false',]
    	if 'False' in trial_table['practice'].unique():
    		trial_table = trial_table.loc[trial_table['practice']=='False',]

    ##################
    # switch x and o # s
    # prob_o_r is prob_outcome #
    if combined ==True:
    	o_l = trial_table.revealed_o_l.copy()
    	o_r = trial_table.revealed_o_r.copy()
    	x_l = trial_table.revealed_x_l.copy()
    	x_r = trial_table.revealed_x_r.copy()
    	trial_table.revealed_o_l = x_l
    	trial_table.revealed_o_r = x_r
    	trial_table.revealed_x_l = o_l
    	trial_table.revealed_x_r = o_r

    #  keep only first instance of trials #
    trial_table = trial_table.drop_duplicates(subset='trial_number',keep='first')

    #trial_table['firstinstance'] = np.zeros(len(trial_table))
    #for i in range(200):
    #    trial_table.loc[trial_table.trial_number==i,'firstinstance']=1
    #trial_table = trial_table.loc[trial_table['firstinstance']==1,]

    if combined==False:
	       num_trials=200
    else:
	       num_trials=300


    # if less than 200 trials, fill in with NaN's
    if len(trial_table)<num_trials:
        trial_table = trial_table.set_index('trial_number').reindex(pd.Index(np.arange(0,num_trials),name='trial_number')).reset_index()
        # sets the column as index, reindex adds new levels to the index and fills the rest with nan, then reset index returns to the default index


    # rename to useful things

    # probability of outcome, not 'O' - because it's an X is

    trial_table['prob_o_l'] = trial_table.revealed_o_l / (trial_table.revealed_x_l + trial_table.revealed_o_l)
    trial_table['prob_o_r'] = trial_table.revealed_o_r / (trial_table.revealed_x_r + trial_table.revealed_o_r)
    trial_table['info_l'] = (trial_table.revealed_x_l + trial_table.revealed_o_l)/50.0
    trial_table['info_r'] = (trial_table.revealed_x_r + trial_table.revealed_o_r)/50.0
    trial_table['ambig_l']=(trial_table['info_l']!=1.).astype('int')
    trial_table['ambig_r']=(trial_table['info_r']!=1.).astype('int')
    trial_table['resp_r_1'] = (trial_table.resp=='right').astype('int') # right =1
    trial_table.loc[trial_table.resp.isnull(),'resp_r_1'] =np.nan

    trial_table.loc[trial_table['ambig_r']==1,'revealed_o_ambig']=trial_table.loc[trial_table['ambig_r']==1,'revealed_o_r'].as_matrix()
    trial_table.loc[trial_table['ambig_l']==1,'revealed_o_ambig']=trial_table.loc[trial_table['ambig_l']==1,'revealed_o_l'].as_matrix()

    trial_table.loc[trial_table['ambig_r']==1,'revealed_x_ambig']=trial_table.loc[trial_table['ambig_r']==1,'revealed_x_r'].as_matrix()
    trial_table.loc[trial_table['ambig_l']==1,'revealed_x_ambig']=trial_table.loc[trial_table['ambig_l']==1,'revealed_x_l'].as_matrix()

    # sqrt info
    trial_table['info_l_sqrt'] =np.sqrt(trial_table.info_l) # may need 1-
    trial_table['info_r_sqrt'] =np.sqrt(trial_table.info_r)

    # previous choice
    trial_table['resp_r_1_prev']= np.append(np.nan,trial_table['resp_r_1'].as_matrix()[0:-1])

    # convert to bayesian probabilities
    trial_table['prob_o_l_bayes'] = trial_table['prob_o_l'] # for non-ambiguous trials, use the regular probabilities
    trial_table['prob_o_r_bayes'] = trial_table['prob_o_r']
    trial_table.loc[trial_table['info_l']!=1.0,'prob_o_l_bayes'] = (trial_table.revealed_o_l +1) / (trial_table.revealed_x_l + trial_table.revealed_o_l + 2) # for ambiguous trials, use the posterior expectation on uniform prior.
    trial_table.loc[trial_table['info_r']!=1.0,'prob_o_r_bayes'] = (trial_table.revealed_o_r +1) / (trial_table.revealed_x_r + trial_table.revealed_o_r + 2)


    #####
    ## create things with respect to ambiguity
    #####

    # selectors
    amb_l = trial_table['ambig_l']==1
    amb_r = trial_table['ambig_r']==1
    both_unamb = (trial_table['ambig_l']==0) & (trial_table['ambig_r']==0)
    amb = (trial_table['ambig_l']==1) | (trial_table['ambig_r']==1)

    # start with Nan
    trial_table['prob_o_ambig'] = np.nan
    trial_table['prob_o_unambig'] = np.nan
    trial_table['mag_ambig'] = np.nan
    trial_table['mag_unambig'] = np.nan
    trial_table['resp_amb_1'] = np.nan

    # create probability of o on ambiguous trials
    trial_table.loc[amb_l,'prob_o_ambig']=trial_table.revealed_o_l[amb_l]/(trial_table.revealed_x_l[amb_l] + trial_table.revealed_o_l[amb_l])
    trial_table.loc[amb_r,'prob_o_ambig']=trial_table.revealed_o_r[amb_r]/(trial_table.revealed_x_r[amb_r] + trial_table.revealed_o_r[amb_r])


    # create bayesian adjusted probabilities
    trial_table.loc[amb_l,'prob_o_ambig_bayes']=(trial_table.revealed_o_l[amb_l]+1)/(trial_table.revealed_x_l[amb_l] + trial_table.revealed_o_l[amb_l]+2)
    trial_table.loc[amb_r,'prob_o_ambig_bayes']=(trial_table.revealed_o_r[amb_r]+1)/(trial_table.revealed_x_r[amb_r] + trial_table.revealed_o_r[amb_r]+2)

    # create proabbility on umambiguous (switch index right v left e.g. left revealed when right is ambiguous)
    trial_table.loc[amb_r&amb,'prob_o_unambig']=trial_table.revealed_o_l[amb_r&amb]/(trial_table.revealed_x_l[amb_r&amb] + trial_table.revealed_o_l[amb_r&amb])
    trial_table.loc[amb_l&amb,'prob_o_unambig']=trial_table.revealed_o_r[amb_l&amb]/(trial_table.revealed_x_r[amb_l&amb] + trial_table.revealed_o_r[amb_l&amb])

    # create magnitutde on ambiguous trials
    trial_table.loc[amb_l,'mag_ambig']=trial_table.mag_left[amb_l]
    trial_table.loc[amb_r,'mag_ambig']=trial_table.mag_right[amb_r]
    trial_table.loc[amb_r&amb,'mag_unambig']=trial_table.mag_left[amb_r&amb]
    trial_table.loc[amb_l&amb,'mag_unambig']=trial_table.mag_right[amb_l&amb]

    #create choice for ambi/unambig on ambiguous trials
    trial_table.loc[amb_l,'resp_amb_1'] = (trial_table.loc[amb_l,'resp']=='left').astype('int')
    trial_table.loc[amb_r,'resp_amb_1'] = (trial_table.loc[amb_r,'resp']=='right').astype('int')
    trial_table.loc[amb_r&amb,'resp_amb_1'] = (np.logical_not(trial_table.loc[amb_r&amb,'resp']=='left')).astype('int')
    trial_table.loc[amb_l&amb,'resp_amb_1'] = (np.logical_not(trial_table.loc[amb_l&amb,'resp']=='right')).astype('int')

    trial_table.loc[:,'info_amb']  = np.minimum(trial_table['info_l'].as_matrix(),trial_table['info_r'].as_matrix())
    trial_table.loc[:,'info_amb_sqrt']  = np.minimum(trial_table['info_l_sqrt'].as_matrix(),trial_table['info_r_sqrt'].as_matrix())


    trial_table = trial_table.set_index('trial_number',drop=False)


    # change strings to float
    for i in range(num_trials):
    	for col in ['mag_left','mag_right','mag_ambig','mag_unambig']:
    		ml = trial_table.loc[trial_table['trial_number']==i,col].astype('str')
    		ml = ml.iloc[0]
    		ml = ml.replace("'",'')
    		ml = np.float(ml)
    		trial_table.loc[i,col]=ml


    # change mags to negative
    task =get_gain_loss_from_filename(filename)
    if task=='loss':
    	trial_table['mag_left']= trial_table['mag_left']*-1.0
    	trial_table['mag_right'] = trial_table['mag_right']*-1.0
    	trial_table['mag_ambig'] = trial_table['mag_ambig']*-1.0
    	trial_table['mag_unambig']=trial_table['mag_unambig']*-1.0

    # this is for the combined model
    for i in range(num_trials):
        mag=trial_table.loc[trial_table['trial_number']==i,'mag_left'].as_matrix()
        if mag>0:
            trial_table.loc[trial_table['trial_number']==i,'gain_or_loss_trial'] = 'gain'
        elif mag<0:
            trial_table.loc[trial_table['trial_number']==i,'gain_or_loss_trial'] = 'loss'

    # no brainers #



    if task =='combined':
    	# identify switch trials;
    	gainloss = trial_table['gain_or_loss_trial'].as_matrix().copy()
    	gainloss[gainloss=='gain']=0
    	gainloss[gainloss=='loss']=1
    	gainloss = gainloss.astype('float')
    	gain_loss_switches = np.where(np.diff(gainloss)!=0)[0]+1
    	trial_table['gain_loss_switches']=0
    	trial_table.loc[gain_loss_switches,'gain_loss_switches']=1

    return(trial_table)



#########


def get_params_df(modelname,task,data_participants,results_folder,combined=False):
    # load a single subject

    files=glob.glob(results_folder+'*'+modelname+'.p') # added task
    #Tracer()()
    model_results = pickle.load(open(files[0],'rb'))
    params = list(model_results['params'].index)
    print(params)

    if combined==False:
        # get more manageable dataframe
        if task=='gain' or task=='loss':
            id_vars = ['MID','no_brainer_per_cor_'+task,'CameBackTwice','no_brainer_per_cor_gain','no_brainer_per_cor_loss',
                           'num_no_resp_'+task,'ntrials_'+task,'date','STAI_Trait']
        elif task=='shock':
            id_vars = ['MID','no_brainer_per_cor_'+task,'CameBackTwice','no_brainer_per_cor_gain','no_brainer_per_cor_loss',
                           'num_no_resp_'+task,'ntrials_'+task,'STAI_Trait','BDI']

    else:
        id_vars = ['MID','CameBackTwice','no_brainer_per_cor_gain','no_brainer_per_cor_loss',
                       'num_no_resp_'+task,'ntrials_'+task,'date','STAI_Trait','BDI',
                   'MASQ.AD','STAI_Trait_dep','PSWQ','MASQ.AA',modelname+'_pred_acc_'+task]#,'FA1','FA2','FA3']


    columns = [modelname+'_'+param+'_'+task for param in params]+id_vars

    df1 = data_participants[columns]
    df1 = pd.melt(df1,id_vars=id_vars)
    df1 = df1.rename(columns={'variable':'parameter','value':'beta'})
    replacedict ={}
    for param in params:
        replacedict[modelname+'_'+param+'_'+task]= param
    df1 = df1.replace(to_replace = replacedict)
    df1['task']=task

    #from IPython.core.debugger import Tracer
    #Tracer()()

    # change prob loss to be on the same side
    for index,row in df1.iterrows():
        #if 'prob_diff_rl_loss' in row.parameter:
        #    df1.loc[index,'beta']=df1.loc[index,'beta']*-1.0
        #if 'prob_diff_amb_loss' in row.parameter:
    #        df1.loc[index,'beta']=df1.loc[index,'beta']*-1.0
        #if 'sqrt' in row.parameter:
        #    df1.loc[index,'beta']=df1.loc[index,'beta']*-1.0
        if 'loss' in row.parameter:
            df1.loc[index,'outcome type']='loss'
        elif 'gain' in row.parameter:
            df1.loc[index,'outcome type']='gain'
        else:
            df1.loc[index,'outcome type']='shared'

    return(df1)



def remove_bad_participants(df,task):
    if task =='gain' or task =='loss':
        nbperf = df['no_brainer_per_cor_'+task].as_matrix()
        cbt = df['CameBackTwice'].as_matrix() # this is a indiactor the the second time a subject took the tasks
        timeouts = df['num_no_resp_'+task].as_matrix() # number of trials >8 sec
        numtrials =df['ntrials_'+task].as_matrix() # how many trials were recroded
        selector = (nbperf>0.7)&(cbt!=1)&(timeouts<10)&(numtrials>150)
    if task=='combined':
        nbperf_g = df['no_brainer_per_cor_gain'].as_matrix()
        nbperf_l = df['no_brainer_per_cor_loss'].as_matrix()
        #cbt = df['CameBackTwice'].as_matrix() # this is a indiactor the the second time a subject took the tasks
        timeouts = df['num_no_resp_'+task].as_matrix() # number of trials >8 sec
        numtrials =df['ntrials_'+task].as_matrix() # how many trials were recroded
        MID = df['MID'].as_matrix()
        selector = (nbperf_g>0.6)&(nbperf_l>0.6)&(timeouts<10)&(numtrials>280)&(MID!='A2KG59JUICJLP0') #has good no brainer perf but <0.5 model fit? Only subject so excluding rather than figuring out why.
    df = df.loc[selector,]
    return(df)
