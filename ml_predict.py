from .models import SnowIncident
from .models import historical_data
from .models import incident_counts
from .models import shift_rota
import pandas as pd
import pickle
import time
import operator
import numpy as np


from predict import Predict




#eng_tmp_data = pd.read_excel(r'C:\Users\10644499\djangoprojects\Intelligent-Incident-Dispatcher\app\Resolved Data.xlsx', engine='openpyxl')

#encoding_dict_asg = {0: 'dba-l2', 1: 'infosec-l2', 2: 'infrastructuremonitoring', 3: 'linuxengineering-l2', 4: 'media services-l2', 5: 'p-eaiadmin', 6: 'rg infrastructure - tsa', 7: 'rg service desk', 8: 'windowsengineering-l2'}
#encoding_dict_tag = {0: 'account creation/reset request', 1: 'account termination request', 2: 'ad extension request', 3: 'alias job stuck', 4: 'alias queues cross threshold rise', 5: 'application termination request', 6: 'catapult jobs time breach', 7: 'catapult transform jobs alert', 8: 'crowdstrike alert-custom intelligence', 9: 'crowdstrike alert-exploit mitigation', 10: 'crowdstrike alert: machine learning (adware/pup)', 11: 'crowdstrike alert: machine learning (sensor-based ml)', 12: 'ctp failed jobs alert', 13: 'ctp processing time exceeded job alert', 14: 'database down', 15: 'eaiprd hst alert', 16: 'high cpu utilization', 17: 'host not reporting', 18: 'memory threshold breach', 19: 'notyetsubmitted deliver job time breach', 20: 'server load breach', 21: 'unchanged priority queue', 22: 'zoom related issue'}

def getPredictions(incident_no, short_desc, long_desc, priority):
    
    #asg_model = pickle.load(open("assgn_group_model.sav", "rb"))
    #tag_model = pickle.load(open("tag_model.sav", "rb"))
    #tf = pickle.load(open("tf_idf.sav", "rb"))

    concat_desc = str(" ".join([short_desc, long_desc]))
    final_desc = concat_desc.lower()
    print('-'*100)
    print(final_desc)
    print('-'*100)
    final_desc = [final_desc]


    #################################################################################################################################################

    #change----------------assuming final_desc is string of list like ['mmamaj asjssns jassjsmsswwmw']


    import re
    from bs4 import BeautifulSoup
    from tqdm import tqdm
    stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
                'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
                'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
                've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
                "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
                "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
                'won', "won't", 'wouldn', "wouldn't"])




    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase


    desc = []
    count = 0
    # tqdm is for printing the status bar
    for sentance in tqdm(final_desc):
        
        try:
            
            sentance = re.sub(r"http\S+", "", sentance)
            sentance = BeautifulSoup(sentance, "html.parser").get_text()
            sentance = decontracted(sentance)
            sentance = re.sub("\S*\d\S*", "", sentance).strip()
            sentance = re.sub('[^A-Za-z]+', ' ', sentance)
            # https://gist.github.com/sebleier/554280
            sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
            desc.append(sentance.strip())
        except:
            print('Exception Ocuured During Cleaning of data')
            #print(sentance)
            #print(count)
        else:
            count= count+1

    final_desc = desc.copy()



    #########################################################################################################################################################
    #test_desc = tf.transform(final_desc)

    #model-1: assignment group prediction.
    #start_time_asg = time.time()
    #pred_prob_asg = asg_model.predict_proba(test_desc)

    
    #Calculates the prediction confidence.
    #pred_confidence_asg = (max(pred_prob_asg[0])*100)
    #print(pred_confidence_asg)
    #pred_col_asg_len = np.shape(pred_prob_asg)[1]
    
    #filtering logic. Only takes the class probability that is greater than our threshold.
    #pred_prob_class_asg = [999999 if sum(p<0.70)==pred_col_asg_len else asg_model.classes_[np.argmax(p)] for p in pred_prob_asg]
    #print("pred_prob_class_asg:", pred_prob_class_asg)
    #pred_asg = list(map(lambda x: encoding_dict_asg.get(x), pred_prob_class_asg))
    #pred_time_asg = (time.time() - start_time_asg)
    #print("prediction time: ", pred_time_asg)

    #------#########################################################################################################--------------------------#_-----------------------------------------------------------------------------------------
    start_time_asg = time.time()
    pred_obj_1 = Predict('assignment_group')
    predicted_ag,percent_ag = pred_obj_1.ModelPredictionAG(final_desc)# assuming final_desc is string
    pred_time_asg = (time.time() - start_time_asg)
    #-----############################################-##############################################################---------------------------------------------------------------------------------------------------------------





    #model-2: categorical tags prediction.

    ###########################################################################################################
    start_time_tag = time.time()
    pred_obj_2 = Predict('cleaned_tags')
    predicted_tag,percent_tag=pred_obj_2.ModelPredictionTAG(final_desc)
    pred_time_tag = (time.time() - start_time_tag)
    ###############################################################################################################

    #start_time_tag = time.time()
    #pred_prob_tag = tag_model.predict_proba(test_desc)
    
    #Calculates the prediction confidence.
    #pred_confidence_tag = (max(pred_prob_tag[0])*100)
    #print(pred_confidence_tag)
    #pred_col_tag_len = np.shape(pred_prob_tag)[1]
    
    #filtering logic. Only takes the class probability that is greater than our threshold.
    #pred_prob_class_tag = [999999 if sum(p<0.70)==pred_col_tag_len else tag_model.classes_[np.argmax(p)] for p in pred_prob_tag]
    #print("pred_prob_class_tag:", pred_prob_class_tag)
    #pred_tag = list(map(lambda x: encoding_dict_tag.get(x), pred_prob_class_tag))
    #print(pred_tag)
    #pred_time_tag = (time.time() - start_time_tag)
    #print("prediction time: ", pred_time_tag)


    #--------------------------------------------------------------------------------------------------------------------------#
    if(percent_ag <=50):
        engineer = "Could not predict the engineer!"
        asg_grp = "Service Desk"
        if(percent_tag <=50):
            result = {'group': asg_grp, 'tag': 'Could not predict Categorical Tag', 'eng': engineer, 'asg_con': percent_ag, 'tag_con': percent_tag, 'pred_time_asg': pred_time_asg, 'pred_time_tag': pred_time_tag, 'res_time': 0}
        else: 
            result = {'group': asg_grp, 'tag': predicted_tag, 'eng': engineer, 'asg_con': percent_ag, 'tag_con': percent_tag, 'pred_time_asg': pred_time_asg, 'pred_time_tag': pred_time_tag, 'res_time': 0}
        
        return result
    
    else:
        #Engineer allotment.
        eng_total_dict = {}
        available_eng_dict = {}
        eng_total_count = {}
        count_dict = {}
        

        ############Total list of all support engineers for the combination.
        eng_list = historical_data.objects.raw("""SELECT 1 id, avg(timestampdiff(second, sys_created_time, resolved_at_time)) resolution, resolved_by FROM project.historical_data where assigned_group = %s and tags = %s GROUP BY resolved_by ORDER BY resolution""", (pred_asg[0], pred_tag[0]))
        for x in eng_list:
            eng_total_dict.update({x.resolved_by : x.resolution})
            
            
        
        eng_total_list = eng_total_dict.keys()
        #sorted_check_dict = dict(sorted(check_dict.items(), key=operator.itemgetter(1)))        
        ############All available engineers on shift.
        available_eng = shift_rota.objects.raw("""SELECT 1 id, engineer_name, shift_start_time, shift_end_time, TIMESTAMPDIFF(minute, now(), shift_end_time) as availability from project.shift_rota where now() BETWEEN shift_start_time and shift_end_time and assignment_group = %s""",(pred_asg[0],))
        
        available_eng_list = []
        for obj in available_eng:
            available_eng_list.append(obj.engineer_name)
        #print(available_eng_list)

        #Removal of non available engineers in the dictionary.
        for k in eng_total_list:
            if k in available_eng_list:
                available_eng_dict.update({k : eng_total_dict.get(k)})

        ##########This list contains only the engineers who can handle ticket based(historical_data) on the combination of predicted values
        ##########from the available shift members.
        considered_available_eng_list = available_eng_dict.keys()
        
        ##########Engineer In-Progress Incident count.
        inc_count = incident_counts.objects.raw("""SELECT 1 as id, count(*) as num, engineer_name from project.incident_counts where assignment_group = %s and incident_state = 'In Progress' group by engineer_name""",(pred_asg[0],))
        if (priority == '0' or priority == '1' or priority == '2'):
            engineer = min(available_eng_dict, key=available_eng_dict.get, default='EMPTY')
            if engineer == 'EMPTY':
                res_time = 0
            else: res_time = available_eng_dict.get(engineer)
            print(res_time)

        else:
            for t in inc_count:
                eng_total_count.update({t.engineer_name : t.num})
            
            eng_total_count_list = eng_total_count.keys()

            for j in eng_total_count_list:
                if j in considered_available_eng_list:
                    count_dict.update({j : eng_total_count.get(j)})

            engineer = min(count_dict, key=count_dict.get, default='EMPTY')
            if engineer == 'EMPTY':
                res_time = 0
            else: res_time = available_eng_dict.get(engineer)
            print(res_time)


        #--------------------------------------------------------------------------------------------------------------------------------#
            
        

        result = {'group': predicted_ag, 'tag': predicted_tag, 'eng': engineer, 'asg_con': percent_ag, 'tag_con': percent_tag, 'pred_time_asg': pred_time_asg, 'pred_time_tag': pred_time_tag, 'res_time': res_time, 'eng_list': available_eng_dict}
        return result