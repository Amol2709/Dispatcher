# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django.http import JsonResponse
from django import template
from django.db import connection
from .models import SnowIncident
from .models import historical_data
from .models import incident_counts
from .models import shift_rota
from .models import Tickets
from django.db.models import Count
from . import ml_predict
from . import sectotime
from app.forms import PreferredTagForm, PreferredGroupForm
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from AGAccuracy import AgAccuracy
from TAGAccuracy import TagAccuracy


################################################################################################################################
#change
import tensorflow_hub as hub


#####################################################################################################################################


def dictfetchall(cursor): 
    "Returns all rows from a cursor as a dict" 
    desc = cursor.description 
    return [
            dict(zip([col[0] for col in desc], row)) 
            for row in cursor.fetchall() 
    ]

##########Home page, or the incident dashboard##############
@login_required(login_url="/login/")
def index(request):
    
    if request.method == 'POST':

        context = {}
        #context['segment'] = 'index'
        cursor = connection.cursor()

        selected_group = request.POST['group-select']
        print(selected_group)
        last_predicted = SnowIncident.objects.all().order_by('-sys_created_time').first()
        snowincident = SnowIncident.objects.filter(assigned_to='Could not predict the engineer!').order_by('-sys_created_time')[:5]
        snow = SnowIncident.objects.exclude(assigned_to='Could not predict the engineer!').order_by('-sys_created_time')[:5]

        #This section of code calculates the efficiency of the model.
        overall_inc_count = SnowIncident.objects.count()
        overall_pred_inc_count = SnowIncident.objects.exclude(assigned_to='Could not predict the engineer!').count()
        
        cursor.execute("select count(incident_id) as num from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1)")
        week_inc_count = cursor.fetchone()[0]
        cursor.execute("select count(incident_id) as num from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1) and assigned_to != 'Could not predict the engineer!'")
        week_pred_inc_count = cursor.fetchone()[0]

        cursor.execute("select count(incident_id) as num from project.snow_inc where sys_created_time >= curdate() and sys_created_time < curdate() + INTERVAL 1 DAY")
        day_inc_count = cursor.fetchone()[0]
        cursor.execute("select count(incident_id) as num from project.snow_inc where sys_created_time >= curdate() and sys_created_time < curdate() + INTERVAL 1 DAY and assigned_to != 'Could not predict the engineer!'")
        day_pred_inc_count = cursor.fetchone()[0]

        overall_efficiency = (overall_pred_inc_count/overall_inc_count)*100
        #To avoid zero division error in week and day counts i.e 0 divided by 0 causes exception
        if week_inc_count != 0:
            week_efficiency = (week_pred_inc_count/week_inc_count)*100
        else: week_efficiency = 0
        if day_inc_count != 0:    
            day_efficiency = (day_pred_inc_count/day_inc_count)*100
        else: day_efficiency = 0
        ####################################################################################################
        asg_group = last_predicted.assigned_group
        cat_tag = last_predicted.tags

        eng_list = historical_data.objects.raw("""SELECT 1 id, sec_to_time(avg(timestampdiff(second, sys_created_time, resolved_at_time))) resolution, resolved_by FROM project.historical_data where assigned_group = %s and tags = %s GROUP BY resolved_by ORDER BY resolution""", (asg_group, cat_tag))
        eng_availability = shift_rota.objects.raw("""SELECT 1 id, engineer_name, shift_type, assignment_group, SEC_TO_TIME(TIMESTAMPDIFF(second, now(), shift_end_time)) as availability from project.shift_rota where now() BETWEEN shift_start_time and shift_end_time and assignment_group = %s""",(selected_group,))
        total_group = shift_rota.objects.raw("""SELECT id, assignment_group from project.shift_rota group by assignment_group""")

        assignment_group_data = incident_counts.objects.raw("""SELECT 1 as id, count(*) as num, assignment_group from project.incident_counts where incident_state = 'In Progress' group by assignment_group""")

        assignment_group_list = []

        for value in assignment_group_data:
            assignment_group_list.append(value.assignment_group)

        assignment_group_count = {}
        for value in assignment_group_data:
            assignment_group_count.update({value.assignment_group: value.num})

        inc_count_values = list(map(lambda x: assignment_group_count.get(x), assignment_group_list))


        context['total_group'] = total_group
        context['snowincident'] = snowincident
        context['snow'] = snow
        context['last_predicted'] = last_predicted
        context['eng_list'] = eng_list
        context['eng_availability'] = eng_availability
        context['assignment_group_data'] = assignment_group_data
        context['assignment_group_count'] = assignment_group_count
        context['assignment_group_list'] = assignment_group_list
        context['inc_count_values'] = inc_count_values
        context['overall_inc_count'] = overall_inc_count
        context['overall_efficiency'] = overall_efficiency
        context['week_inc_count'] = week_inc_count
        context['week_efficiency'] = week_efficiency
        context['day_inc_count'] = day_inc_count
        context['day_efficiency'] = day_efficiency
        # html_template = loader.get_template( 'index.html' )
        # return HttpResponse(html_template.render(context, request))
        return render(request, 'index.html', context)
    
    else:
        context = {}
        context['segment'] = 'index'
        cursor = connection.cursor()

        last_predicted = SnowIncident.objects.all().order_by('-sys_created_time').first()
        snowincident = SnowIncident.objects.filter(assigned_to='Could not predict the engineer!').order_by('-sys_created_time')[:5]
        snow = SnowIncident.objects.exclude(assigned_to='Could not predict the engineer!').order_by('-sys_created_time')[:5]

        #This section of code claculates the efficiency of the model.
        overall_inc_count = SnowIncident.objects.count()
        overall_pred_inc_count = SnowIncident.objects.exclude(assigned_to='Could not predict the engineer!').count()
        
        cursor.execute("select count(incident_id) as num from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1)")
        week_inc_count = cursor.fetchone()[0]
        cursor.execute("select count(incident_id) as num from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1) and assigned_to != 'Could not predict the engineer!'")
        week_pred_inc_count = cursor.fetchone()[0]

        cursor.execute("select count(incident_id) as num from project.snow_inc where sys_created_time >= curdate() and sys_created_time < curdate() + INTERVAL 1 DAY")
        day_inc_count = cursor.fetchone()[0]
        cursor.execute("select count(incident_id) as num from project.snow_inc where sys_created_time >= curdate() and sys_created_time < curdate() + INTERVAL 1 DAY and assigned_to != 'Could not predict the engineer!'")
        day_pred_inc_count = cursor.fetchone()[0]

        overall_efficiency = (overall_pred_inc_count/overall_inc_count)*100
        #To avoid zero division error in week and day counts i.e 0 divided by 0 causes exception
        if week_inc_count != 0:
            week_efficiency = (week_pred_inc_count/week_inc_count)*100
        else: week_efficiency = 0
        if day_inc_count != 0:    
            day_efficiency = (day_pred_inc_count/day_inc_count)*100
        else: day_efficiency = 0
        ########################################################################################################
        asg_group = last_predicted.assigned_group
        cat_tag = last_predicted.tags

        eng_list = historical_data.objects.raw("""SELECT 1 id, sec_to_time(avg(timestampdiff(second, sys_created_time, resolved_at_time))) resolution, resolved_by FROM project.historical_data where assigned_group = %s and tags = %s GROUP BY resolved_by ORDER BY resolution""", (asg_group, cat_tag))
        eng_availability = shift_rota.objects.raw("""SELECT 1 id, engineer_name, shift_type, assignment_group, SEC_TO_TIME(TIMESTAMPDIFF(second, now(), shift_end_time)) as availability from project.shift_rota where now() BETWEEN shift_start_time and shift_end_time""")
        total_group = shift_rota.objects.raw("""SELECT id, assignment_group from project.shift_rota group by assignment_group""")

        assignment_group_data = incident_counts.objects.raw("""SELECT 1 as id, count(*) as num, assignment_group from project.incident_counts where incident_state = 'In Progress' group by assignment_group""")

        assignment_group_list = []

        for value in assignment_group_data:
            assignment_group_list.append(value.assignment_group)

        assignment_group_count = {}
        for value in assignment_group_data:
            assignment_group_count.update({value.assignment_group: value.num})

        inc_count_values = list(map(lambda x: assignment_group_count.get(x), assignment_group_list))

        context['total_group'] = total_group
        context['snowincident'] = snowincident
        context['snow'] = snow
        context['last_predicted'] = last_predicted
        context['eng_list'] = eng_list
        context['eng_availability'] = eng_availability
        context['assignment_group_data'] = assignment_group_data
        context['assignment_group_count'] = assignment_group_count
        context['assignment_group_list'] = assignment_group_list
        context['inc_count_values'] = inc_count_values
        context['overall_inc_count'] = overall_inc_count
        context['overall_efficiency'] = overall_efficiency
        context['week_inc_count'] = week_inc_count
        context['week_efficiency'] = week_efficiency
        context['day_inc_count'] = day_inc_count
        context['day_efficiency'] = day_efficiency

        # html_template = loader.get_template( 'index.html' )
        # return HttpResponse(html_template.render(context, request))
        return render(request, 'index.html', context)


##########As the name suggests, This is the logic for the Engineer Page Details/Description##############
@login_required(login_url="/login/")
def engineer_ticket_details(request, engineer_name):

    context = {}
    try:
        engineer_ticket_data = incident_counts.objects.raw("""SELECT 1 as id, count(*) as num, engineer_name from project.incident_counts where incident_state = 'In Progress' and engineer_name = %s group by engineer_name""", (engineer_name, ))

        engineer_ticket_values = incident_counts.objects.raw("""SELECT 1 as id, incident_id, engineer_name from project.incident_counts where incident_state = 'In Progress' and engineer_name = %s""", (engineer_name,))

        engineer_tickets_list = []

        for data in engineer_ticket_values:
            engineer_tickets_list.append(data.incident_id)

        context['engineer_name'] = engineer_name
        context['engineer_tickets_list'] = engineer_tickets_list
        context['engineer_ticket_values'] = engineer_ticket_values
        context['engineer_ticket_data'] = engineer_ticket_data

        return render(request, 'engineer_ticket_details.html', context)
    
    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########Logic for the page that displays all of the assignment group details##############
@login_required(login_url="/login/")
def assignment_groups_tickets(request, assignment_group):
    
    context = {}
    try:

        all_assignment_groups = incident_counts.objects.raw("""SELECT 1 as id, incident_id, assignment_group from project.incident_counts where incident_state = 'In Progress' and assignment_group = %s""",(assignment_group, ))

        asg_group_tickets = []

        for data in all_assignment_groups:
            asg_group_tickets.append(data.incident_id)

        context['assignment_group'] = assignment_group
        context['all_assignment_groups'] = all_assignment_groups
        context['asg_group_tickets'] = asg_group_tickets

        return render(request, 'assignment_groups_tickets.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########Table list view or the page that displays all incident details in our DB##############
@login_required(login_url="/login/")
def ui_table(request):
    context = {}    
    try:
        
        context['segment'] = 'ui-tables'

        snowincident = SnowIncident.objects.all().order_by('-sys_created_time')
        #context = {'snowincident': snowincident}
        context['snowincident'] = snowincident
        return render(request, 'ui-tables.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########User Profile page##############
@login_required(login_url="/login/")
def page_user(request):
    context = {}    
    try:
        
        context['segment'] = 'page-user'

        return render(request, 'page-user.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########Notifications page##############
@login_required(login_url="/login/")
def ui_notifications(request):
    context = {}    
    try:
        
        context['segment'] = 'ui-notifications'

        return render(request, 'ui-notifications.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########Incident details input page for quick testing##############
@login_required(login_url="/login/")
def ui_new_incidents(request):
    context = {}    
    try:
        
        context['segment'] = 'ui-new-incidents'
        #context['snowincident'] = SnowIncident.objects.all()
        return render(request, 'ui-new-incidents.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########Performance Charts Page##############
@login_required(login_url="/login/")
def ui_chart(request):
    context = {}    
    try:
        labels_priority = []
        data_priority = []

        context['segment'] = 'ui-chart'
        
        total_priority = SnowIncident.objects.all().values('priority').annotate(total=Count('priority'))
        for query in total_priority:
            labels_priority.append(query.get('priority'))
            data_priority.append(query.get('total'))

        context['data_priority'] = data_priority
        context['labels_priority'] = labels_priority

        return render(request, 'ui-chart.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


##########Priority chart##############
@login_required(login_url="/login/")
def total_priority_chart(request):   
        
    labels_priority = []
    data_priority = []
            
    total_priority = SnowIncident.objects.all().values('priority').annotate(total=Count('priority'))
    for query in total_priority:
        labels_priority.append(query.get('priority'))
        data_priority.append(query.get('total'))


    return JsonResponse(data={
        'labels_priority': labels_priority,
        'data_priority': data_priority,
    })

##########Category Tags chart in Performance Graphs page##############
@login_required(login_url="/login/")
def category_tags_chart(request):   
        
    labels_tags = []
    data_tags = []
            
    total_tags = SnowIncident.objects.all().values('tags').annotate(total=Count('tags'))
    for query in total_tags:
        labels_tags.append(query.get('tags'))
        data_tags.append(query.get('total'))


    return JsonResponse(data={
        'labels_tags': labels_tags,
        'data_tags': data_tags,
    })

##########Assignment Group chart in Performance Graphs page##############
@login_required(login_url="/login/")
def asg_grp_chart(request):   
        
    labels_asg = []
    data_asg = []
            
    total_asg = SnowIncident.objects.all().values('assigned_group').annotate(total=Count('assigned_group'))
    for query in total_asg:
        labels_asg.append(query.get('assigned_group'))
        data_asg.append(query.get('total'))


    return JsonResponse(data={
        'labels_asg': labels_asg,
        'data_asg': data_asg,
    })

##########Support Engineer chart in Performace Graphs page##############
@login_required(login_url="/login/")
def sup_eng_chart(request):   
        
    labels_eng = []
    data_eng = []
            
    total_eng = SnowIncident.objects.all().values('assigned_to').annotate(total=Count('assigned_to'))
    for query in total_eng:
        labels_eng.append(query.get('assigned_to'))
        data_eng.append(query.get('total'))


    return JsonResponse(data={
        'labels_eng': labels_eng,
        'data_eng': data_eng,
    })

##########Assignment group confidence level chart in home page##############
@login_required(login_url="/login/")
def asg_con_chart(request):   
        
    labels_asg = []
    data_asg = []
            
    total_asg = SnowIncident.objects.values('incident_id', 'pred_confidence_asg').order_by('-sys_created_time')[:10]
    for query in total_asg:
        labels_asg.append(query.get('incident_id'))
        data_asg.append(query.get('pred_confidence_asg'))


    return JsonResponse(data={
        'labels_asg': labels_asg,
        'data_asg': data_asg,
    })

##########Categorical tag confidence level chart in home page##############
@login_required(login_url="/login/")
def tag_con_chart(request):   
        
    labels_tag = []
    data_tag = []
            
    total_tag = SnowIncident.objects.values('incident_id', 'pred_confidence_tag').order_by('-sys_created_time')[:10]
    for query in total_tag:
        labels_tag.append(query.get('incident_id'))
        data_tag.append(query.get('pred_confidence_tag'))


    return JsonResponse(data={
        'labels_tag': labels_tag,
        'data_tag': data_tag,
    })

##########Pie Chart representing the overall-count-efficiency chart in home(index) page.#############
@login_required(login_url="/login/")
def overall_inc_count_asg_chart(request):   

    data_overall_inc_count_asg = []
    labels_overall_inc_count_asg = ['Predicted Incidents', 'Not predicted incidents']

    overall_not_pred_inc_count = SnowIncident.objects.filter(assigned_to='Could not predict the engineer!').count()
    overall_pred_inc_count = SnowIncident.objects.exclude(assigned_to='Could not predict the engineer!').count()
    
    data_overall_inc_count_asg.append(overall_pred_inc_count)
    data_overall_inc_count_asg.append(overall_not_pred_inc_count)

    return JsonResponse(data={
        'labels_overall_inc_count_asg': labels_overall_inc_count_asg,
        'data_overall_inc_count_asg': data_overall_inc_count_asg,
    })


##########Pie Chart representing the week-count-efficiency chart in home(index) page.#############
@login_required(login_url="/login/")
def week_inc_count_asg_chart(request):   

    data_week_inc_count_asg = []
    labels_week_inc_count_asg = ['Predicted Incidents', 'Not predicted incidents']

    cursor = connection.cursor()

    cursor.execute("select count(incident_id) as num from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1) and assigned_to = 'Could not predict the engineer!'") 
    week_not_pred_inc_count = cursor.fetchone()[0]
    cursor.execute("select count(incident_id) as num from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1) and assigned_to != 'Could not predict the engineer!'")
    week_pred_inc_count = cursor.fetchone()[0]

    data_week_inc_count_asg.append(week_pred_inc_count)
    data_week_inc_count_asg.append(week_not_pred_inc_count)

    return JsonResponse(data={
        'labels_week_inc_count_asg': labels_week_inc_count_asg,
        'data_week_inc_count_asg': data_week_inc_count_asg,
    })

##########Pie Chart representing the day-count-efficiency chart in home(index) page.#############
@login_required(login_url="/login/")
def day_inc_count_asg_chart(request):   

    data_day_inc_count_asg = []
    labels_day_inc_count_asg = ['Predicted Incidents', 'Not predicted incidents']

    cursor = connection.cursor()

    cursor.execute("select count(incident_id) as num from project.snow_inc where sys_created_time >= curdate() and sys_created_time < curdate() + INTERVAL 1 DAY and assigned_to = 'Could not predict the engineer!'")
    day_not_pred_inc_count = cursor.fetchone()[0]
    cursor.execute("select count(incident_id) as num from project.snow_inc where sys_created_time >= curdate() and sys_created_time < curdate() + INTERVAL 1 DAY and assigned_to != 'Could not predict the engineer!'")
    day_pred_inc_count = cursor.fetchone()[0]

    data_day_inc_count_asg.append(day_pred_inc_count)
    data_day_inc_count_asg.append(day_not_pred_inc_count)

    return JsonResponse(data={
        'labels_day_inc_count_asg': labels_day_inc_count_asg,
        'data_day_inc_count_asg': data_day_inc_count_asg,
    })

####Model accuracy vs overall model efficiency.
@login_required(login_url="/login/")
def acc_eff_model_asg(request):   

    data_acc_model = []
    data_eff_model = []
    labels_date = []

    cursor = connection.cursor()

    cursor.execute("select distinct(date(sys_created_time)) from project.snow_inc where yearweek(sys_created_time, 1) = yearweek(curdate(),1) order by sys_created_time")
    temp_label = cursor.fetchall()
    for obj in temp_label:
        labels_date.append(obj)

    for t in labels_date:
        cursor.execute("select asg_model_accuracy from project.snow_inc where date(sys_created_time) = %s order by sys_created_time desc limit 1",(t,))
        temp_acc = cursor.fetchone()
        cursor.execute("select asg_model_eff from project.snow_inc where date(sys_created_time) = %s order by sys_created_time desc limit 1",(t,))
        temp_eff = cursor.fetchone()
        for i in temp_acc:
            data_acc_model.append(i)
        
        for j in temp_eff:
            data_eff_model.append(j)
            
    

    return JsonResponse(data={
        'labels_date': labels_date,
        'data_acc_model': data_acc_model,
        'data_eff_model': data_eff_model
    })


############# Shows quick result. This page is for quick testing purposes.###########
@login_required(login_url="/login/")
def ui_result(request):

    context = {}
    try:

        if request.method == 'POST':
            incident_id = request.POST.get('incident_id').upper()
            short_desc = request.POST.get('short_desc')
            long_desc = request.POST.get('long_desc')
            priority = request.POST.get('priority')

            if incident_id and short_desc and long_desc and priority:
                snow = SnowIncident()
                inc_cnt = incident_counts()

                snow.incident_id = incident_id
                snow.short_description = short_desc
                snow.long_description = long_desc
                snow.priority = priority

                ######Getting the output of prediction model.
                result = ml_predict.getPredictions(incident_id ,short_desc, long_desc, priority)

                ######Saving the incident details in the snow_inc table.
                snow.assigned_group = result.get('group')
                snow.tags = result.get('tag')
                snow.assigned_to = result.get('eng')
                snow.pred_confidence_asg = result.get('asg_con')
                snow.pred_confidence_tag = result.get('tag_con')
                snow.pred_time_asg = result.get('pred_time_asg')
                snow.pred_time_tag = result.get('pred_time_tag')
                snow.average_resolution_time = result.get('res_time')

                #Model metrics details -added on 18-02-2021

                ################################################################################################################################
                #Acc_ag = AgAccuracy.AgAccuracy('assignment_group')
                #Acc_tag = TagAccuracy('cleaned_tags')


                #----db connection#-------------------------------------------------
                status = 'assignment_group'
                db_connection = sql.connect(host='localhost', database='my-db', user='root', password='root@123root@123',port=3307)
                df=pd.read_sql('SELECT * FROM cleanviacom', con=db_connection)
                print("*"*100)
                #-------------------------------------------------------------------
                df = df[["desc",status]]
                A=dict(df[status].value_counts())
                x=list(A.keys()) # number of new tags
                y= list(A.values())
                le = preprocessing.LabelEncoder()
                le.fit(x)
                Labels=le.transform(list(le.classes_))
                a =Labels.copy()
                b = np.zeros((a.size, a.max()+1))
                b[np.arange(a.size),a] = 1
                train_label = np.zeros((df.count()[0],len(Labels)))
                #print(list(self.le.classes_))
                for i in range(0,df.count()[0]):
                    #print(self.df[self.status][i])
                    Index=list(le.classes_).index(df[status][i])
                    train_label[i,:] = b[Index,:]

                #self.training_desc = list(self.df['desc'])
                clean_train_desc = list(df['desc'])
                
                print("*"*100)
                print('Data Preparing Finished')
                print("*"*100)
                print('Loading Old Assignnment Group Model From Disk .... ............')

                print("*"*100)
                #print('Building New Model..............')

                #######################################################################################################################################################
                #change
                model = tf.keras.models.load_model('Assignmentgroup_model.h5',custom_objects={'KerasLayer': hub.KerasLayer})


                ##########################################################################################################################################################
                
                print('model Loaded')
                model.summary()
                print("*"*100)
               
                #self.num_epochs = epoch
                #self.name=name
                
                # vocab_size = 1500
                # embedding_dim = 32
                # max_length = 150
                # trunc_type='post'
                # oov_tok = "<OOV>"
                # tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
                # tokenizer.fit_on_texts(clean_train_desc)

                # word_index = tokenizer.word_index
                # clean_train_sequences = tokenizer.texts_to_sequences(clean_train_desc)
                # clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)
                print('start evaluation..............................................')
                #history = self.model.fit(self.clean_train_padded, self.train_label, epochs=self.num_epochs)

                #############################################################################################################################################################################
                #change
                scores = model.evaluate(np.array(clean_train_desc),train_label)



                ############################################################################################################################################################
                print('Accuracy of assignment_group: {}'.format(scores[1]*100))
                snow.asg_model_accuracy = scores[1]*100

















                status = 'cleaned_tags'
                db_connection = sql.connect(host='localhost', database='my-db', user='root', password='root@123root@123',port=3307)
                df=pd.read_sql('SELECT * FROM cleanviacom', con=db_connection)
                print("*"*100)
                #-------------------------------------------------------------------
                df = df[["desc",status]]
                A=dict(df[status].value_counts())
                x=list(A.keys()) # number of new tags
                y= list(A.values())
                le = preprocessing.LabelEncoder()
                le.fit(x)
                Labels=le.transform(list(le.classes_))
                a =Labels.copy()
                b = np.zeros((a.size, a.max()+1))
                b[np.arange(a.size),a] = 1
                train_label = np.zeros((df.count()[0],len(Labels)))
                #print(list(self.le.classes_))
                for i in range(0,df.count()[0]):
                    #print(self.df[self.status][i])
                    Index=list(le.classes_).index(df[status][i])
                    train_label[i,:] = b[Index,:]

                #self.training_desc = list(self.df['desc'])
                clean_train_desc = list(df['desc'])
                
                print("*"*100)
                print('Data Preparing Finished')
                print("*"*100)
                print('Loading Old ML Tag Model From Disk .... ............')

                print("*"*100)
                #print('Building New Model..............')

                ###############################################################################################################################################################################

                #change
                model = tf.keras.models.load_model('ML_TAGmodel.h5',custom_objects={'KerasLayer': hub.KerasLayer})


                #####################################################################################################################################################
                
                print('model Loaded')
                model.summary()
                print("*"*100)
               
                #self.num_epochs = epoch
                #self.name=name
                
                # vocab_size = 1500
                # embedding_dim = 32
                # max_length = 150
                # trunc_type='post'
                # oov_tok = "<OOV>"
                # tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
                # tokenizer.fit_on_texts(clean_train_desc)

                # word_index = tokenizer.word_index
                # clean_train_sequences = tokenizer.texts_to_sequences(clean_train_desc)
                # clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)
                print('start evaluation..............................................')
                #history = self.model.fit(self.clean_train_padded, self.train_label, epochs=self.num_epochs)
                ########################################################################################################################################################



                #change
                scores = model.evaluate(np.array(clean_train_desc),train_label)


                ##########################################################################################################################################################
                print('Accuracy of ML_TAGmodel: {}'.format(scores[1]*100))
                snow.tag_model_accuracy = scores[1]*100
                print('Accuracy of Tag: {}'.format(scores[1]*100))
                ###################################################################################################################################
                
                #####  ----------------------------------********************************************--------------------------#################
                ####################################################################################################################################3
                snow.asg_model_version = 'v1.0'
                snow.tag_model_version = 'v1.0'

                snow.ml_processed = True
                snow.save()

                cursor = connection.cursor()
                #Logic to get model efficiency and update every model
                overall_inc_count = SnowIncident.objects.all().count()
                overall_pred_inc_count = SnowIncident.objects.exclude(assigned_to='Could not predict the engineer!').count()
                eff = (overall_pred_inc_count/overall_inc_count)*100

                cursor.execute("update project.snow_inc set asg_model_eff = %s, tag_model_eff = %s where incident_id = %s",(eff, eff, incident_id))

                result['res_time'] = sectotime.getTime(result.get('res_time'))

                #####Saving new incidents to the incident_counts table to increment the value of count.
                #####P3, P4 and P5 incidents are considering the count of incidents when predicting the best fit engineer.
                inc_cnt.incident_id = incident_id
                inc_cnt.incident_state = "In Progress"
                inc_cnt.engineer_name = result.get('eng')
                inc_cnt.assignment_group = result.get('group')
                inc_cnt.save()
                
                group = result.get('group')
                tag = result.get('tag')


                ######Raw SQL queries for extracting relevant info for display.
                avail_eng = shift_rota.objects.raw("""SELECT 1 id, engineer_name, shift_start_time, shift_end_time, TIMESTAMPDIFF(second, now(), shift_end_time) as availability from project.shift_rota where now() BETWEEN shift_start_time and shift_end_time""")
                list_of_eng = historical_data.objects.raw("""SELECT 1 id, avg(timestampdiff(second, sys_created_time, resolved_at_time)) resolution, resolved_by FROM project.historical_data where assigned_group = %s and tags = %s GROUP BY resolved_by ORDER BY resolution""", (group, tag))
                count_of_inc = incident_counts.objects.raw("""SELECT 1 as id, count(*) as num, engineer_name from project.incident_counts where assignment_group = %s and incident_state = 'In Progress' group by engineer_name""",(group,))

                ######list that contains all the available engineers for the combination of the predicted values of assignment group and tags.
                eng_name_list = []
                for f in list_of_eng:
                    eng_name_list.append(f.resolved_by)

                ######Count of incidents in progress saved in a dictionary along with engineer name.
                tmp_count_dict = {}
                for d in count_of_inc:
                    tmp_count_dict.update({d.engineer_name: d.num})

                ######Mapping the eng_name_list with the count of in progress incidents. This is to display in the engineering matrix section.
                inc_count_list = list(map(lambda x: tmp_count_dict.get(x), eng_name_list))

                ######list of engineers currently available on shift
                avail_tmp_list = []
                for z in avail_eng:
                    avail_tmp_list.append(z.engineer_name)

                #####list of available engineers from the total engineers list -> mapping True or False.
                #####This is for the availability tick or cross display.
                avail_bool = []
                for j in list_of_eng:
                    if(j.resolved_by in avail_tmp_list):
                        avail_bool.append(True)
                    else: avail_bool.append(False)

                ######List of the resolution time of each engineer. Saved on the list and sent to conversion to our required format.
                res_time_list = []
                for x in list_of_eng:
                    val = sectotime.getTime(x.resolution)
                    res_time_list.append(val)

                # ranking_count = []
                # for i in range(1,len(res_time_list)+1):
                #     ranking_count.append(i)
                
                #output = {}

                result['resolution_time'] = res_time_list
                #output['ranking'] = ranking_count
                #output['list_of_eng'] = list_of_eng
                result['avail_bool'] = avail_bool
                result['inc_bucket'] = inc_count_list

                return render(request, 'ui-result.html', {'result': result, 'incident_id': incident_id, 'avail_eng': avail_eng, 'list_of_eng': list_of_eng})
        else:
            return render(request, 'page-blank.html')

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


#########Logic for the page that displays all of the incident details###########
@login_required(login_url="/login/")
def inc_analyze(request, incident_id):

    context = {}
    try:

        inc_data = SnowIncident.objects.raw("""SELECT * from project.snow_inc where incident_id = %s """, (incident_id,))
        for obj in inc_data:

            group = obj.assigned_group
            tag = obj.tags
            resolution_time = sectotime.getTime(obj.average_resolution_time)
            sup_eng = obj.assigned_to
            priority = int(obj.priority)


        ######Raw SQL queries for extracting relevant info for display.
        avail_eng = shift_rota.objects.raw("""SELECT 1 id, engineer_name, shift_start_time, shift_end_time, TIMESTAMPDIFF(second, now(), shift_end_time) as availability from project.shift_rota where now() BETWEEN shift_start_time and shift_end_time""")
        list_of_eng = historical_data.objects.raw("""SELECT 1 id, avg(timestampdiff(second, sys_created_time, resolved_at_time)) resolution, resolved_by FROM project.historical_data where assigned_group = %s and tags = %s GROUP BY resolved_by ORDER BY resolution""", (group, tag))
        count_of_inc = incident_counts.objects.raw("""SELECT 1 as id, count(*) as num, engineer_name from project.incident_counts where assignment_group = %s and incident_state = 'In Progress' group by engineer_name""",(group,))

        ######list that contains all the available engineers for the combination of the predicted values of assignment group and tags.
        eng_name_list = []
        for f in list_of_eng:
            eng_name_list.append(f.resolved_by)

        ######Count of incidents in progress saved in a dictionary along with engineer name.
        tmp_count_dict = {}
        for d in count_of_inc:
            tmp_count_dict.update({d.engineer_name: d.num})

        quick_count = tmp_count_dict.get(sup_eng)

        ######Mapping the eng_name_list with the count of in progress incidents. This is to display in the engineering matrix section.
        inc_count_list = list(map(lambda x: tmp_count_dict.get(x), eng_name_list))

        ######list of engineers currently available on shift
        avail_tmp_list = []
        for z in avail_eng:
            avail_tmp_list.append(z.engineer_name)

        #####list of available engineers from the total engineers list -> mapping True or False.
        #####This is for the availability tick or cross display.
        avail_bool = []
        for j in list_of_eng:
            if(j.resolved_by in avail_tmp_list):
                avail_bool.append(True)
            else:
                avail_bool.append(False)

        ######List of the resolution time of each engineer. Saved on the list and sent to conversion to our required format.
        res_time_list = []
        for x in list_of_eng:
            val = sectotime.getTime(x.resolution)
            res_time_list.append(val)


        context['inc_data'] = inc_data
        context['incident_id'] = incident_id
        context['priority'] = priority
        context['sup_eng'] = sup_eng
        context['quick_count'] = quick_count
        context['res_time_list'] = res_time_list
        context['avail_bool'] = avail_bool
        context['inc_count_list'] = inc_count_list
        context['eng_name_list'] = eng_name_list
        context['resolution_time'] = resolution_time

        return render(request, 'inc-analyze.html', context)

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def admin_feedback(request):
    group_form = PreferredGroupForm()
    tag_form = PreferredTagForm()
    service_desk_tickets = Tickets.objects.filter(assigned_group='Service Desk')

    if request.method == 'POST':
        group_form = PreferredGroupForm(request.POST)
        if group_form.is_valid():
            group_form.save()

    if request.method == 'POST':
        tag_form = PreferredTagForm(request.POST)
        if tag_form.is_valid():
            tag_form.save()

    page = request.GET.get('page')
    paginator = Paginator(service_desk_tickets, 1)

    try:
        service_desk_tickets = paginator.page(page)
    except PageNotAnInteger:
        service_desk_tickets = paginator.page(1)
    except EmptyPage:
        service_desk_tickets = paginator.page(paginator.num_pages)

    context = {}

    context['group_form'] = group_form
    context['tag_form'] = tag_form
    context['service_desk_tickets'] = service_desk_tickets
    context['paginator'] = paginator
    context['page'] = page

    return render(request, 'admin_feedback.html', context)
