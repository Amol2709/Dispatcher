sent = '''Account_name: Music_Entertainment_Multiplatform  
 Condition_name: High CPU% 
 Current_state: open 
 Details: CPU % > 90 for at least 15 minutes on 'host-qa-arccms-03.us-east-1.aws.cloud.viacom.com' 
 Incident_id: 139349191 
 Incident_url: https://alerts.newrelic.com/accounts/1519096/incidents/139349191 
 Owner: 
 Policy_name: [Cloud Engineering] P3 Base Server Policy 
 Policy_url: https://alerts.newrelic.com/accounts/1519096/policies/379367 
 Runbook_url: https://confluence.mtvi.com/display/INO/Alert+Handling+by+Policy 
 '''


from predict import Predict
pred_obj_1 = Predict('assignment_group')
pred_obj_2 = Predict('cleaned_tags')

pred_obj_2.ModelPredictionTAG(sent)
pred_obj_1.ModelPredictionAG(sent)





 