import pandas as pd
import numpy as np
import os, sys, re, csv
from datetime import datetime as dt
from datetime import timedelta
import pickle as pkl
import statistics
from collections import Counter, defaultdict
import random
from csv import reader



def get_common_path(relative_path):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(cur_path, relative_path)
    return file_path

def get_qualied_ptSK_set(input_path, prediction_start_daygap, days_before_index_date ):
    # get qualified ptSK set
    # if last record time >= prediction start day, the patient will be included as qualified patient
    qualified_ptSK_set = set()
    ptSK_all_set = set()
    ptSK_daygap_list_dic = defaultdict(list)
    excluded_set = set()
    # row_counter = 0
    with open(input_path, 'r') as input_file:
        csv_reader = reader(input_file)
        header = next(csv_reader)
    # Check file as empty
        if header != None:
            for row in csv_reader:
                ptSK_daygap_list_dic[row[1]].append(row[19]) #row[8]: day gap
                ptSK_all_set.add(row[1])
                
        for ptSK, daygap_list in ptSK_daygap_list_dic.items():
            if int(daygap_list[-1]) >= prediction_start_daygap:
                qualified_ptSK_set.add(ptSK)
        print(f'prediction_start_daygap is {prediction_start_daygap}, len of qualified_ptSK_set is {len(qualified_ptSK_set)}, len of ptSK_all_set is {len(ptSK_all_set)}', '\n')
    

    if prediction_start_daygap == 1 or prediction_start_daygap == 31 or prediction_start_daygap == 181 or prediction_start_daygap == 366:#or  prediction_start_daygap == 0:
        with open(input_path, 'r') as input_file:
            csv_reader = reader(input_file)
            header = next(csv_reader)
        # Check file as empty
            if header != None:
                for row in csv_reader:
                    daygap = int(row[21])
                    # if row[1] == '560499200734235':
                    #     print ('hi')
                    # if prediction_start_daygap == 0:
                    #     if row[2] < row[3]  and daygap < prediction_start_daygap and daygap > -days_before_index_date and row[1] in qualified_ptSK_set:
                    #         qualified_ptSK_set.remove(row[1])
                    #         excluded_set.add(row[1])
                    if prediction_start_daygap == 1:
                        if daygap >= 0 and daygap < prediction_start_daygap and row[15] == '1' and row[1] in qualified_ptSK_set:
                            qualified_ptSK_set.remove(row[1])
                            excluded_set.add(row[1])
                    if prediction_start_daygap == 31:
                        # if (daygap >= 0 and daygap < prediction_start_daygap) and (row[15] == '1' or row[16] == '1' ) and (row[1] in qualified_ptSK_set):
                        #     qualified_ptSK_set.remove(row[1])
                        #     excluded_set.add(row[1])
                        if (row[15] == '1' or row[16] == '1' ) and (row[1] in qualified_ptSK_set):
                            qualified_ptSK_set.remove(row[1])
                            excluded_set.add(row[1])
                    if prediction_start_daygap == 181:
                        # if (daygap >= 0 and daygap < prediction_start_daygap) and (row[15] == '1' or row[16] == '1' or row[17] == '1') and (row[1] in qualified_ptSK_set):
                        #     qualified_ptSK_set.remove(row[1])
                        #     excluded_set.add(row[1])
                        if (row[15] == '1' or row[16] == '1'  or row[17] == '1' ) and (row[1] in qualified_ptSK_set):
                            qualified_ptSK_set.remove(row[1])
                            excluded_set.add(row[1])
                        #print(len(qualified_ptSK_set))
                    if prediction_start_daygap == 366:
                        #if (row[15] == '1' or row[16] == '1'  or row[17] == '1' or row[18]== '1') and (row[1] in qualified_ptSK_set):
                        if (row[19] == '1' ) and (row[1] in qualified_ptSK_set):
                            qualified_ptSK_set.remove(row[1])
                            excluded_set.add(row[1])
            print(len(excluded_set))
            # file = open("Python.txt", "w")
            # str = repr(excluded_set)
            # file.write("input_dictionary = " + str + "\n")
            # file.close()
            print(f'prediction_start_daygap is {prediction_start_daygap}, len of modified qualified_ptSK_set is {len(qualified_ptSK_set)}, len of ptSK_all_set is {len(ptSK_all_set)}', '\n')
    return qualified_ptSK_set

def get_qualified_input_encid_dic(outcome_input_path, qualified_ptSK_set, prediction_start_daygap, days_before_index):
    # to get the input enc_id before prediction start
    ptSK_qualified_input_encid_dic = defaultdict(dict)
    row_counter = 0
    with open(outcome_input_path, 'r') as new_outcome_file:
        csv_reader = reader(new_outcome_file)
        header = next(csv_reader)
        # outcome_label_columns_list ['patient_sk-0', 'encounter_id-1', 'age_in_years-2', 'age_group_label-3', 'discharged_dt_tm-4', 'ischemic_endpoint_label-5', 'bleeding_endpoint_label-6', 'bleeding_label-7', 'GI_bleeding_label-8', 'intracranial_bleeding_label-9', 'transfusion_label-10', 'hemoglobin_mean_result-11', 'hemoglobin_max_min_difference-12', 'ischemic_label-13', 'acute_ischemic_event_label-14', 'eluting_stent_label-15', 'revascularization_label-16', 'stroke_label-17', 'stroke_ischemic_label-18', 'DAPT_label-19', 'day_gap-20', 'index_encounter_label-21', 'expire_label-22'] 
        if header != None:
            for row in csv_reader:
                if row[1] in qualified_ptSK_set: 
                    #if row[1] == '560499200154057':
                        #print('hi')                       
                    # 'encounter_id-1', 'discharged_dt_tm'-4, 'day_gap'-20
                    daygap = int(row[19])
                    # Here we decide to consider the aggregated or the actual encounter????
                    # Considering the actual or aggregated for testing for inclusion will not make a difference
                    # actual_encounter_dt = row[2].split(' ')[0] # only needs date, no hour-minute-sec
                    agg_encounter_dt = row[22].split(' ')[0] # only needs date, no hour-minute-sec
                    #discharged_dt = row[2].split(' ')[0] # only needs date, no hour-minute-sec
                    # record two years before index record
                    # if daygap < prediction_start_daygap and daygap >= -730:
                    
                    if daygap < prediction_start_daygap and daygap >= -days_before_index: # and daygap !=0 :
                        #if daygap != 0:
                        ptSK_qualified_input_encid_dic[row[1]][row[0]] = agg_encounter_dt
    print(f'len of ptSK_qualified_input_encid_dic is {len(ptSK_qualified_input_encid_dic)}', '\n')
    
    return ptSK_qualified_input_encid_dic

def get_case_control_set(outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap):
    '''
    to get case and control 
    during [prediction_start_daygap, prediction_end_daygap]
    '''
    case_ptSK_set = set()
    control_ptSK_set = set()
    case_dic = defaultdict(list)
  
    with open(outcome_input_path, 'r') as new_outcome_file:
        csv_reader = reader(new_outcome_file)
        header = next(csv_reader)
        if header != None:
        # outcome_label_columns_list ['patient_sk-0', 'encounter_id-1', 'age_in_years-2', 'age_group_label-3', 'discharged_dt_tm-4', 'ischemic_endpoint_label-5', 'bleeding_endpoint_label-6', 'bleeding_label-7', 'GI_bleeding_label-8', 'intracranial_bleeding_label-9', 'transfusion_label-10', 'hemoglobin_mean_result-11', 'hemoglobin_max_min_difference-12', 'ischemic_label-13', 'acute_ischemic_event_label-14', 'eluting_stent_label-15', 'revascularization_label-16', 'stroke_label-17', 'stroke_ischemic_label-18', 'DAPT_label-19', 'day_gap-20', 'index_encounter_label-21', 'expi re_label-22']  
            for row in csv_reader:
                if row[1] in qualified_ptSK_set:
                    # if row[1] == '560499899730111':
                    #     print("hi")
                    daygap = int(row[19])
                    if daygap >= prediction_start_daygap and daygap <= prediction_end_daygap:
                        # for ischemic: 'acute_ischemic_event_label-14', 'stroke_ischemic_label-18',
                        #if int(row[4]) != 0: # mixed > 0   #test mixed state but doesn't take into consideration the MACE date
                        #if int(row[10]) != 0: # mixed > 0
                        if prediction_start_daygap == 0 and prediction_end_daygap == 30:
                            if row [15] == '1':
                                case_dic[row[1]].append(1)  
                        elif prediction_start_daygap == 0 and prediction_end_daygap == 365:
                            if row [16] == '1':
                                case_dic[row[1]].append(1)  
                        elif prediction_start_daygap == 0 and prediction_end_daygap == 1095:
                            if row [17] == '1':
                                case_dic[row[1]].append(1)  
                        elif prediction_start_daygap == 0 and prediction_end_daygap == 1825:
                            if  row[18] == '1':
                                case_dic[row[1]].append(1)  
                        # if row[4] == '1' or row[5] == '1'  or row[6] == '1' or row[7] == '1'  or row[8] == '1' or row[9] == '1':
                        #     case_dic[row[1]].append(1)
        for ptSK in qualified_ptSK_set:
            if ptSK in case_dic:
                case_ptSK_set.add(ptSK)
            else: 
                control_ptSK_set.add(ptSK)
    print(f'len of case_ptSK_set is {len(case_ptSK_set)}, len of control_ptSK_set is {len(control_ptSK_set)}', '\n')

    return case_ptSK_set, control_ptSK_set  

def get_specific_file_input_dic(file_type, input_file_path, qualified_ptSK_set, ptSK_qualified_input_encid_dic, tm_label, code_dic):
    '''
    tm_label: == 1: individual event time; tm_label == 2: discharged time
    ''' 
    count = 0
    data_lists = []
    if file_type == 'age':
        with open(input_file_path, 'r') as new_outcome_file:
            csv_reader = reader(new_outcome_file)
            header = next(csv_reader)
        # Check file as empty
            if header != None:
                for row in csv_reader: 
                    if row[1] in qualified_ptSK_set:
                        before_index_enc_dic = ptSK_qualified_input_encid_dic[row[1]]
                     
                        if row[0] in before_index_enc_dic:
                            age = int(float(row[20]))
                            age_code = 'A_' + str(age)
                            # age_list = [ptSK, age_code, discharged_tm]
                            age_list = [row[1], age_code, before_index_enc_dic[row[0]]]
                            if count < 5:
                                print(age_list)
                                count += 1
                            data_lists.append(age_list)

    elif file_type == 'gender':       
        with open(outcome_input_path, 'r') as new_outcome_file:
            csv_reader = reader(new_outcome_file)
            header = next(csv_reader)
        # Check file as empty
            if header != None:
                for row in csv_reader: 
                    if row[1] in qualified_ptSK_set:
                        before_index_enc_dic = ptSK_qualified_input_encid_dic[row[1]]
                        if row[0] in before_index_enc_dic:
                        #for _, discharged_dt in before_index_enc_dic.items():
                            ## gender_list: ['patient_sk-0', 'gender-2', 'discharged_dt]
                            gener_code = 'G_' + row[13]
                            #gender_list = [row[1], gener_code, discharged_dt]
                            gender_list = [row[1], gener_code, before_index_enc_dic[row[0]]]
                            if count < 5:
                                print(gender_list)
                                count += 1
                            data_lists.append(gender_list)

    elif file_type == 'diagnosis':
        # diagnosis = pd.read_csv(diag_head_input_path, delimiter = '\t', low_memory = False)
        # diagnosis['encounter_id'] = diagnosis.patid.astype(str)+'_'+diagnosis.fst_dt.astype(str)
        # diagnosis['modified_diag'] = np.where((diagnosis['icd_flag']=='10'), 'D_9_' + diagnosis['diag'].replace('.', ''), 'D_10_' + diagnosis['diag'].replace('.', ''))
        # diag_id_code_dic = diagnosis['modified_diag'].drop_duplicates().reset_index(drop=True).to_dict()
        #return diag_id_code_dic
        with open(orig_diag_input_path, 'r') as orig_diag_file: 
            #  ['patient_sk-0', 'encounter_id-1', 'diagnosis_id-2', 'diagnosis_priority-3', 'merged_encid_label-4']
            csv_reader = reader(orig_diag_file)
            header = next(csv_reader)
        # Check file as empty
            if header != None:
                for row in csv_reader: 
                    #print(row[0],row[0] in qualified_ptSK_set)
                    if row[0] == '560499899955561':
                        print("found")
                    if  row[0] in qualified_ptSK_set:
                        before_index_enc_dic = ptSK_qualified_input_encid_dic[row[0]]
                        #encounter_id = row[0]+'_'+row[1]
                        if row[1] in before_index_enc_dic:
                            diag_code = row[2]
                            agg_encounter_dt  = before_index_enc_dic[row[1]]
                            diag_list = [row[0], diag_code, agg_encounter_dt]
                            if count < 5:
                                print(diag_list)
                                count += 1
                            data_lists.append(diag_list)

    elif file_type == 'medication':
        with open(orig_med_input_path, 'r') as orig_med_file:
            # ['patient_sk-0', 'encounter_id-1', 'medication_id-2', 'total_dispensed_doses-3', 'dose_quantity-4', 'initial_dose_quantity-5', 'order_strength-6', 'med_started_dt_tm-7', 'med_entered_dt_tm-8', 'med_stopped_dt_tm-9', 'med_discontinued_dt_tm-10', 'merged_encid_label-11']
            csv_reader = reader(orig_med_file)
            header = next(csv_reader)
            if header != None:
                for row in csv_reader:
                    if row[0] in qualified_ptSK_set:
                        # get index_enc_dic
                        before_index_enc_dic = ptSK_qualified_input_encid_dic[row[0]]
                        if row[1] in before_index_enc_dic:
                            generic_name_code = row[2]
                            agg_encounter_dt = before_index_enc_dic[row[1]]
                            med_list = [row[0], generic_name_code, agg_encounter_dt]
                            if count < 5:
                                print(med_list)
                                count += 1
                            data_lists.append(med_list)

    elif file_type == 'procedure':
        with open(orig_proce_input_path, 'r') as orig_proce_file:
            # ['patient_sk-0', 'encounter_id-1', 'procedure_id-2', 'procedure_code-3', 'procedure_type-4', 'procedure_cat-5', 'procedure_priority-6', 'procedure_dt_tm-7', 'merged_encid_label-8']   
            csv_reader = reader(orig_proce_file)
            header = next(csv_reader)
            if header != None:
                for row in csv_reader:
                    if row[0] in qualified_ptSK_set:
                        # get index_enc_dic
                        before_index_enc_dic = ptSK_qualified_input_encid_dic[row[0]]
                        if row[1] in before_index_enc_dic:
                            new_proce_code = row[2]
                            agg_encounter_dt = before_index_enc_dic[row[1]]
                            proce_list = [row[0], new_proce_code, agg_encounter_dt]
                            if count < 5:
                                print(proce_list)
                                count += 1
                            data_lists.append(proce_list)
               
    return data_lists

def write_data_lists_to_tsv(all_all_data_lists, output_path):
    with open(output_path, 'wt') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        head_row = ['Pt_id', 'ICD', 'Time']
        writer.writerow(head_row)
        for lst in all_all_data_lists:
            writer.writerows(lst)
        print(f'Finished write_data_lists_to_tsv for --{len(all_all_data_lists)}-- types of input files')

def deduplicate_tsv(output_path, dedup_output_path):
    # to depulicate the same row in tsv 
    with open(output_path, 'r') as output_file:
        df_data = pd.read_csv(output_file, sep = "\t")
        print(f'before depulication: row num is {len(df_data)}; column num is {len(df_data.columns)}') 
        df_data_result = df_data.drop_duplicates()
        print(f'after depulication: row num is {len(df_data_result)}; column num is {len(df_data_result.columns)}') 
        result_tsv = df_data_result.to_csv(dedup_output_path, sep = '\t', index = False, header = True)

    return result_tsv

def get_input_file_1st_splitting(common_path):
    caseFile = common_path + 'dedup_case.tsv'
    controlFile = common_path + 'dedup_control.tsv' 
    typeFile = 'NA'
    # outFile = common_path + 'Results/' + file_type
    outFile = common_path  + 'lt'
    cls_type = 'binary'
    pts_file_pre = 'NA'

    return caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre

def get_input_file_non_1st_splitting(common_path, first_splitting_folder_path):
    caseFile = common_path + 'dedup_case.tsv'
    controlFile = common_path + 'dedup_control.tsv' 
    typeFile = 'NA'
    outFile = common_path  +'lt'
    # outFile = common_path + 'Results/' + file_type
    cls_type = 'binary'
    pts_file_pre = first_splitting_folder_path + 'lt'+'.pts'
    # 2nd time: /data/fli2/Stent/Cerner/Results/For_ML/Results_20210219/Results_all_input_all_time_180_6m_window/Results/bleeding.pts.test

    return caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre

def spliting_train_test_data(caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre):
    #_start = timeit.timeit()
    # debug = False
    #np.random.seed(1)
    # time_list = []
    # dates_list =[]
    label_list = []
    pt_list = []

    print ("Loading cases and controls" ) 

    ## loading Case
    print('loading cases')
    data_case = pd.read_table(caseFile)
    # data_case.columns = ["Pt_id", "ICD", "Time", "tte"] # ?"tte": time to event
    # index: stenting   1st bleeding: 100 days; 150; 200
    # survival analysis: only get the first outcome event

    if cls_type=='surv':
        data_case = data_case[["Pt_id", "ICD", "Time", "tte"]]
    else:
        data_case = data_case[["Pt_id", "ICD", "Time"]]
    data_case['Label'] = 1
    #data_case=data_case[~(data_case["ICD"].str.startswith('P') |data_case["ICD"].str.startswith('L'))] ### use if you need to exclude or include only certain type of codes
    print('Case counts: ', data_case["Pt_id"].nunique())

    ## loading Control
    print('loading ctrls')
    data_control = pd.read_table(controlFile)
    if cls_type == 'surv':
        data_control = data_control[["Pt_id", "ICD", "Time", "tte"]]
    else:
        data_control = data_control[["Pt_id", "ICD", "Time"]]
    data_control['Label'] = 0

    #data_control=data_control[~(data_control["ICD"].str.startswith('P') | data_control["ICD"].str.startswith('L'))] ### use if you need to exclude certain type of codes
    print('Ctrl counts: ', data_control["Pt_id"].nunique())

    ### An example of sampling code: Control Sampling
    #print('ctrls sampling')       
    #ctr_sk=data_control["Pt_id"]
    #ctr_sk=ctr_sk.drop_duplicates()
    #ctr_sk_samp=ctr_sk.sample(n=samplesize_ctrl)
    #data_control=data_control[data_control["Pt_id"].isin(ctr_sk_samp.values.tolist())]

    data_l = pd.concat([data_case, data_control])
    print('total counts: ', data_l["Pt_id"].nunique())   

    ## loading the types
    if typeFile=='NA': 
       types = {"Zeropad":0}
    else:
      with open(typeFile, 'rb') as t2:
             types = pkl.load(t2)

    label_list = []
    pt_list = []
    dur_list = []
    newVisit_list = []
    count = 0

    for Pt, group in data_l.groupby('Pt_id'):
        data_i_c = [] # i - icd code
        data_dt_c = [] # dt - date
        for Time, subgroup in group.sort_values(['Time'], ascending = False).groupby('Time', sort=False): ### ascending=True normal order ascending=False reveresed order
            data_i_c.append(np.array(subgroup['ICD']).tolist())             
            data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
        if len(data_i_c) > 0:
            # creating the duration in days between visits list, first visit marked with 0        
            v_dur_c = []
        if len(data_dt_c) <= 1:
            v_dur_c = [0]
        else:
            for jx in range(len(data_dt_c)):
                if jx == 0:
                    v_dur_c.append(jx)
                else:
                    #xx = (data_dt_c[jx]- data_dt_c[jx-1]).days ### normal order
                    xx = (data_dt_c[jx-1] - data_dt_c[jx]).days ## reversed order                            
                    v_dur_c.append(xx)                                  

        ### Diagnosis recoding
        newPatient_c = []
        for visit in data_i_c:
            newVisit_c = []
            for code in visit:
                if code in types: 
                    newVisit_c.append(types[code])
                else:                             
                    types[code] = max(types.values()) + 1 
                    # types[code] = len(types) + 1
                    newVisit_c.append(types[code])
            newPatient_c.append(newVisit_c)

        if len(data_i_c) > 0: ## only save non-empty entries
            if cls_type == 'surv':
                label_list.append([group.iloc[0]['Label'], group.iloc[0]['tte']]) #### LR ammended for surv
            else:
                label_list.append(group.iloc[0]['Label'])
            pt_list.append(Pt)
            newVisit_list.append(newPatient_c)
            dur_list.append(v_dur_c)

        count = count + 1
        if count % 5000 == 0: print ('processed %d pts' % count)

    ### Creating the full pickled lists ### uncomment if you need to dump the all data before splitting
    #pickle.dump(label_list, open(outFile+'.labels', 'wb'), -1)
    #pickle.dump(newVisit_list, open(outFile+'.visits', 'wb'), -1)
    pkl.dump(types, open(outFile + '.types', 'wb'), -1)
    #pickle.dump(pt_list, open(outFile+'.pts', 'wb'), -1)
    #pickle.dump(dur_list, open(outFile+'.days', 'wb'), -1)

         
    fset = []
    print('Reparsing')
    for pt_idx in range(len(pt_list)):
        pt_sk = pt_list[pt_idx]
        pt_lbl = label_list[pt_idx]
        pt_vis = newVisit_list[pt_idx]
        pt_td = dur_list[pt_idx]
        n_seq = []
        for v in range(len(pt_vis)):
            nv = []
            nv.append([pt_td[v]])
            nv.append(pt_vis[v])                   
            n_seq.append(nv)
        n_pt = [pt_sk, pt_lbl, n_seq]
        fset.append(n_pt)  
           
### 5-Fold Cross Validation split to train, test and validation sets
    print("Splitting")
    num_splits=5
    if pts_file_pre == 'NA':
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold, KFold
        
        
        
        kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=31)
        #kfold = KFold(n_splits=num_splits, shuffle=True,  random_state=41)
        X = np.array(pt_list)
        Y = np.array(label_list)
       
        fold_index = 0
        for train, test in kfold.split(X, Y):
            #indices_train_f,indices_validate_f  = train_test_split(train, test_size=0.125, random_state=31, stratify=Y[train])
            indices_train_f,indices_validate_f  = train_test_split(train, test_size=0.125, random_state=41)
            for subset in ['train', 'valid', 'test']:
                if subset =='train':
                    indices = indices_train_f
                elif subset =='valid':
                    indices = indices_validate_f
                elif subset =='test':
                    indices = test.tolist()
                else: 
                    print ('error')
                subset_p = [pt_list[i] for i in indices]
                nptfile = outFile + '.pts.' + subset + str(fold_index)
                pkl.dump(subset_p, open(nptfile, 'wb'), protocol=2)  

            train_set_full = [fset[i] for i in indices_train_f]
            test_set_full = [fset[i] for i in test.tolist()]
            valid_set_full = [fset[i] for i in indices_validate_f]
            ctrfilename = outFile + '.combined.train' + str(fold_index)
            ctstfilename = outFile + '.combined.test' + str(fold_index)
            cvalfilename = outFile + '.combined.valid'  + str(fold_index)  
            pkl.dump(train_set_full, open(ctrfilename, 'wb'), -1)
            pkl.dump(test_set_full, open(ctstfilename, 'wb'), -1)
            pkl.dump(valid_set_full, open(cvalfilename, 'wb'), -1)
            fold_index +=1
    else:
        print('loading previous splits')
        # for fold_index in range(num_splits):
        #     pt_train = pkl.load(open(pts_file_pre + '.train' +str(fold_index), 'rb'))
        #     pt_valid = pkl.load(open(pts_file_pre + '.valid' +str(fold_index), 'rb'))
        #     pt_test = pkl.load(open(pts_file_pre + '.test'+str(fold_index), 'rb'))
            
        #     test_indices = np.intersect1d(pt_list, pt_test, assume_unique=True, return_indices=True)[1]
        #     valid_indices= np.intersect1d(pt_list, pt_valid, assume_unique=True, return_indices=True)[1]
        #     train_indices= np.intersect1d(pt_list, pt_train, assume_unique=True, return_indices=True)[1]
            
        #     fset = []
        #     print('Reparsing')
        #     for pt_idx in range(len(pt_list)):
        #         pt_sk = pt_list[pt_idx]
        #         pt_lbl = label_list[pt_idx]
        #         pt_vis = newVisit_list[pt_idx]
        #         pt_td = dur_list[pt_idx]
        #         n_seq = []
        #         for v in range(len(pt_vis)):
        #             nv = []
        #             nv.append([pt_td[v]])
        #             nv.append(pt_vis[v])                   
        #             n_seq.append(nv)
        #         n_pt = [pt_sk, pt_lbl, n_seq]
        #         fset.append(n_pt)     
        #     train_set_full = [fset[i] for i in train_indices]
        #     test_set_full = [fset[i] for i in   test_indices]
        #     valid_set_full = [fset[i] for i in  valid_indices]
        #     # train_set_full = pt_train
        #     # test_set_full = pt_test
        #     # valid_set_full =  pt_valid
        #     ctrfilename = outFile + '.combined.train' + str(fold_index)
        #     ctstfilename = outFile + '.combined.test' + str(fold_index)
        #     cvalfilename = outFile + '.combined.valid'  + str(fold_index)  
        #     pkl.dump(train_set_full, open(ctrfilename, 'wb'), -1)
        #     pkl.dump(test_set_full, open(ctstfilename, 'wb'), -1)
        #     pkl.dump(valid_set_full, open(cvalfilename, 'wb'), -1)
   

if __name__ == "__main__":
    outcome_input_common_path = get_common_path('../Data_V3/')
    outcome_input_path = outcome_input_common_path + 'patients_encounters_15_V3.csv'
    orig_diag_input_path = outcome_input_common_path + 'mod_diagnosis.csv'
    orig_med_input_path = outcome_input_common_path + 'mod_drug.csv'
    orig_proce_input_path = outcome_input_common_path + 'mod_proc.csv'

    output_folder_name = 'Results_15_V3'
    output_part_common_path = get_common_path('../Results_V3/') + output_folder_name + '/'
            
    count_file = 0
    days_before_index_date = 1095
    #daygap_parameter_dic = {'0_30' : [0, 30], '0_365':[0, 365],'0_1095':[0, 1095],'0_1825':[0, 1825]}
    #daygap_parameter_dic = {'0_1095':[0, 1095],'0_1825':[0, 1825]}
    daygap_parameter_dic = {'0_30' : [0, 30]}
    #daygap_parameter_dic = {'0_365':[0, 365]}
    #daygap_parameter_dic = {'0_1095':[0, 1095]}
    #daygap_parameter_dic = {'0_1825':[0, 1825]}
    for key_para, value_list in daygap_parameter_dic.items():       
        prediction_start_daygap, prediction_end_daygap = value_list[0], value_list[1]      
        start_time = dt.now()
        start_time_format = start_time.strftime("%Y-%m-%d %H:%M:%S")
        count_file += 1
        print(f'{count_file}. Starting processing {key_para}')
        print(f'start time to get multiple lists is {start_time_format}')
        print(f'prediction_start_daygap is {prediction_start_daygap}d, prediction_end_daygap is {prediction_end_daygap}d', '\n')
        specific_folder_name = 'Results_' + str(prediction_start_daygap) + 'd_' + str(prediction_end_daygap) + 'd_window'
        
        qualified_ptSK_set = get_qualied_ptSK_set(outcome_input_path, prediction_start_daygap, days_before_index_date )
        ptSK_qualified_input_encid_dic = get_qualified_input_encid_dic(outcome_input_path, qualified_ptSK_set, prediction_start_daygap, days_before_index_date)
        case_ptSK_set, control_ptSK_set = get_case_control_set(outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap) 
        
        to_write_and_deduplicate_tsv_label = 1
        if to_write_and_deduplicate_tsv_label:
            #ptSK_set_list = [case_ptSK_set, control_ptSK_set]
            ptSK_set_list = [control_ptSK_set]
            output_file_name_list = ['case.tsv', 'control.tsv']
    
            for idx1, ptSK_set in enumerate(ptSK_set_list):
                print(f'{count_file}-{idx1+1}. start generating {output_file_name_list[idx1]} for {specific_folder_name}')
                
                age_lists = get_specific_file_input_dic('age', outcome_input_path, ptSK_set, ptSK_qualified_input_encid_dic, tm_label = False, code_dic = False)
                gender_lists = get_specific_file_input_dic('gender', outcome_input_path, ptSK_set, ptSK_qualified_input_encid_dic, tm_label = False, code_dic = False)

                # # # race_lists = get_specific_file_input_dic('race', history_input_path, ptSK_set, ptSK_qualified_input_encid_dic, tm_label = False, code_dic = False)

                diag_lists = get_specific_file_input_dic('diagnosis', orig_diag_input_path, ptSK_set, ptSK_qualified_input_encid_dic, 2, code_dic = False)
                #print("hi")
                # # # event_lists = get_specific_file_input_dic('event', orig_event_input_path, ptSK_set, ptSK_qualified_input_encid_dic, 2, event_id_group_name_dic)

                # # # lab_lists = get_specific_file_input_dic('lab', orig_lab_input_path, ptSK_set, ptSK_qualified_input_encid_dic, 2, code_dic = False) 

                med_lists = get_specific_file_input_dic('medication', orig_med_input_path, ptSK_set, ptSK_qualified_input_encid_dic, 2, code_dic = False)

                proce_lists = get_specific_file_input_dic('procedure', orig_proce_input_path, ptSK_set, ptSK_qualified_input_encid_dic, 2, code_dic = False)

                                
                # # all_input_disch_time_list = [age_lists, gender_lists, race_lists, diag_lists, event_lists, lab_lists, med_lists, proce_lists] 
                #all_input_disch_time_list = [age_lists, gender_lists, diag_lists, med_lists, proce_lists, ptSK_DAPT_label_lists] 
                all_input_disch_time_list = [age_lists, gender_lists, diag_lists, med_lists,  proce_lists]
                #all_input_disch_time_list = [age_lists, diag_lists, med_lists, proce_lists] 
                #all_input_disch_time_list = [age_lists, med_lists, proce_lists] 
                #all_input_disch_time_list = [diag_lists, med_lists, proce_lists]#, proce_lists] 
                #all_input_disch_time_list = [age_lists] 
                # # all_input_disch_time_list = [ptSK_DAPT_label_lists]    
                
                # # for idx2, input_list in enumerate(all_input_lists):                    
                output_common_path = output_part_common_path + str(days_before_index_date) + "/" + specific_folder_name + '/'
                if not os.path.exists(output_common_path):
                    print("hello")
                    os.makedirs(output_common_path) 
                output_path = output_common_path + output_file_name_list[idx1]
                dedup_output_path = output_common_path + 'dedup_' + output_file_name_list[idx1] 
                print(f'{count_file}-{idx1+1}-1. Start writing {output_file_name_list[idx1]}-{specific_folder_name}')            
                write_data_lists_to_tsv(all_input_disch_time_list, output_path)
                print(f'{count_file}-{idx1+1}-2. start deduplicating {output_file_name_list[idx1]}-{specific_folder_name}') 
                dedup_result_csv = deduplicate_tsv(output_path, dedup_output_path) 
                print(f'{count_file}-{idx1+1}-3. finish deduplicating {output_file_name_list[idx1]}-{specific_folder_name}', '\n')    
        
        end_time_case_ctl = dt.now()
        print(f'Time for get case/control is {end_time_case_ctl - start_time}', '\n')
        
        ## to split into train/valid/test files 
        to_split_train_test_label = 1
        if to_split_train_test_label:
            print(f'Start to_split_train_test_label at {dt.now()}')
            print(f'Generating- train/test/valid for {specific_folder_name}')            
            split_input_file_path = output_part_common_path +  str(days_before_index_date) +  "/" +  specific_folder_name + '/'           
            #if specific_folder_name == 'Results_0d_365d_window': 
            print(f'count_file is {count_file}, the first splitting train/test/control')
            caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre = get_input_file_1st_splitting(split_input_file_path)
            #else: 
                #print(f'count_file is {count_file}, non-first splitting train/test/control')
                #first_splitting_folder_path = output_part_common_path + str(days_before_index_date) +  "/" + 'Results_0d_365d_window/'
                #caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre = get_input_file_non_1st_splitting(split_input_file_path, first_splitting_folder_path)
            
            spliting_train_test_data(caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre)
            print(f'{count_file}-{idx1+ 1}. finish generating - train/test/valid for {specific_folder_name}', '\n')
        
        end_time_train_test = dt.now()
        print(f'Time for get train/test/valid is {end_time_train_test - end_time_case_ctl}, total time for get case/control and  train/test/valid is {end_time_train_test - start_time}', '\n')