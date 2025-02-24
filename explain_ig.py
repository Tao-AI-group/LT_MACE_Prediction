## Import basic requirements
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),'libs/'))
os.environ['CUDA_VISIBLE_DEVICES'] = '14'
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pk
import pandas as pd
import random
use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(6)

### Load our EHR model related files
from EHRDataloader import EHRdataFromPickles, EHRdataloader  
from EHREmb import EHREmbeddings
import models as model 
import utils as ut 


from importlib import reload



use_cuda = torch.cuda.is_available()

## Just using original code for now
def sigmoid(x):
    return (1 / (1 + torch.exp(-x)))


#let's define this class method
def EmbedPatients_MB_explain(bmodel,mb_t, mtd): #let's define this
    bmodel.bsize=len(mb_t) ## no of pts in minibatch
    embedded_1 = bmodel.embed(mb_t).detach().requires_grad_(True)  ## Embedding for codes
    embedded = torch.sum(embedded_1, dim=2)#.detach().requires_grad_(True)
    if bmodel.time:
        mtd_t= Variable(torch.stack(mtd,0))
        # if use_cuda: mtd_t.cuda()
        out_emb= torch.cat((embedded,mtd_t),dim=2)
    else:
        out_emb= embedded
    if use_cuda:
        out_emb.cuda()        
    return out_emb, embedded_1


def int_grad_pt(best_model , diag_t, code_desc, pt , m=64 , qc=False):
    x1, label,seq_len,time_diff = pt
    with torch.no_grad(): output_score = best_model(x1,seq_len,time_diff)
    x_in,x_in2= EmbedPatients_MB_explain(best_model,x1,time_diff)
    
    #x_inp = nn.utils.rnn.pack_padded_sequence(x_in,seq_len,batch_first=True)
    
    code_visit_cont = torch.zeros_like(x_in2)
    best_model.train()### LR added for error
    if best_model.time==True:
        #f = lambda x, x1: best_model.sigmoid(best_model.out(torch.cat(best_model.rnn_c(torch.cat((x.sum(2), x1[...,-1:]), dim=2))[1][-1],best_model.rnn_c(torch.cat((x.sum(2), x1[...,-1:]), dim=2))[1][-2]))).squeeze()
        #f = lambda x, x1: best_model.sigmoid(best_model.out(best_model.rnn_c(torch.cat((x.sum(2), x1[...,-1:]), dim=2))[1][-1])).squeeze() 
        #output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1))) ## if multiclass will be softmax
    
        for k in range(1, m):
            x_in2.grad = None
            
            X = x_in2/m*k
            X1= x_in
            
            aa = torch.cat((X.sum(2), X1[...,-1:]), dim=2)
            bb = best_model.rnn_c(aa)
            cc = bb[1][-1]
            dd = bb[1][-2]
            ee = torch.cat((cc,dd), dim=1 )
            ff = best_model.out(ee)
            pred = best_model.sigmoid(ff).squeeze()
            
            #pred = f(x_in2/m*k, x_in)
            pred.backward()
            code_visit_cont += x_in2.grad/k
        best_model.eval()### LR added
        code_visit_cont *= x_in2
        
    else: 
        #f = lambda x, x1: best_model.sigmoid(best_model.out(best_model.rnn_c(x.sum(2))[1][-1])).squeeze()
        for k in range(1, m):
            x_in2.grad = None
            
            X = x_in2/m*k
            X1 = x_in
            
            aa = (X.sum(2), X1[1,-1:])
            bb = best_model.rnn_c(aa)
            cc = bb[1][-1]
            dd = bb[1][-2]
            ee = torch.cat((cc,dd), dim=1 )
            
            ff = best_model.out(ee)
            pred = best_model.sigmoid(ff).squeeze()
            
            #pred = f(x_in2/m*k, x_in)
            pred.backward()
            code_visit_cont += x_in2.grad/k
        best_model.eval()### LR added
        code_visit_cont *= x_in2
    
    ### QC check
    if qc==True:
    #     with torch.no_grad():
            
    #         diff = f(x_in2, x_in) - f(x_in2/m, x_in)
            
            
            
        print("sum of contribution scores = ", code_visit_cont.sum())
    
    ### building DF for visualizations
    if use_cuda: 
            pt_inp=pd.DataFrame(x1.squeeze(0).cpu().detach().numpy()).reset_index()
    else:   pt_inp=pd.DataFrame(x1.squeeze(0).numpy()).reset_index()
    pt_inp=pt_inp.melt(id_vars=['index'],value_vars=pt_inp.columns[1:])
    pt_inp.columns=['visit_ind','token_pos','TOKEN_ID']
    
    if use_cuda: code_contrib=pd.DataFrame(code_visit_cont.sum(-1).cpu().detach().squeeze(0).numpy()).reset_index()
    else: code_contrib=pd.DataFrame(code_visit_cont.sum(-1).detach().squeeze(0).numpy()).reset_index()
    code_contrib=code_contrib.melt(id_vars=['index'],value_vars=code_contrib.columns[1:])
    code_contrib.columns=['visit_ind','token_pos','cont_score']
    if best_model.time==True:
        with torch.no_grad(): 
            #print(x_in[...,-1:].shape)
            if x_in[...,-1:].shape[1]>1:
                if use_cuda: timedf=pd.DataFrame(x_in[...,-1:].squeeze(0).cpu().numpy())
                else: timedf=pd.DataFrame(x_in[...,-1:].squeeze(0).cpu().numpy())
            else:timedf=pd.DataFrame([0])
        timedf.columns=['Time']
        pt_inp=pd.merge(pt_inp, timedf, how='left',left_on='visit_ind',right_on=timedf.index)
    pt_inp = pd.merge(pt_inp,diag_t, how='left')
    pt_inp_cont=pd.merge(pt_inp,code_contrib, how='left')#.dropna()
    if best_model.time==True:
        pt_inp_cont=pd.merge(pt_inp_cont,code_desc, how='left')[['visit_ind','Time','token_pos','TOKEN_ID','DIAGNOSIS_ID','cont_score','cat','description']].drop_duplicates()
    else: pt_inp_cont=pd.merge(pt_inp_cont,code_desc, how='left')[['visit_ind','token_pos','TOKEN_ID','DIAGNOSIS_ID','cont_score','cat','description']].drop_duplicates()
    
    return label, output_score ,pt_inp_cont[pt_inp_cont['TOKEN_ID']>0]



def load_model(model_pth):
    best_model = torch.load(model_pth)
    if use_cuda:best_model.cuda()
    return best_model



### Load the data
def load_data(data_File,multilbl):
    ds = EHRdataFromPickles(root_dir = '', 
                                  file = data_File, 
                                  sort= False,
                                  model='RNN')
    mbs_list = list(tqdm(EHRdataloader(ds, batch_size=1, packPadMode=True,multilbl=multilbl)))
    
    
    
    return(mbs_list)



    


# load the token dict
def load_vocab_dict(type_File,desc_file):
    tk_dc= pk.load(open(type_File,'rb'))
    diag_t= pd.DataFrame.from_dict(tk_dc,orient='index').reset_index()
    diag_t.columns=['DIAGNOSIS_ID','TOKEN_ID']
    code_desc=pd.read_table(desc_file, sep=',')
    code_desc.columns=['DIAGNOSIS_ID','cat','description']
    return diag_t, code_desc
    
    # code_desc = diag_t.copy() 
    # code_desc['description'] = code_desc .loc[:, 'DIAGNOSIS_ID']
    # code_desc.columns=['DIAGNOSIS_ID','cat','description']
    # code_desc['cat'] = np.where(code_desc.DIAGNOSIS_ID.str[:1] == 'A',  "Age" , code_desc.cat)
    # code_desc['cat'] = np.where(code_desc.DIAGNOSIS_ID.str[:1] == 'P',  "Proc" , code_desc.cat)
    # code_desc['cat'] = np.where(code_desc.DIAGNOSIS_ID.str[:1] == 'M',  "Med" , code_desc.cat)
    # code_desc['cat'] = np.where(code_desc.DIAGNOSIS_ID.str[:1] == 'D',  "Diag" , code_desc.cat)
    # code_desc['cat'] = np.where(code_desc.DIAGNOSIS_ID.str[:1] == 'G',  "Gender" , code_desc.cat)
    
    # #code_desc=pd.read_table(desc_file)
    
    
    
    # return diag_t, code_desc




def calc_contribution (model_pth,data_File,type_File,desc_file,task='NA',qc=True):
  if task == 'NA': multilbl=False
  else:  multilbl=True
  
  best_model=load_model(model_pth)
  mbs_list=load_data(data_File,multilbl=multilbl)
  
  #another method to load data
  #import pickle
  #pkl_result = pickle.load(open(data_File, 'rb'), encoding='bytes')
  
  
  diag_t, code_desc = load_vocab_dict(type_File,desc_file)
  
  #diag_t = load_vocab_dict(type_File,desc_file)
  l_df=[]
  count = 0
  for i , pt in enumerate(mbs_list):
    label, output_score , pt_inp_cont_1=int_grad_pt(best_model , diag_t, code_desc, pt , m=64, qc=qc)
    pt_inp_cont_1['PT']=i
    if task=='mort':
          pt_inp_cont_1['True_label']=label[:,:,0].squeeze().cpu().numpy()
    elif task=='mort_surv':
          pt_inp_cont_1['True_label']=label[:,:,0:2].squeeze().cpu().numpy()
          pt_inp_cont_1['LOS']=label[:,:,1].squeeze().cpu().numpy()
    elif task=='NA':
          pt_inp_cont_1['True_label']=label.squeeze().cpu().numpy()
    else: print ('for this tutorial task can be either mort or NA')
      
    pt_inp_cont_1['predicted_score']=output_score.cpu().numpy()
    l_df.append(pt_inp_cont_1)
    df_f=pd.concat(l_df).sort_values(by=['PT','visit_ind'])
    
    # count = count + 1
    # if count == 1000:
    #      break
  
  return df_f


if __name__ == '__main__':

    # model_pth= sys.argv[1]
    # data_File= sys.argv[2]
    # type_File= sys.argv[3]
    # desc_File = sys.argv[4]
    # out_File = sys.argv[5]
    #task = sys.argv[5]
    
    model_pth = "/raid/aabdelhameed/Liver_Transplantation/Test_results/optuna_tuning/objective_BiGRU/0d_30d_test/test_model.p"
    
    data_File = "/raid/aabdelhameed/Liver_Transplantation/Results/Results_15_V2/1095/Results_0d_30d_window/lt.combined.test3"

    #parser = OptionParser()
    #(options, args) = parser.parse_args()
    
    #Original
    type_File  = "/raid/aabdelhameed/Liver_Transplantation/Results/Results_15_V2/1095/Results_0d_30d_window/lt.types"
    #type_File  = "/raid/aabdelhameed/Liver_Transplantation/Results_V3/Results_15_V3/1095/Results_0d_30d_window/lt.types"
    
    desc_file = "Liver_Transplantation/Data_V2/descriptions.csv"
    task = "NA"
   
    #df_f= calc_cont(model_pth,data_File,type_File,desc_file,task=task)

    df_f= calc_contribution(model_pth,data_File,type_File, desc_file,task=task)
    
    out_File = "/raid/aabdelhameed/Liver_Transplantation/Test_results/Saved/BiGRU/0d_30d/final_v3.csv"
    
    
    
    df_f.to_csv(out_File,sep='\t',index=False)
    print("Done")

