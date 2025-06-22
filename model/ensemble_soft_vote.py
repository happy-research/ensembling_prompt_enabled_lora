import csv
import ast
import torch.nn as nn
import torch

softmax=nn.Softmax(dim=0)

def soft_vote(list1, list2, list3, list4, list5, list6, list7):
    n=len(list1)
    result=[]
    #print(list1)
    list1=torch.Tensor(list1)
    list1=softmax(list1)
    #print(list1)
    #print(list2)
    list2=torch.Tensor(list2)
    list2=softmax(list2)
    #print(list2)
    #print(list3)
    list3=torch.Tensor(list3)
    list3=softmax(list3)
    #print(list3)
    #assert(1==2)
    for i in range(n):
        tmp=list1[i]+list2[i]+list3[i]
        result.append(tmp)

    #print(result)
    #assert(1==2)
    final_result=zip(result, range(0,n))
    #print(final_result)
    final_result=sorted(final_result, key=lambda x : x[0], reverse=True)
    #print(final_result)

    index_result=[]
    for i in range(n):
        index_result.append(final_result[i][1])

    #print(index_result)
    #assert(1==2)
    return index_result

def vote(list1, list2, list3, list4, list5, list6, list7):
    n=len(list1)
    #print("n in vote is: ",n)
    result=[]
    for i in range(n):
        #check each index and vote
        tmp={}
        if(list1[i][1] not in tmp.keys()):
            tmp[list1[i][1]]=1
        else:
            tmp[list1[i][1]]+=1
            
        if(list2[i][1] not in tmp.keys()):
            tmp[list2[i][1]]=1
        else:
            tmp[list2[i][1]]+=1
            
        if(list3[i][1] not in tmp.keys()):
            tmp[list3[i][1]]=1
        else:
            tmp[list3[i][1]]+=1
        """
        if(list4[i][1] not in tmp.keys()):
            tmp[list4[i][1]]=1
        else:
            tmp[list4[i][1]]+=1
        
        if(list5[i][1] not in tmp.keys()):
            tmp[list5[i][1]]=1
        else:
            tmp[list5[i][1]]+=1
            
        if(list6[i][1] not in tmp.keys()):
            tmp[list6[i][1]]=1
        else:
            tmp[list6[i][1]]+=1
            
        if(list7[i][1] not in tmp.keys()):
            tmp[list7[i][1]]=1
        else:
            tmp[list7[i][1]]+=1
        """
        sort_vote=sorted(tmp.items(), key=lambda x:x[1],reverse=True)
        #print(tmp)
        #print(sort_vote)
        
        result.append(sort_vote[0][0])#remember the vote reuslt on index i
    """
    for i in range(n):
        if(i not in result):
            print("result is: ",result)
            break
    """
    return result

record=[[] for i in range(3)]#3 model for ensemble
count=0
file_name=["./result/LoRA_ConfigProductRecommend/alpindale-Llama-3.2-3B/r4_run1/LLM_LoRA_all_epoch_5_test_res.csv",\
          "./result/LoRA_ConfigProductRecommend/alpindale-Llama-3.2-3B/r4_run2/LLM_LoRA_all_epoch_2_test_res.csv",\
          "./result/LoRA_ConfigProductRecommend/alpindale-Llama-3.2-3B/r4_run3/LLM_LoRA_all_epoch_3_test_res.csv",\
          "new test result/ensemblePBERT/ensemble_prompt_NoFT/No4/mixprompt_NoFT_all_epoch_20_test_res.csv",\
          "new test result/ensemblePBERT/ensemble_prompt_NoFT/No5/mixprompt_NoFT_all_epoch_20_test_res.csv",\
          "new test result/ensemblePBERT/ensemble_prompt_NoFT/No6/mixprompt_NoFT_all_epoch_20_test_res.csv",\
          "new test result/ensemblePBERT/ensemble_prompt_NoFT/No7/mixprompt_NoFT_all_epoch_20_test_res.csv"]

for i in range(3):#3 model for ensemble
    file=file_name[i]
    count=0
    k=14#indicate the attribute index
    with open(file,newline='') as csvfile:
        read=csv.reader(csvfile)
        for item in read:
            if(count==0):
                count+=1
                continue
            #tmp=item[:]
            tmp=item[:1]+item[k:k+3]
            #print(tmp)
            tmp[0]=int(tmp[0])
            tmp[1]=int(tmp[1])
            tmp[2]=int(tmp[2])

            #assert(1==2)
            #numpy version
            """tmp[3]=tmp[3][1:-1].split(' ')
            #print(tmp[3])
            n=len(tmp[3])
            logit=[]
            for i in range(n):
                if(tmp[3][i]==''):
                    continue
                logit.append(float(tmp[3][i]))"""

            #normal version
            tmp[3]=ast.literal_eval(tmp[3])

            """tmp[3]=logit"""
            record[i].append(tmp)

    #print(record[i][0])
    #print(len(record[i]))


#--------------------------precision rate-------------------------------

n=len(record[0])
#print("n is ",n)
cls_num=len(record[0][0][3])#class number used to define range of k
deno=[0 for i in range(cls_num)]#number of samples of each class
nume=[[0 for j in range(cls_num)] for i in range(cls_num)]#external range is for each class, internal range is for @k
#e.g. nume[i][j] is the numerator for calculating class i's @j recall rate

for i in range(0,n):
    label_i=record[0][i][1]
    
    #create a list containing tuples in form of (logit, index)
    ind_len=len(record[0][i][3])

    logit_index_list_1=record[0][i][3]
    #logit_index_list_1=zip(record[0][i][3], range(0,ind_len))
    #logit_index_list_1=list(logit_index_list_1)
    #logit_index_list_1=sorted(logit_index_list_1, key=lambda x : x[0], reverse=True)
    #print('list 1 length: ',len(logit_index_list_1))

    logit_index_list_2=record[1][i][3]
    #logit_index_list_2=zip(record[1][i][3], range(0,ind_len))
    #logit_index_list_2=list(logit_index_list_2)
    #logit_index_list_2=sorted(logit_index_list_2, key=lambda x : x[0], reverse=True)
    #print('list 2 length: ',len(logit_index_list_2))

    logit_index_list_3=record[2][i][3]
    #logit_index_list_3=zip(record[2][i][3], range(0,ind_len))
    #logit_index_list_3=list(logit_index_list_3)
    #logit_index_list_3=sorted(logit_index_list_3, key=lambda x : x[0], reverse=True)
    #print('list 3 length: ',len(logit_index_list_3))

    #hard voting
    #logit_index_list=vote(logit_index_list_1, logit_index_list_2, logit_index_list_3, None, None, None, None)
    #soft voting
    logit_index_list=soft_vote(logit_index_list_1, logit_index_list_2, logit_index_list_3, None, None, None, None)
    """
    logit_index_list_4=zip(record[3][i][3], range(0,ind_len))
    logit_index_list_4=list(logit_index_list_4)
    logit_index_list_4=sorted(logit_index_list_4, key=lambda x : x[0], reverse=True)
    
    logit_index_list_5=zip(record[4][i][3], range(0,ind_len))
    logit_index_list_5=list(logit_index_list_5)
    logit_index_list_5=sorted(logit_index_list_5, key=lambda x : x[0], reverse=True)
    
    logit_index_list_6=zip(record[5][i][3], range(0,ind_len))
    logit_index_list_6=list(logit_index_list_6)
    logit_index_list_6=sorted(logit_index_list_6, key=lambda x : x[0], reverse=True)
    
    logit_index_list_7=zip(record[6][i][3], range(0,ind_len))
    logit_index_list_7=list(logit_index_list_7)
    logit_index_list_7=sorted(logit_index_list_7, key=lambda x : x[0], reverse=True)
    
    logit_index_list=vote(logit_index_list_1, logit_index_list_2, logit_index_list_3, logit_index_list_4, logit_index_list_5, logit_index_list_6, logit_index_list_7)
    """
    
    #turn the list above to a index list
    index_list=logit_index_list
    """index_list=[]
    for item in logit_index_list:
        index_list.append(item[1])"""
    #print('index_list length: ',len(index_list))
    
    #assert(1==2)
    #record info for calculating the @k recall rate
    deno[label_i]+=1
    for j in range(cls_num):
        #(j+1) is k
        if(label_i in index_list[:j+1]):
            nume[label_i][j]+=1

precision_rate_k=[0 for i in range(cls_num)]
#calculate the @k recall rate
for i in range(cls_num):
    #(i+1) is k
    tmp=cls_num
    for j in range(cls_num):
        #j is class
        if(deno[j]==0):
            tmp-=1
            continue
        precision_rate_k[i]+=((nume[j][i]/((i+1)*deno[j]))*(deno[j]/sum(deno)))

for item in precision_rate_k:
    print(item)