import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
import csv
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import AdamW

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name_or_path = "openai-community/gpt2"
tokenizer_name_or_path = "openai-community/gpt2"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

#print(model)

"""
#The following code is a demo for LoRA model testing.

text = "This is a sample input for the LoRA model."
inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Example label

# Forward pass
outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels, output_hidden_states=True)

print("input's shape: ",inputs['input_ids'].shape)
print(outputs.keys())
print(outputs.logits.shape)

print("length of the hidden_states: ",len(outputs.hidden_states))
print(outputs.hidden_states[0].shape)
print(outputs.hidden_states[1].shape)
print(outputs.hidden_states[2].shape)
loss = outputs.loss
logits = outputs.logits
"""

def read_data_csv(file,ratio):
    record=[]
    with open(file,newline='') as csvfile:
        read=csv.reader(csvfile)
        for item in read:
            record.append(item[1:])
    record=record[1:]
    for ind,sample in enumerate(record):
        sample.insert(0,ind)
        sample[2]=int(sample[2])#cpu
        sample[3]=int(sample[3])#graphic
        sample[4]=int(sample[4])#hardisk
        sample[5]=int(sample[5])#ram
        sample[6]=int(sample[6])#screen
    train_set, valid_set=random_split(record,
                 #[0.7,0.3],
                 ratio,
                 generator=torch.Generator().manual_seed(42))
    
    train_text=[]
    train_labels=[]
    valid_text=[]
    valid_labels=[]
    for record in train_set:
        tmp=record[1]+"Given the above needs information, predict the corresponding processor, graphic card, hard disk, ram and screen configuration separately in the following five label token: label label label label label."
        #train_text.append(record[1])
        train_text.append(tmp)
        train_labels.append(record[2:])
    for record in valid_set:
        tmp=record[1]+"Given the above needs information, predict the corresponding processor, graphic card, hard disk, ram and screen configuration separately in the following five label token: label label label label label."
        #valid_text.append(record[1])
        valid_text.append(tmp)
        valid_labels.append(record[2:])
    
    return train_text, valid_text, train_labels, valid_labels

cls_cpu=nn.Linear(in_features=768, out_features=5)
cls_graphic=nn.Linear(in_features=768, out_features=6)
cls_hardisk=nn.Linear(in_features=768, out_features=6)
cls_ram=nn.Linear(in_features=768, out_features=5)
cls_screen=nn.Linear(in_features=768, out_features=8)

cls_cpu.to(device)
cls_graphic.to(device)
cls_hardisk.to(device)
cls_ram.to(device)
cls_screen.to(device)

def extract_at_last_token(hidden_states, mask):
    #print("hidden_states's shape: ",hidden_states.shape)
    #print("mask's shape: ",mask.shape)
    last_token_id = torch.sum(mask, dim=1, keepdim=True)-1
    #print("last_token_id's shape: ",last_token_id.shape)
    #print(torch.sum(mask, dim=1, keepdim=True))
    #print("last_token_id: ",last_token_id)

    shape=hidden_states.shape
    #result=torch.zeros([shape[0],1,shape[2]], device=device)
    result=torch.zeros([shape[0],5,shape[2]], device=device)
    for i in range(shape[0]):
        index_list=[i for i in range(last_token_id[i]-5,last_token_id[i])]
        index_list=torch.tensor(index_list, device=device)
        #print("index list length: ",len(index_list))
        #print("index list: ",index_list)
        #print("last token index: ",last_token_id[i])
        #result[i] = torch.index_select(hidden_states[i], dim=0, index=last_token_id[i])
        result[i] = torch.index_select(hidden_states[i], dim=0, index=index_list)
    #result=result.squeeze(dim=1)
    #print("result'shape: ",result.shape)
    return result

def transform_hiddens_to_logits(raw_outputs, mask):
    #print(raw_outputs.hidden_states[-1].shape)
    last_hidden_state = extract_at_last_token(raw_outputs.hidden_states[-1], mask)

    #print("before transpose, the shape of hidden state is: ",last_hidden_state.shape)
    last_hidden_state = torch.transpose(last_hidden_state, 0, 1)
    #print("after transpose, the shape of hidden state is: ",last_hidden_state.shape)
    
    cpu_output = cls_cpu(last_hidden_state[0])
    graphic_output = cls_graphic(last_hidden_state[1])
    hardisk_output = cls_hardisk(last_hidden_state[2])
    ram_output= cls_ram(last_hidden_state[3])
    screen_output = cls_screen(last_hidden_state[4])
    
    return cpu_output, graphic_output, hardisk_output, ram_output, screen_output

finetune_text, test_text, finetune_labels, test_labels = read_data_csv("./newdata/new_need_all_map.csv",[1149,1148])

max_seq_len = 512

#tokenizer.pad_token = "<|pad|>"
#tokenizer.add_tokens([tokenizer.pad_token])
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# tokenize and encode sequences in the training set
tokens_finetune = tokenizer.batch_encode_plus(
    finetune_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)
"""
print("length of tokens_finetune: ",len(tokens_finetune['input_ids']))
print("length of No0 sample: ",len(tokens_finetune['input_ids'][0]))
print("token sequence of No0 sample: ",tokens_finetune['input_ids'][0])
print("mask sequence of No0 sample: ",tokens_finetune['attention_mask'][0])
print("last token of No0 sample: ", tokens_finetune['input_ids'][0][sum(tokens_finetune['attention_mask'][0])-1])
print("text of No0 sample: ", finetune_text[0])

print("length of No1 sample: ",len(tokens_finetune['input_ids'][1]))
print("token sequence of No1 sample: ",tokens_finetune['input_ids'][1])
print("mask sequence of No1 sample: ",tokens_finetune['attention_mask'][1])
print("last token of No1 sample: ", tokens_finetune['input_ids'][1][sum(tokens_finetune['attention_mask'][1])-1])
print("text of No1 sample: ", finetune_text[1])
"""
# tokenize and encode sequences in the validation set
tokens_test = tokenizer.batch_encode_plus(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
finetune_seq = torch.tensor(tokens_finetune['input_ids'])
finetune_mask = torch.tensor(tokens_finetune['attention_mask'])
finetune_y = torch.tensor(finetune_labels)

# for validation set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels)

batch_size = 16

# wrap tensors
finetune_data = TensorDataset(finetune_seq, finetune_mask, finetune_y)

# sampler for sampling the data during training
finetune_sampler = RandomSampler(finetune_data)

# dataLoader for train set
finetune_dataloader = DataLoader(finetune_data, sampler=finetune_sampler, batch_size=batch_size)

# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)

CEL=nn.CrossEntropyLoss()
trainable_param = [p for p in model.parameters() if p.requires_grad]
trainable_param.extend([p for p in cls_cpu.parameters() if p.requires_grad])
trainable_param.extend([p for p in cls_graphic.parameters() if p.requires_grad])
trainable_param.extend([p for p in cls_hardisk.parameters() if p.requires_grad])
trainable_param.extend([p for p in cls_ram.parameters() if p.requires_grad])
trainable_param.extend([p for p in cls_screen.parameters() if p.requires_grad])

optimizer = AdamW(trainable_param, lr = 5e-5)
print("trainable param number: ", sum([p.numel() for p in trainable_param]))
epoch=5
for i in range(epoch):
    #
    count=0
    loss_rec=0
    model.train()
    #model.print_trainable_parameters()
    for batch in finetune_dataloader:
        batch = [r.to(device) for r in batch]
        inputs, input_mask, labels=batch
        
        cpu_output, graphic_output, hardisk_output,\
        ram_output, screen_output = transform_hiddens_to_logits(model(inputs, attention_mask=input_mask, output_hidden_states=True), input_mask)
        
        label_trans=torch.transpose(labels,0,1)
        cpu_labels=label_trans[0]
        graphic_labels=label_trans[1]
        hard_labels=label_trans[2]
        ram_labels=label_trans[3]
        scre_labels=label_trans[4]
        #print(cpu_output.shape)
        cpu_loss=CEL(cpu_output,cpu_labels)
        graphic_loss=CEL(graphic_output,graphic_labels)
        hard_loss=CEL(hardisk_output,hard_labels)
        ram_loss=CEL(ram_output,ram_labels)
        scre_loss=CEL(screen_output,scre_labels)
        
        loss = cpu_loss+graphic_loss+hard_loss+ram_loss+scre_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count+=1
        loss_rec+=loss
    print("No.",i," epoch loss: ",loss_rec/count)

    if(True):
        with torch.no_grad():
            model.eval()
            cpu_preds=[]
            cpu_labels=[]
            cpu_all_pred=[]
            graphic_preds=[]
            graphic_labels=[]
            graphic_all_pred=[]
            hard_preds=[]
            hard_labels=[]
            hard_all_pred=[]
            ram_preds=[]
            ram_labels=[]
            ram_all_pred=[]
            scre_preds=[]
            scre_labels=[]
            scre_all_pred=[]
            for batch in test_dataloader:
                batch = [r.to(device) for r in batch]
                inputs, input_mask, label=batch
                
                cpu_logits, graphic_logits, hard_logits,\
                ram_logits, scre_logits=transform_hiddens_to_logits(model(inputs, attention_mask=input_mask, output_hidden_states=True), input_mask)

                label_trans=torch.transpose(label,0,1)
                cpu_label=label_trans[0]
                graphic_label=label_trans[1]
                hard_label=label_trans[2]
                ram_label=label_trans[3]
                scre_label=label_trans[4]

                cpu_labels.extend(cpu_label.cpu().tolist())
                cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())
                cpu_all_pred.extend(cpu_logits.cpu().tolist())
                graphic_labels.extend(graphic_label.cpu().tolist())
                graphic_preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
                graphic_all_pred.extend(graphic_logits.cpu().tolist())
                hard_labels.extend(hard_label.cpu().tolist())
                hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())
                hard_all_pred.extend(hard_logits.cpu().tolist())
                ram_labels.extend(ram_label.cpu().tolist())
                ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())
                ram_all_pred.extend(ram_logits.cpu().tolist())
                scre_labels.extend(scre_label.cpu().tolist())
                scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
                scre_all_pred.extend(scre_logits.cpu().tolist())
                
            """acc=sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)"""
            cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
            graphic_acc=sum([int(i==j) for i,j in zip(graphic_preds, graphic_labels)])/len(graphic_preds)
            hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
            ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
            scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)
        
        """print(i," epoch test accuracy is : ",acc)"""
        print(i," epoch test cpu accuracy is : ",cpu_acc)
        print(i," epoch test graphic card accuracy is : ",graphic_acc)
        print(i," epoch test hard disk accuracy is : ",hard_acc)
        print(i," epoch test ram accuracy is : ",ram_acc)
        print(i," epoch test scre accuracy is : ",scre_acc)
print("test success")