import os
import gc
import torch
import logging
import argparse
from model.ctrNet import ctrNet 
import pickle
import gensim
import random
import pandas as pd
import numpy as np
from src.data_loader import TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--parse', type=int, default=1)  
    args = parser.parse_args()

    parser.add_argument('--kfold', type=int, default=20)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--max_len_text', type=int, default=128)
    parser.add_argument('--dropout_prob', type=float, default=0.2)
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--display_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # parser.add_argument('--num_hidden_layers', type=int, default=6)
    # parser.add_argument('--output_path', type=str, default=None)
    # parser.add_argument('--pretrained_model_path', type=str, default=None)
    # parser.add_argument('--hidden_size', type=int, default=1024)
    # parser.add_argument('--vocab_size_v1', type=int, default=500000)
    # parser.add_argument('--vocab_dim_v1', type=int, default=64)
    # parser.add_argument('--num_label', type=int, default=20)    

    #定义用户点击的序列特征
    text_features=[
        "sequence_text_user_id_creative_id",
        "sequence_text_user_id_ad_id",
        "sequence_text_user_id_advertiser_id",
        "sequence_text_user_id_product_id",
        "sequence_text_user_id_industry",
        "sequence_text_user_id_product_category"
    ]

    #定义浮点数特征
    dense_features=['user_id__size', 
        'user_id_ad_id_unique', 
        'user_id_creative_id_unique', 
        'user_id_advertiser_id_unique', 
        'user_id_industry_unique', 
        'user_id_product_id_unique', 
        'user_id_time_unique', 
        'user_id_click_times_sum', 
        'user_id_click_times_mean', 
        'user_id_click_times_std']
    # for l in ['age_{}'.format(i) for i in range(10)]+['gender_{}'.format(i) for i in range(2)]:
    #     for f in ['creative_id','ad_id','product_id','advertiser_id','industry']:  
    #         dense_features.append(l+'_'+f+'_mean')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #设置参数
    logger.info("Argument %s", args)    
    args.text_features=text_features
    args.dense_features=dense_features 
    # args.text_dim=sum([x[-1] for x in text_features])
    args.output_dir="saved_models/index_{}".format(args.index) 


    # base_path="data"
    base_path="dev_data"

    min_count = 5 
    method = "sg" 
    window = 50
    shuffle_time = 0 
    
    args.embedding_dir="embeddings_{}_win{}_mincount{}_shuffle{}".format(base_path,window,min_count,shuffle_time)
    args.lr= 1e-3
    

    #读取数据       
    train_df=pd.read_pickle(base_path+'/train_user.pkl')
    test_df=pd.read_pickle(base_path+'/test_user.pkl')
    df=train_df[args.dense_features].append(test_df[args.dense_features])
    ss=StandardScaler()
    ss.fit(df[args.dense_features])
    train_df[args.dense_features]=ss.transform(train_df[args.dense_features])
    test_df[args.dense_features]=ss.transform(test_df[args.dense_features])
    test_dataset = TextDataset(args,test_df)   
    del df 
    gc.collect() 
    
    #建立模型
    skf=StratifiedKFold(n_splits=args.kfold,random_state=2020,shuffle=True)
    model=ctrNet(args) 
    
    #训练模型
    for i,(train_index,test_index) in enumerate(skf.split(train_df,train_df['age'])):
        if i!=args.index:
            continue
        logger.info("Index: %s",args.index)
        train_dataset = TextDataset(args,train_df.iloc[train_index])
        dev_dataset=TextDataset(args,train_df.iloc[test_index])
        model.train(train_dataset,dev_dataset)
        dev_df=train_df.iloc[test_index]
    
    # #输出结果
    # accs=[]
    # for f,num in [('age',10),('gender',2)]:
    #     model.reload(f)
    #     if f=="age":
    #         dev_preds=model.infer(dev_dataset)[0]
    #     else:
    #         dev_preds=model.infer(dev_dataset)[1]
    #     for j in range(num):
    #         dev_df['{}_{}'.format(f,j)]=np.round(dev_preds[:,j],4)
    #     acc=model.eval(dev_df[f].values,dev_preds)['eval_acc']
    #     accs.append(acc)
    #     if f=="age":
    #         test_preds=model.infer(test_dataset)[0]
    #     else:
    #         test_preds=model.infer(test_dataset)[1]

    #     logger.info("Test %s %s",f,np.mean(test_preds,0))
    #     logger.info("ACC %s %s",f,round(acc,5))

    #     out_fs=['user_id','age','gender','predict_{}'.format(f)]
    #     out_fs+=['{}_{}'.format(f,i) for i in range(num)]
    #     for i in range(num):
    #         test_df['{}_{}'.format(f,i)]=np.round(test_preds[:,i],4)
    #     test_df['predict_{}'.format(f)]=np.argmax(test_preds,-1)+1
    #     try:
    #         os.system("mkdir submission")
    #     except:
    #         pass

    #     test_df[out_fs].to_csv('submission/submission_test_{}_{}_{}.csv'.format(f,args.index,round(acc,5)),index=False)
        
    # logger.info("  best_acc = %s",round(sum(accs),4))