import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import logging
import random
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def w2v(dfs,feat,L=128):
    print("w2v",feat)

    output_dir = "embeddings_{}_win{}_mincount{}_shuffle{}".format(base,window,min_count,shuffle_time)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir) 

    sentences=[]
    for df in dfs:
        for line in df[feat].values:
            sentences.append(line) 
            for i in range(shuffle_time):
                random.shuffle(line)
                sentences.append(line) 
    
    print("Sentence Num {}".format(len(sentences)))
    model=Word2Vec(sentences,size=L, window=window,min_count=min_count,sg=1,workers=32,iter=10)
    
    print("save w2v to {}".format(output_dir)) 
    vocab = model.wv.index2word 
    vocab_dict ={}
    for i,v in enumerate(vocab):
        vocab_dict[v] = i 
    # vocab_dict[PAD] = len(vocab)

    with open("{}/{}_vocab.pkl".format(output_dir,feat),"wb") as file: 
        pickle.dump(vocab_dict,file) 
    
    vectors_path = "{}/{}_vectors.npy".format(output_dir,feat) 
    # if not os.path.exists(vectors_path):
    with open(vectors_path,"wb") as file:
        np.save(file,model.wv.vectors)



if __name__ == "__main__": 
    base = "data"
    # base = "dev_data"
    min_count = 5 
    method = "sg" 
    window = 50
    shuffle_time = 2

    logger.info("base: {}".format(base)) 
    logger.info("method: {}".format(method)) 
    logger.info("window: {}".format(window)) 
    logger.info("min_count: {}".format(min_count)) 
    logger.info("shuffle_time: {}".format(shuffle_time)) 

    
    train_df=pd.read_pickle(base+'/train_user.pkl')
    test_df=pd.read_pickle(base+'/test_user.pkl')
    #训练word2vector，维度为128
    w2v([train_df,test_df],'sequence_text_user_id_ad_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_creative_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_advertiser_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_product_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_industry',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_product_category',L=128)
    # w2v([train_df,test_df],'sequence_text_user_id_time',L=128)
    # w2v([train_df,test_df],'sequence_text_user_id_click_times',L=128)


