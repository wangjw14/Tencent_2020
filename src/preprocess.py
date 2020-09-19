import pandas as pd 
import numpy as np
import gc 
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
base = "raw_data/"
###################################################################################
logger.info("merge ads...")
# 读取数据
ads_df = pd.read_csv(base+"train_preliminary/ad.csv")
ads_df.append(pd.read_csv(base+"test/ad.csv"))

# 去重
ads_df = ads_df.drop_duplicates("creative_id")
ads_df.reset_index(drop=True,inplace=True)

# print(ads_df.info())


###################################################################################
logger.info("merge log...")
# 读取数据
log_df = pd.read_csv(base+"train_preliminary/click_log.csv")
log_df.append(pd.read_csv(base+"test/click_log.csv"))

# 按照时间顺序排序
log_df.sort_values("time",inplace=True ) 
log_df.reset_index(drop=True,inplace=True) 

# print(log_df.info())

###################################################################################
logger.info("get user...") 

# 读取数据
train = pd.read_csv(base+"train_preliminary/user.csv")
test  = pd.read_csv(base+"test/user.csv") 

# 转换为从0开始的标签
train['age'] = train["age"].apply(lambda x:x-1).astype(np.int16) 
train['gender'] = train["gender"].apply(lambda x:x-1).astype(np.int16) 
test['age'] = -1 
test['gender'] = -1
test['age'] = test['age'].astype(np.int16) 
test['gender'] = test["gender"].astype(np.int16) 

# print(train.info())
# print(test.info())


###################################################################################
logger.info("merge all...")

user = pd.concat([train, test])
user.reset_index(drop=True,inplace=True) 

log_df = log_df.merge(ads_df,on="creative_id",how='left')
log_df = log_df.merge(user,on="user_id",how='left')
del user 
gc.collect()
log_df.fillna(-1,inplace=True)
log_df.replace("\\N",-1,inplace=True)
for col in log_df: 
    log_df[col] = log_df[col].astype(int)

for i in range(10):
    log_df["age_{}".format(i)] = log_df["age"].apply(lambda x:x==i).astype(np.int16) 
for i in range(2):
    log_df["gender_{}".format(i)] = log_df["gender"].apply(lambda x:x==i).astype(np.int16)


###################################################################################
logger.info("save data to disk...")
save_dir = "data/"
log_df.to_pickle(save_dir+"log.pkl") 
train.to_pickle(save_dir+"train.pkl")
test.to_pickle(save_dir+"test.pkl") 
