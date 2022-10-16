from glob import glob
import os
import copy 
import pandas as pd
import numpy as np

# PRODUCTS-10K
print('Looking for Products-10K...')
pro10K = glob('data/products10k/*')
assert len(pro10K)>0, 'Products-10K not found, make sure it is placed at data/products10k.'
os.system('cp -r data/products10k data_preprocess/')
print('Products-10K is successfully processed!')



# SHOPEE
print('Looking for Shopee...')
shopee10K = glob('data/shopee/*')
assert len(shopee10K)>0, 'Shopee not found, make sure it is placed at data/shopee.'

def make_label():
    # make the dictionary to project from label_group to label
    for ind, label_group in zip(range(classes_num,),classes_unique):
        labelgroup2label[label_group] = ind

def trans_labelgroup2label():
    # transport label2group to label
    for i in range(len(label_group_list)):
        label_group_list[i] = labelgroup2label[label_group_list[i]]
        
def make_img_full_path():
    # make the img_list full path
    for i in range(len(image_list)):
        image_list[i] = os.path.join('data_preprocess/shopee/train_images',image_list[i])
        
csv = pd.read_csv(os.path.join('data/shopee','train.csv'))
classes_unique = csv['label_group'].unique()
classes_num = len(classes_unique)

# here we make a dictionary to project from label_group to label
labelgroup2label = {}
make_label()

image_list = csv['image'].to_list()
label_group_list = csv['label_group'].to_list()

trans_labelgroup2label()
make_img_full_path()

label_list = label_group_list
del(label_group_list)
os.system('cp -r data/shopee data_preprocess/')
np.save('data_preprocess/shopee/image_list.npy', image_list)
np.save('data_preprocess/shopee/label_list.npy', label_list)

print('Shopee is successfully processed!')

# MET

print('Looking for MET...')
met = glob('data/MET/*')
assert len(met)>0, 'MET not found, make sure it is placed at data/MET.'
os.system('cp -r data/MET data_preprocess/')
print('MET is successfully processed!')

# Alibaba goods

print('Looking for Alibaba Goods...')
aliproduct = glob('data/Alibaba Goods/*')
assert len(aliproduct)>0, 'Alibaba Goods not found, make sure it is placed at data/Alibaba Goods.'
os.system('cp -r data/Alibaba Goods data_preprocess/')
print('Alibaba Goods is successfully processed!')

# H&M Personalized Fashion


hm = glob('data/hm/*')
assert len(hm)>0, 'H&M Personalized Fashion not found, make sure it is placed at data/hm.'

train_csv  = pd.read_csv('data/hm/articles.csv')
product_code_unique = train_csv.product_code.unique()
mapping_table = {}
for n, i in enumerate(product_code_unique):
    mapping_table[i] = n
train_csv.product_code = train_csv.product_code.map(lambda x: mapping_table[x])
train_csv['avaliable'] = train_csv.article_id.map(lambda x: os.path.exists('data/hm/images/' + '0' + str(x)[:2] +'/'+ '0' + str(x) + '.jpg' ) )
train_csv.drop(train_csv[train_csv.avaliable == False].index, inplace = True)

os.system('cp -r data/hm data_preprocess/')
train_csv.reset_index().to_csv('data_preprocess/hm/articles_post.csv',index=False)

print('H&M Personalized Fashion is successfully processed!')

# GPR1200

print('Looking for GPR1200...')

gpr = glob('data/gpr1200/*')
assert len(gpr)>0, 'GPR1200 not found, make sure it is placed at data/gpr1200.'
os.system('cp -r data/gpr1200 data_preprocess/')
print('GPR1200 is successfully processed!')

# GLD-V2-FULL

print('Looking for GLD-V2-FULL...')

gldv2full = glob('data/gldv2full/*')
assert len(gldv2full)>0, 'GLD-V2-FULL not found, make sure it is placed at data/gldv2full.'
os.system('cp -r data/gldv2full data_preprocess/')
print('GLD-V2-FULL is successfully processed!')

# DeepFashion - Consumer-to-shop Clothes Retrieval Benchmark

print('Looking for DeepFashion...')

DeepFashion = glob('data/DeepFashion/*')
assert len(DeepFashion)>0, 'DeepFashion not found, make sure it is placed at data/DeepFashion.'
os.system('cp -r data/DeepFashion data_preprocess/')
print('DeepFashion is successfully processed!')

print('ALL DATASETS ARE DONE PREPROCESSED!')




