import os
import re
import numpy as np

######################
##Preparing the data##
######################

train_dir="../datasets/train"
test_dir="../datasets/test"
train_file_names=os.listdir(train_dir)
test_file_names=os.listdir(test_dir)

train_G=[0]*2000
train_Y=[0]*2000
for file_name in train_file_names:
    if re.match(".*_graph.txt",file_name)!=None:
        file_g=open(train_dir+"/"+file_name,"r")
        #we don't use the first row because it's not matrix
        matrix=file_g.readlines()[1:]
        #change the matrix from string to int
        train_g=[[int(i) for i in elm.strip().replace(" ","")] for elm in matrix]
        #extract the id from the file name
        m=re.search("(\d*)_graph.txt",file_name)
        index=int(m.group(1))
        #save the matrix to the same index as id 
        train_G[index]=np.array(train_g)
    else:
        file_y=open(train_dir+"/"+file_name,"r")
        vec=file_y.readlines()
        #change the label from string to int
        train_y=[int(i.strip()) for i in vec] 
        #extract the id from the file name
        m=re.search("(\d*)_label.txt",file_name)
        index=int(m.group(1))
        #save the label to the same index as id 
        train_Y[index]=np.array(train_y)

train_G=np.array(train_G)
train_Y=np.array(train_Y)

test_G=[0]*500
for file_name in test_file_names:
    file_g=open(test_dir+"/"+file_name,"r")
    #we don't use the first row because it's not matrix
    matrix=file_g.readlines()[1:]
    #change the matrix from string to int
    test_g=[[int(i) for i in elm.strip().replace(" ","")] for elm in matrix]
    #extract the id from the file name
    m=re.search("(\d*)_graph.txt",file_name)
    index=int(m.group(1))
    #save the matrix to the same index as id 
    test_G[index]=np.array(test_g)

test_G=np.array(test_G)

#divide the training data into training data and validation data
train_g=train_G[0:1600]
train_y=train_Y[0:1600]
val_g=train_G[1600:2000]
val_y=train_Y[1600:2000]

np.save('../datasets/train_g.npy', train_g)
np.save("../datasets/train_y.npy",train_y)
np.save("../datasets/val_g.npy",val_g)
np.save("../datasets/val_y.npy",val_y)
np.save("../datasets/test_G.npy",test_G)