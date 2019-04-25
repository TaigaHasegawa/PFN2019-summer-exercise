#import necesarry package
import numpy as np
import os
import re
import matplotlib.pyplot as plt

######################################
######################################
############# Question1 ##############
######################################
######################################

#function to make Adjacency matrix
def make_symmetric_matrix(arr,shape):
    """
    arr: arr which specify where is non-zero
    shape: shape of the symmetric matrix
    """
    G=np.zeros((shape))
    for i in arr:
        G[i[0]][i[1]]=1
        G[i[1]][i[0]]=G[i[0]][i[1]]
    return G
    
#Adjacency matrix for test
G=make_symmetric_matrix(np.array([[1,0],[2,1],[3,2],[3,4],[1,3],[4,2],[4,5],[6,4],[6,3],[6,5],[7,2],[7,0],[7,3],[8,0],[8,5],[8,6],[9,2],[9,8]]),(10,10))
#feature vector when D=4
X=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])

#dimension of feature vector 
D=4
#set the seed
np.random.seed(2018)
#initialize D*D size weight parameter 
W=np.random.normal(0,0.4,(D,D))

#relu function 
def relu(x):
    y = np.maximum(0, x)
    return y

#return feature vector h
def GNN_basic(G,X,t):
    """
    G: Adjacency matrix
    X: feature vector 
    t: the number of steps
    """
    for i in range(t):
        a=np.dot(G,X)
        X=relu(np.dot(a,W))
    h=np.sum(X,axis=0)
    return h

#toy example
h=GNN_basic(G,X,t=2)

#show the result of toy example 
h
#array([ 0., 31.56446968, 22.3687154 ,  0.])


######################################
######################################
############# Question2 ##############
######################################
######################################


#sigmoid function
def sigmoid(x):
    if x<-10:
        y=0
    else:
        y=1/(1+np.exp(-x))
    return y

###########
##  GNN  ##
###########

#return predicted label and calculate loss 
def GNN(G,X,W,A,b,t,y):
    """
    G: Adjacency matrix
    X: feature vector 
    W: weight parameter     
    A: weight parameter 
    b: bias 
    t: the number of steps
    y: ground truth
    """
    for i in range(t):
        a=np.dot(G,X)
        X=relu(np.dot(a,W))
    h=np.sum(X,axis=0)
    s=np.dot(A,h)+b
    #probability
    p=sigmoid(s)
    #if it's over 1/2, it returns 1. otherwise it returns 0.
    y_hat=1 if p>1/2 else 0
    #binary cross-entropy loss
    #use approximation to avoid overflaw
    if s>10:
        loss=(1-y)*s+y*np.log(1+np.exp(-s))
    elif s<-10:
        loss=y*s+(1-y)*np.log(1+np.exp(s))
    else:
        loss=y*np.log(1+np.exp(-s))+(1-y)*np.log(1+np.exp(s))
    return y_hat,loss

#dimension of feature vector
D=4
#feature vector when D=4
X=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
#set the seed 
np.random.seed(2013)
#initialize D size weight parameter A
A=np.random.uniform(0,0.4,D)
#initialize bias
b=0
#initialize D*D size weight parameter W
W=np.random.uniform(0,0.4,(D,D))

#prediction and loss without gradient descent 
y_hat,loss=GNN(G,X,W,A,b,t=5,y=0)

print(y_hat,loss)
#1 741.1999772981253

#learning rate
alpha=0.01

######################
## Gradient Descent ##
######################

#20000 iterations 
for i in range(20001):
    _,loss_1=GNN(G,X,W,A,b,t=5,y=0)
    _,loss_2=GNN(G,X,W,A,b+0.001,t=5,y=0)
    #gradient of b
    gra_b=(loss_2-loss_1)/0.001
    gra_A=np.zeros_like(A)
    gra_W=np.zeros_like(W)
    for j in range(D):
        delta=np.zeros(D)
        delta[j]=1
        _,loss_3=GNN(G,X,W,A+0.001*delta,b,t=5,y=0)
        #gradient of A
        gra_A[j]=(loss_3-loss_1)/0.001
        for k in range(D):
            delta=np.zeros((D,D))
            delta[j,k]=1
            _,loss_4=GNN(G,X,W+0.001*delta,A,b,t=5,y=0)
            #gradient of W
            gra_W[j,k]=(loss_4-loss_1)/0.001
    #update
    b=b-gra_b*alpha
    A=A-gra_A*alpha
    W=W-gra_W*alpha
    #show the prediction and loss every 1000 iteration  
    if i%1000==0:
        y_hat,loss=GNN(G,X,W,A,b,t=5,y=0)
        print(y_hat,loss)


######################################
######################################
############# Question3 ##############
######################################
######################################


#################################
## Stochastic gradient descent ##
#################################

def Stochastic_gradient_descent(D, W, A, b, t, train, alpha):
    """
    D: dimension of feature vector 
    W: weight parameter     
    A: weight parameter 
    b: bias 
    t: the number of steps
    train: train_loader
    alpha: learning rate
    """
    for g_batch, y_batch in train:
        delta_b=[]
        delta_A=[]
        delta_W=[]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        for i,g in enumerate(g_batch):
            dim=len(g)
            #make the feature vec of V (vertex)
            X=np.array([feature_vec for i in range(dim)])
            _,loss_1=GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=y_batch[i])
            _,loss_2=GNN(G=g,X=X,W=W,A=A,b=b+0.001,t=t,y=y_batch[i])
            #gradient of b
            gra_b=(loss_2-loss_1)/0.001
            #save the gradient b to delta_b
            delta_b.append(gra_b)
            gra_A=np.zeros_like(A)
            gra_W=np.zeros_like(W)
            for j in range(D):
                delta=np.zeros(D)
                delta[j]=1
                _,loss_3=GNN(g,X,W,A+0.001*delta,b,t,y=y_batch[i])
                #gradient of A
                gra_A[j]=(loss_3-loss_1)/0.001
                for k in range(D):
                    delta=np.zeros((D,D))
                    delta[j,k]=1
                    _,loss_4=GNN(g,X,W+0.001*delta,A,b,t,y=y_batch[i])
                    #gradient of W
                    gra_W[j,k]=(loss_4-loss_1)/0.001
            #save the gradient A to delta_A
            delta_A.append(gra_A)
            #save the gradient W to delta_W
            delta_W.append(gra_W)
        #take the mean of all gradients saved in the delta_b, delta_A and delta_W
        #this is the stochastic gradient 
        delta_b=np.mean(delta_b,axis=0)
        delta_A=np.mean(delta_A,axis=0)
        delta_W=np.mean(delta_W,axis=0)
        #update
        b=b-delta_b*alpha
        A=A-delta_A*alpha
        W=W-delta_W*alpha
    return b,A,W


############################################
### Momentum stochastic gradient descent ###
############################################

def momentum_stochastic_gradient_descent(D, W, A, b, t, train, alpha, eta):
    """
    D: dimension of feature vector 
    W: weight parameter     
    A: weight parameter 
    b: bias 
    t: the number of steps
    train: train_loader
    alpha: learning rate
    eta: decay rates for the moment estimates
    """
    #initialize the momentum 
    omega_b=0
    omega_A=np.zeros(D)
    omega_W=np.zeros((D,D))
    for g_batch, y_batch in train:  
        delta_b=[]
        delta_A=[]
        delta_W=[]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        for i,g in enumerate(g_batch):
            dim=len(g)
            #first make the feature vec of V (vertex)
            X=np.array([feature_vec for i in range(dim)])
            _,loss_1=GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=y_batch[i])
            _,loss_2=GNN(G=g,X=X,W=W,A=A,b=b+0.001,t=t,y=y_batch[i])
            #gradient of b
            gra_b=(loss_2-loss_1)/0.001
            #save the gradient b to delta_b
            delta_b.append(gra_b)
            gra_A=np.zeros(D)
            gra_W=np.zeros((D,D))
            for j in range(D):
                delta=np.zeros(D)
                delta[j]=1
                _,loss_3=GNN(g,X,W,A+0.001*delta,b,t,y=y_batch[i])
                #gradient of A
                gra_A[j]=(loss_3-loss_1)/0.001
                for k in range(D):
                    delta=np.zeros((D,D))
                    delta[j,k]=1
                    _,loss_4=GNN(g,X,W+0.001*delta,A,b,t,y=y_batch[i])
                    #gradient of W
                    gra_W[j,k]=(loss_4-loss_1)/0.001
            #save the gradient A to delta_A
            delta_A.append(gra_A)
            #save the gradient W to delta_W
            delta_W.append(gra_W)
        #take the mean of all gradients saved in the delta_b, delta_A and delta_W
        #this is the stochastic gradient 
        delta_b=np.mean(delta_b,axis=0)
        delta_A=np.mean(delta_A,axis=0)
        delta_W=np.mean(delta_W,axis=0)
        #update
        b=b-delta_b*alpha+eta*omega_b
        A=A-delta_A*alpha+eta*omega_A
        W=W-delta_W*alpha+eta*omega_W
        #update momentum
        omega_b=-delta_b*alpha+eta*omega_b
        omega_A=-delta_A*alpha+eta*omega_A
        omega_W=-delta_W*alpha+eta*omega_W
    return b,A,W


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


###############################################
## training with stochastic gradient descent ##
###############################################

#hyper parameter
alpha=0.01
D=8
t=2
batch_size=20
num_epoch=100

#initialize parameter 
np.random.seed(100)
W=np.random.normal(0,0.4,(D,D))
A=np.random.normal(0,0.4,D)
b=0

#list for saving the mean accuracy and loss of training and validation data
acc_for_train_SGD=[]
acc_for_val_SGD=[]
loss_for_train_SGD=[]
loss_for_val_SGD=[]

#train the model with stochastic gradient descent 
for epoch in range(num_epoch):
    
    ##make the mini batch##
    train_loader = []
    N=len(train_g)
    #shuffle the index 
    perm = np.random.permutation(N)
    #split the index 
    for i in range(0, N, batch_size):
        G_mini = train_g[perm[i:i + batch_size]]
        Y_mini = train_y[perm[i:i + batch_size]]
        train_loader.append((G_mini, Y_mini)) 

    #stochastic gradient descent 
    b,A,W=Stochastic_gradient_descent(D=D,W=W,A=A,b=b,t=t,train=train_loader,alpha=alpha)
    
    #list for saving mean loss and accuracy of training and validation data
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]
    
    #make feature vector 
    feature_vec=np.zeros(D)
    feature_vec[0]=1

    #calculate the mean loss and accuracy of each training data
    for i,g in enumerate(train_g):
        dim=g.shape[0]
        #make the feature vec of V (vertex)
        X=[feature_vec for j in range(dim)]
        train_loss.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=train_y[i])[1])
        train_acc.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=train_y[i])[0]==train_y[i])
    #calculate the mean loss and accuracy of each validation data
    for i,g in enumerate(val_g):
        dim=g.shape[0]
        #make the feature vec of V (vertex)
        X=[feature_vec for j in range(dim)]
        val_loss.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=val_y[i])[1])
        val_acc.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=val_y[i])[0]==val_y[i])
    
    #get the mean loss and accuracy of training data and validation data
    train_loss=np.mean(train_loss)
    train_acc=np.mean(train_acc)
    val_loss=np.mean(val_loss)
    val_acc=np.mean(val_acc)


    print("Train:","loss","%.2f" % train_loss,"accuracy","%.2f"%(train_acc*100))
    print("Validation:","loss","%.2f" % val_loss,"accuracy","%.2f"%(val_acc*100))

    #save the mean loss and accuracy to the list so that we can use later 
    acc_for_train_SGD.append(train_acc)
    acc_for_val_SGD.append(val_acc)
    loss_for_train_SGD.append(train_loss)
    loss_for_val_SGD.append(val_loss)


########################################################
## training with momentum stochastic gradient descent ##
########################################################

#hyper parameter
alpha=0.001
eta=0.9
D=8
t=2
batch_size=20
num_epoch=100

#initialize parameter 
np.random.seed(100)
W=np.random.normal(0,0.4,(D,D))
A=np.random.normal(0,0.4,D)
b=0

#list for saving the mean accuracy and loss of training and validation data
acc_for_train_MSGD=[]
acc_for_val_MSGD=[]
loss_for_train_MSGD=[]
loss_for_val_MSGD=[]

#train the model with momentum stochastic gradient descent 
for epoch in range(num_epoch):

    #make mini batch data
    train_loader = []
    N=len(train_g)
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        G_mini = train_g[perm[i:i + batch_size]]
        Y_mini = train_y[perm[i:i + batch_size]]
        train_loader.append((G_mini, Y_mini)) 

    #momentum stochastic gradient descent 
    b,A,W=momentum_stochastic_gradient_descent(D=D,W=W,A=A,b=b,t=t,train=train_loader,alpha=alpha,eta=eta)

    #list for saving mean loss and accuracy of training and validation data
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    #calculate the loss and accuracy of each training data 
    for i,g in enumerate(train_g):
        dim=g.shape[0]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        #make the feature vec of V (vertex)
        X=[feature_vec for j in range(dim)]
        train_loss.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=train_y[i])[1])
        train_acc.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=train_y[i])[0]==train_y[i])

    #calculate the loss and accuracy of each validation data
    for i,g in enumerate(val_g):
        dim=g.shape[0]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        #make the feature vec of V (vertex)
        X=[feature_vec for j in range(dim)]
        val_loss.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=val_y[i])[1])
        val_acc.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=val_y[i])[0]==val_y[i])

    #get the mean loss and accuracy of training data and validation data
    train_loss=np.mean(train_loss)
    train_acc=np.mean(train_acc)
    val_loss=np.mean(val_loss)
    val_acc=np.mean(val_acc)

    print("Train:","loss","%.2f" % train_loss,"accuracy","%.2f"%(train_acc*100))
    print("Validation:","loss","%.2f" % val_loss,"accuracy","%.2f"%(val_acc*100))

    #save the mean loss and accuracy to the list so that we can use later 
    acc_for_train_MSGD.append(train_acc)
    acc_for_val_MSGD.append(val_acc)
    loss_for_train_MSGD.append(train_loss)
    loss_for_val_MSGD.append(val_loss)


##############################################
## Plot the mean loss and accuracy vs epoch ##
##############################################

#epoch vs mean accuracy 
plt.plot(range(100),acc_for_train_SGD,"r",label="train")
plt.plot(range(100),acc_for_val_SGD,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.savefig('epoch_vs_accuracy_SGD.png')
plt.show()

#epoch vs mean loss 
plt.plot(range(100),loss_for_train_SGD,"r",label="train")
plt.plot(range(100),loss_for_val_SGD,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.savefig('epoch_vs_loss_SGD.png')
plt.show()

#epoch vs mean accuracy 
plt.plot(range(100),acc_for_train_MSGD,"r",label="train")
plt.plot(range(100),acc_for_val_MSGD,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.savefig('epoch_vs_accuracy_MSGD.png')
plt.show()

#epoch vs mean loss
plt.plot(range(100),loss_for_train_MSGD,"r",label="train")
plt.plot(range(100),loss_for_val_MSGD,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.savefig('epoch_vs_loss_MSGD.png')
plt.show()

######################################
######################################
############# Question4 ##############
######################################
######################################


####################
####### Adam #######
####################

def Adam(D, W, A, b, t, train, alpha, beta_1=0.9, beta_2=0.999):
    """
    D: dimension of feature vector 
    W: weight parameter     
    A: weight parameter 
    b: bias 
    t: the number of steps
    train: train_loader
    alpha: learning rate
    beta_1: Exponential decay rates for the 1st moment estimates
    beta_2: Exponential decay rates for the 2nd moment estimates
    """
    #Initialize 1st moment vector
    s_b=0 
    s_A=np.zeros(D)
    s_W=np.zeros((D,D))
    #Initialize 2nd moment vector
    r_b=0 
    r_A=np.zeros(D)
    r_W=np.zeros((D,D))
    for itr, (g_batch, y_batch) in enumerate(train):
        itr+=1
        delta_b=[]
        delta_A=[]
        delta_W=[]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        for i,g in enumerate(g_batch):
            dim=len(g)
            #make the feature vec of V (vertex)
            X=np.array([feature_vec for i in range(dim)])
            _,loss_1=GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=y_batch[i])
            _,loss_2=GNN(G=g,X=X,W=W,A=A,b=b+0.001,t=t,y=y_batch[i])
            #gradient of b
            gra_b=(loss_2-loss_1)/0.001
            #save the gradient b to delta_b
            delta_b.append(gra_b)
            gra_A=np.zeros(D)
            gra_W=np.zeros((D,D))
            for j in range(D):
                delta=np.zeros(D)
                delta[j]=1
                _,loss_3=GNN(g,X,W,A+0.001*delta,b,t,y=y_batch[i])
                #gradient of A
                gra_A[j]=(loss_3-loss_1)/0.001
                for k in range(D):
                    delta=np.zeros((D,D))
                    delta[j,k]=1
                    _,loss_4=GNN(g,X,W+0.001*delta,A,b,t,y=y_batch[i])
                    #gradient of W
                    gra_W[j,k]=(loss_4-loss_1)/0.001
            #save the gradient A to delta_A
            delta_A.append(gra_A)
            #save the gradient W to delta_W
            delta_W.append(gra_W)
        #take the mean of all gradients saved in the delta_b, delta_A and delta_W
        #this is the stochastic gradient 
        delta_b=np.mean(delta_b,axis=0)
        delta_A=np.mean(delta_A,axis=0)
        delta_W=np.mean(delta_W,axis=0)
        #update 1st moment
        s_b=beta_1*s_b+(1-beta_1)*delta_b
        s_A=beta_1*s_A+(1-beta_1)*delta_A
        s_W=beta_1*s_W+(1-beta_1)*delta_W
        #update 2nd moment 
        r_b=beta_2*r_b+(1-beta_2)*delta_b**2
        r_A=beta_2*r_A+(1-beta_2)*delta_A**2
        r_W=beta_2*r_W+(1-beta_2)*delta_W**2
        #bias-corrected first moment 
        s_b_hat=s_b/(1-beta_1**itr)
        s_A_hat=s_A/(1-beta_1**itr)
        s_W_hat=s_W/(1-beta_1**itr)
        #bias-corrected second raw moment
        r_b_hat=r_b/(1-beta_2**itr)
        r_A_hat=r_A/(1-beta_2**itr)
        r_W_hat=r_W/(1-beta_2**itr)
        #update
        b=b-alpha*s_b_hat/(0.0001+np.sqrt(r_b_hat))
        A=A-alpha*s_A_hat/(0.0001+np.sqrt(r_A_hat))
        W=W-alpha*s_W_hat/(0.0001+np.sqrt(r_W_hat))
    return b,A,W



#########################################
########## Training with Adam ###########
#########################################

#hyper parameter
alpha=0.001
D=8
t=2
batch_size=20
num_epoch=100

#initialize parameter 
np.random.seed(100)
W=np.random.normal(0,0.4,(D,D))
A=np.random.normal(0,0.4,D)
b=0

#list for saving the mean accuracy and loss of training and validation data
acc_for_train_Adam=[]
acc_for_val_Adam=[]
loss_for_train_Adam=[]
loss_for_val_Adam=[]

#train the model with Adam
for epoch in range(num_epoch):

    #make mini batch data
    train_loader = []
    N=len(train_g)
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        G_mini = train_g[perm[i:i + batch_size]]
        Y_mini = train_y[perm[i:i + batch_size]]
        train_loader.append((G_mini, Y_mini)) 

    #momentum stochastic gradient descent 
    b,A,W=Adam(D=D,W=W,A=A,b=b,t=t,train=train_loader,alpha=alpha)

    #list for saving mean loss and accuracy of training and validation data
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    #calculate the loss and accuracy of each training data 
    for i,g in enumerate(train_g):
        dim=g.shape[0]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        #make the feature vec of V (vertex)
        X=[feature_vec for j in range(dim)]
        train_loss.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=train_y[i])[1])
        train_acc.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=train_y[i])[0]==train_y[i])

    #calculate the loss and accuracy of each validation data
    for i,g in enumerate(val_g):
        dim=g.shape[0]
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        #make the feature vec of V (vertex)
        X=[feature_vec for j in range(dim)]
        val_loss.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=val_y[i])[1])
        val_acc.append(GNN(G=g,X=X,W=W,A=A,b=b,t=t,y=val_y[i])[0]==val_y[i])

    #get the mean loss and accuracy of training data and validation data
    train_loss=np.mean(train_loss)
    train_acc=np.mean(train_acc)
    val_loss=np.mean(val_loss)
    val_acc=np.mean(val_acc)

    print("Train:","loss","%.2f" % train_loss,"accuracy","%.2f"%(train_acc*100))
    print("Validation:","loss","%.2f" % val_loss,"accuracy","%.2f"%(val_acc*100))

    #save the mean loss and accuracy to the list so that we can use later 
    acc_for_train_Adam.append(train_acc)
    acc_for_val_Adam.append(val_acc)
    loss_for_train_Adam.append(train_loss)
    loss_for_val_Adam.append(val_loss)

##############################################
## Plot the mean loss and accuracy vs epoch ##
##############################################

#epoch vs mean accuracy 
plt.plot(range(100),acc_for_train_Adam,"r",label="train")
plt.plot(range(100),acc_for_val_Adam,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.savefig('epoch_vs_accuracy_Adam.png')
plt.show()

#epoch vs mean loss 
plt.plot(range(100),loss_for_train_Adam,"r",label="train")
plt.plot(range(100),loss_for_val_Adam,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.savefig('epoch_vs_loss_Adam.png')
plt.show()



######################################
### Predict the label of test data ###
######################################

#use the weight trained with Adam
result=[]
for G in test_G:
    dim=G.shape[0]
    feature_vec=np.zeros(D)
    feature_vec[0]=1
    #make the feature vec of V (vertex)
    X=[feature_vec for j in range(dim)]
    for i in range(2):
        a=np.dot(G,X)
        X=relu(np.dot(a,W))
    h=np.sum(X,axis=0)
    s=np.dot(A,h)+b
    #probability
    p=sigmoid(s)
    #if it's over 1/2, it returns 1. otherwise it returns 0.
    y_hat="1" if p>1/2 else "0"
    result.append(y_hat)


path_w = "../prediction.txt"
with open(path_w, mode='w') as f:
    f.write('\n'.join(result))
