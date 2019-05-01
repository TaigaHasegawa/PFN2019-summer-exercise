from Question2 import GNN
import numpy as np
import matplotlib.pyplot as plt

train_g=np.load("../datasets/train_g.npy")
train_y=np.load("../datasets/train_y.npy")
val_g=np.load("../datasets/val_g.npy")
val_y=np.load("../datasets/val_y.npy")
test_G=np.load("../datasets/test_G.npy")

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
        for i,g in enumerate(g_batch):
            gnn=GNN(D,num_V=g.shape[1])
            _,loss_1=gnn.forward(G=g,W=W,A=A,b=b,t=t,y=y_batch[i])
            _,loss_2=gnn.forward(G=g,W=W,A=A,b=b+0.001,t=t,y=y_batch[i])
            #gradient of b
            gra_b=(loss_2-loss_1)/0.001
            #save the gradient b to delta_b
            delta_b.append(gra_b)
            gra_A=np.zeros(D)
            gra_W=np.zeros((D,D))
            for j in range(D):
                delta=np.zeros(D)
                delta[j]=1
                _,loss_3=gnn.forward(g,W,A+0.001*delta,b,t,y=y_batch[i])
                #gradient of A
                gra_A[j]=(loss_3-loss_1)/0.001
                for k in range(D):
                    delta=np.zeros((D,D))
                    delta[j,k]=1
                    _,loss_4=gnn.forward(g,W+0.001*delta,A,b,t,y=y_batch[i])
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
        gnn=GNN(D=D,num_V=g.shape[1])
        train_loss.append(gnn.forward(g,W,A,b,t,y=train_y[i])[1])
        train_acc.append(gnn.forward(g,W,A,b,t,y=train_y[i])[0]==train_y[i])

    #calculate the loss and accuracy of each validation data
    for i,g in enumerate(val_g):
        gnn=GNN(D=D,num_V=g.shape[1])
        val_loss.append(gnn.forward(g,W,A,b,t,y=val_y[i])[1])
        val_acc.append(gnn.forward(g,W,A,b,t,y=val_y[i])[0]==val_y[i])

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
plt.savefig('../imga/epoch_vs_accuracy_Adam.png')
plt.show()

#epoch vs mean loss 
plt.plot(range(100),loss_for_train_Adam,"r",label="train")
plt.plot(range(100),loss_for_val_Adam,"b",label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.savefig('../image/epoch_vs_loss_Adam.png')
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