from Question2 import GNN
import numpy as np
import matplotlib.pyplot as plt

train_g=np.load("../datasets/train_g.npy")
train_y=np.load("../datasets/train_y.npy")
val_g=np.load("../datasets/val_g.npy")
val_y=np.load("../datasets/val_y.npy")

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
        for i,g in enumerate(g_batch):
            gnn=GNN(D,num_V=g.shape[1])
            _,loss_1=gnn.forward(g,W,A,b,t,y=y_batch[i])
            _,loss_2=gnn.forward(g,W,A,b+0.001,t,y=y_batch[i])
            #gradient of b
            gra_b=(loss_2-loss_1)/0.001
            #save the gradient b to delta_b
            delta_b.append(gra_b)
            gra_A=np.zeros_like(A)
            gra_W=np.zeros_like(W)
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
        #update
        b=b-delta_b*alpha+eta*omega_b
        A=A-delta_A*alpha+eta*omega_A
        W=W-delta_W*alpha+eta*omega_W
        #update momentum
        omega_b=-delta_b*alpha+eta*omega_b
        omega_A=-delta_A*alpha+eta*omega_A
        omega_W=-delta_W*alpha+eta*omega_W
    return b,A,W



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

if __name__ == "__main__":
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

        #calculate the mean loss and accuracy of each training data
        for i,g in enumerate(train_g):
            gnn=GNN(D=D,num_V=g.shape[1])
            train_loss.append(gnn.forward(g,W,A,b,t,y=train_y[i])[1])
            train_acc.append(gnn.forward(g,W,A,b,t,y=train_y[i])[0]==train_y[i])
        #calculate the mean loss and accuracy of each validation data
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

if __name__ == "__main__":
    
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
        acc_for_train_MSGD.append(train_acc)
        acc_for_val_MSGD.append(val_acc)
        loss_for_train_MSGD.append(train_loss)
        loss_for_val_MSGD.append(val_loss)


##############################################
## Plot the mean loss and accuracy vs epoch ##
##############################################

if __name__ == "__main__":
    
    #epoch vs mean accuracy 
    plt.plot(range(100),acc_for_train_SGD,"r",label="train")
    plt.plot(range(100),acc_for_val_SGD,"b",label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.savefig('../img/epoch_vs_accuracy_SGD.png')
    plt.show()

    #epoch vs mean loss 
    plt.plot(range(100),loss_for_train_SGD,"r",label="train")
    plt.plot(range(100),loss_for_val_SGD,"b",label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig('../img/epoch_vs_loss_SGD.png')
    plt.show()

    #epoch vs mean accuracy 
    plt.plot(range(100),acc_for_train_MSGD,"r",label="train")
    plt.plot(range(100),acc_for_val_MSGD,"b",label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.savefig('../img/epoch_vs_accuracy_MSGD.png')
    plt.show()

    #epoch vs mean loss
    plt.plot(range(100),loss_for_train_MSGD,"r",label="train")
    plt.plot(range(100),loss_for_val_MSGD,"b",label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig('../img/epoch_vs_loss_MSGD.png')
    plt.show()


