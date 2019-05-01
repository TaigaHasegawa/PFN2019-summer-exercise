import numpy as np
from Question1 import make_symmetric_matrix 

class GNN():
    def __init__(self,D,num_V):

        #D*num_V dim feature vector 
        feature_vec=np.zeros(D)
        feature_vec[0]=1
        self.X=[feature_vec for j in range(num_V)]
    
    #relu function 
    def relu(self,x):
        y = np.maximum(0, x)
        return y

    #sigmoid function
    def sigmoid(self,x):
        if x<-10:
            y=0
        else:
            y=1/(1+np.exp(-x))
        return y

    def forward(self,G,W,A,b,t,y):
        """
        G: Adjacency matrix
        W: weight parameter     
        A: weight parameter 
        b: bias 
        t: the number of steps
        y: ground truth
        """
        X=self.X
        for i in range(t):
            a=np.dot(G,X)
            X=self.relu(np.dot(a,W))
        h=np.sum(X,axis=0)
        s=np.dot(A,h)+b

        #probability
        p=self.sigmoid(s)

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
        
        #return predicted label and calculate loss 
        return y_hat,loss



######################
## Gradient Descent ##
######################

def gradient_descent(loss,variable,alpha,epsilon=0.001):
    """
    loss: loss before this time's iteration
    Variable: choice of Variable from W, A, or b
    alpha: learning rate
    epsilon: small real value 
    """
    loss_before=loss
    if variable=="W":
        gra_W=np.zeros_like(W)
        for j in range(D):
            for k in range(D):
                delta=np.zeros_like(W)
                delta[j,k]=1
                loss_after=gnn.forward(G,W+delta*epsilon,A,b,t,y)[1]
                #gradient of W[j,k]
                gra_W[j,k]=(loss_after-loss_before)/epsilon
        new_variable=W-alpha*gra_W
    elif variable=="A":
        gra_A=np.zeros_like(A)
        for j in range(D):
            delta=np.zeros_like(A)
            delta[j]=1
            loss_after=gnn.forward(G,W,A+delta*epsilon,b,t,y)[1]
            #gradient of A[j]
            gra_A[j]=(loss_after-loss_before)/epsilon
        new_variable=A-alpha*gra_A
    elif variable=="b":
        gra_b=0
        loss_after=gnn.forward(G,W,A,b+epsilon,t,y)[1]
        #gradient of b
        gra_b=(loss_after-loss_before)/epsilon
        new_variable=b-alpha*gra_b

    return new_variable



################
# Test for GNN #
################
if __name__ == "__main__":

    #Adjacency matrix for test
    G=make_symmetric_matrix(np.array([[1,0],[2,1],[3,2],[3,4],[1,3],[4,2],[4,5],[6,4],[6,3],[6,5],[7,2],[7,0],[7,3],[8,0],[8,5],[8,6],[9,2],[9,8]]),(10,10))

    #set the seed 
    D=4
    np.random.seed(2013)
    #initialize D dim weight parameter A
    A=np.random.uniform(0,0.4,D)
    #initialize bias
    b=0
    #initialize D*D dim weight parameter W
    W=np.random.uniform(0,0.4,(D,D))

    gnn=GNN(D=4,num_V=G.shape[1])

    #forward step
    y_hat,loss=gnn.forward(G,W,A,b,t=5,y=0)
    print(y_hat,loss)
    #1 741.1999772981253

    t=5
    y=0
    #20000 iterations 
    for i in range(20001):
        loss=gnn.forward(G,W,A,b,t=t,y=y)[1]
        W_new=gradient_descent(loss,"W",alpha=0.01)
        A_new=gradient_descent(loss,"A",alpha=0.01)
        b_new=gradient_descent(loss,"b",alpha=0.01)
        W=W_new
        A=A_new
        b=b_new
        #show the prediction and loss every 1000 iteration 
        if i%1000==0:
            loss=gnn.forward(G,W,A,b,t=t,y=y)[1]
            print(loss)
