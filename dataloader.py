
from sklearn.metrics import classification_report
import sklearn
import numpy as np
import random
import numpy.random as rng

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,Xval):
        self.Xval = Xval
        self.n_val,self.n_ex_val,self.w,self.h,self.cells = Xval.shape
    def get_traingPairs(self,batch_size ,prob=0.5):
        """Create batch of n pairs, half same class, half different class"""
        #train , test = tf.split(y_pred, num_or_size_splits=2, axis=0)
        pairs=[np.zeros((batch_size, self.h, self.w,self.cells)) for i in range(2)]
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        left = [];right = []
        target = []
        for _ in range(batch_size):
          res = np.random.choice([0,1],p=[1-prob,prob])
          if res==0:
            p1,p2 = tuple(random.randint(0,(int(self.n_val)-1)) for _ in range(2))
            #選取不同類的輸入對
            p1 = self.Xval[p1,res,:,:]
            p2 = self.Xval[p2,res,:,:]
            left.append(p1);right.append(p2)
            target.append(1)
          else:
            p = np.random.choice(range(0, (int(self.n_val)-1)))
            #選取同類的輸入對
            p1,p2 = self.Xval[p,0,:,:],self.Xval[p,1,:,:]
            left.append(p1);right.append(p2)
            target.append(0)
          pairs = [np.array(left),np.array(right)]
        return pairs, np.array(target)

    def make_oneshot_task(self,N):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = rng.choice(self.n_val,size=(N,),replace=False)
        #idx = random.randint(0,N-1)
        true_category = categories[0]
        test_image = np.asarray([self.Xval[true_category,0,:,:]]*N).reshape(N,self.w,self.h,self.cells)
        test_image = np.array(test_image, dtype='float64')
        support_set = self.Xval[categories,1,:,:]
        support_set[0,:,:] = self.Xval[true_category,1]
        support_set = support_set.reshape(N,self.w,self.h,self.cells)
        support_set =np.array(support_set, dtype='float64')
        pairs = [test_image,support_set]
        targets = np.ones((N,))
        targets[0] = 0
        targets = np.array(targets, dtype='float64')
        return pairs, targets

    def test_oneshot(self,model,N,k,verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        pass
        n_correct = 0
        if verbose:
          print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs,verbose=0)
            #odx = np.argmin(targets)
            if np.argmin(probs) == 0:
                n_correct+=1
        print(np.array(probs))
        percent_correct = (100.0*(n_correct) / k)
        for i,p in enumerate(probs):
           if p>0.5:
              probs[i]=1
           else:
              probs[i]=0
        
        print(classification_report(np.array(targets, dtype='int'),np.array(probs)))
        print("Accuracy:",sklearn.metrics.accuracy_score(np.array(targets, dtype='int'),np.array(probs)))
        print("Precision:",sklearn.metrics.precision_score(np.array(targets, dtype='int'),np.array(probs)))
        print("Recall:",sklearn.metrics.recall_score(np.array(targets, dtype='int'),np.array(probs)))

        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct