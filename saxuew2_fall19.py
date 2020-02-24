import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.feature_extraction.text import CountVectorizer
import  random
import copy
import math

np.random.seed(1337)
dataframe=pd.read_csv("train.csv")
arch_and_hp=dataframe['arch_and_hp'].to_list()
val_acc=dataframe['val_accs_49'].to_list()
train_acc=dataframe['train_accs_49'].to_list()
train_val_error=dataframe['val_error'].to_list()
train_test_error=dataframe['train_error'].to_list()
eopch=dataframe['epochs'].to_list()
param=dataframe['number_parameters'].tolist()
mu=dataframe['init_params_mu'].tolist()
std=dataframe['init_params_std'].tolist()
l2=dataframe['init_params_l2'].tolist()

def getmean(mu):
    mu_cal=[]
    index=0
    for i in mu:
        if index==1185:
            mu_cal.append(0)
        else:
            oneitem=i.split(',')
            list=[]
            for j in range(len(oneitem)):
                intconverted = float(oneitem[j][1:-1])
                list.append(intconverted)
            mean=np.mean(list)
            mu_cal.append(mean)
        index=index+1
    return mu_cal


mu_cal=getmean(mu)
std_cal=getmean(std)
l2_cal=getmean(l2)


def normlize(param):

   a =[random.randint(0, 100) for x in range(0, 1878)]
   amin, amax = min(param), max(param)
   for i, val in enumerate(param):
       a[i] = (val-amin) / (amax-amin)
   paramnum=copy.deepcopy(a)
   return paramnum

paramnum=normlize(param)

train_loss=dataframe['train_losses_49'].to_list()
val_loss=dataframe['val_losses_49'].to_list()

val_second_loss=dataframe['val_losses_48'].to_list()
train_second_loss=dataframe['train_losses_48'].to_list()

val_second_acc=dataframe['val_accs_48'].to_list()
train_second_acc=dataframe['train_accs_48'].to_list()




hisval=[]
for i in range(1,50):
      val_title="val_losses_"+str(i)
      hisval.append(dataframe[val_title].tolist()[0])

def archvec(arch_and_hp):
    total=[]
    for i in arch_and_hp:
        oneitem=i.split('):')
        layername=""
        for j in oneitem[1:]:
            index=j.index('(')
            layername=layername+" "+j[1:index]
        total.append(layername)
    return total

total=archvec(arch_and_hp)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(total)
print(X_train_counts.shape)



def tranform(X_train_counts,val_acc,val_second_acc,val_loss,val_firstloss,paramnum,mu_cal,std_cal,l2_cal):

    x_train=np.ones((1878,21))
    for i in range(X_train_counts.shape[0]):
        array=X_train_counts[i].toarray()
        a = np.append(array, val_acc[i])
        a=np.append(a, val_second_acc[i])
        a = np.append(a, val_loss[i])
        a = np.append(a, val_firstloss[i])
        a=np.append(a,paramnum[i])
        a = np.append(a, mu_cal[i])
        a = np.append(a, std_cal[i])
        a = np.append(a, l2_cal[i])
        x_train[i]=a
    return x_train.reshape((1878,1,21))

#means = np.mean(x_train,axis=1,keepdims=True)
#stds = np.std(x_train,axis=1,keepdims=True)
#print(means.shape)
#print(stds.shape)
#normalized_data = (x_train - means) / stds

x_train=tranform(X_train_counts,val_acc,val_second_acc,val_loss,val_second_loss,paramnum,mu_cal,std_cal,l2_cal)


def tansforam2(X_train_counts,train_acc,train_second_acc,train_loss,train_second_loss,paramnum,mu_cal,std_cal,l2_cal):
    x_train2=np.ones((1878,21))
    for i in range(X_train_counts.shape[0]):
         array2=X_train_counts[i].toarray()
         b = np.append(array2, train_acc[i])
         b = np.append(b, train_second_acc[i])
         b = np.append(b, train_loss[i])
         b = np.append(b, train_second_loss[i])
         b = np.append(b, paramnum[i])
         b = np.append(b, mu_cal[i])
         b = np.append(b, std_cal[i])
         b = np.append(b, l2_cal[i])
         x_train2[i]=b
    return x_train2.reshape((1878,1,21))

x_train2=tansforam2(X_train_counts,train_acc,train_second_acc,train_loss,train_second_loss,paramnum,mu_cal,std_cal,l2_cal)

print(x_train[0])
print(x_train2[0])


y_train = np.asarray(train_val_error)
y_train2 = np.asarray(train_test_error)
model = Sequential()
model.add(LSTM(10, input_shape=(1,21), activation="tanh"))
#model.add(Dense(50,activation='relu',input_dim=15))
model.add(Dense(10,activation='relu',kernel_initializer='random_uniform'))#kernel_initializer='random_uniform'
model.add(Dense(10,activation='relu',kernel_initializer='random_uniform'))



model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam',loss='mse')
model.fit(x_train[0:1500],y_train[0:1500],epochs=80,batch_size=8,verbose=0)
predict_train = model.predict(x_train[1500:],batch_size=1)
print('r2 score',r2_score(y_train[1500:],predict_train))


test=pd.read_csv("test.csv")
test_val_acc=test['val_accs_49'].to_list()
test_train_acc=test['train_accs_49'].to_list()
arch_test=test['arch_and_hp'].to_list()
test_loss=test['train_losses_49'].to_list()
test_val_loss=test['val_losses_49'].to_list()
#test_val_finalloss=test['val_loss'].to_list()
#test_train_finalloss=test['train_loss'].to_list()
test_eopch=test['epochs'].to_list()
test_param=test['number_parameters'].tolist()
test=pd.read_csv("test.csv")


test_val_second_loss=test['val_losses_48'].to_list()
test_train_second_loss=test['train_losses_48'].to_list()

test_val_second_acc=test['val_accs_48'].to_list()
test_train_second_acc=test['train_accs_48'].to_list()


test_mu=test['init_params_mu'].tolist()
test_std=test['init_params_std'].tolist()
test_l2=test['init_params_l2'].tolist()



test_mu_cal=getmean(test_mu)
test_std_cal=getmean(test_std)
test_l2_cal=getmean(test_l2)


total_test=archvec(arch_test)

test_paramnum=normlize(test_param)
X_train_counts_test = count_vect.fit_transform(total_test)
x_test_val=tranform(X_train_counts_test,test_val_acc,test_val_second_acc,test_val_loss,test_val_second_loss,test_paramnum,test_mu_cal,test_std_cal,test_l2_cal)

x_text_train=tansforam2(X_train_counts_test,test_train_acc,test_train_second_acc,test_loss,test_train_second_loss,test_paramnum,test_mu_cal,test_std_cal,test_l2_cal)

model2 = Sequential()
model2.add(LSTM(10, input_shape=(1,21), activation="tanh"))  # 12 is 96
model2.add(Dense(10,activation='relu',kernel_initializer='random_uniform'))
model2.add(Dense(10,activation='relu',kernel_initializer='random_uniform'))

model2.add(Dense(1,activation='linear'))
model2.compile(optimizer='adam',loss='mse')
model2.fit(x_train2[0:1500],y_train2[0:1500],epochs=100,batch_size=10,verbose=0)
predict_train2 = model2.predict(x_train2[1500:])
print('r2 score',r2_score(y_train2[1500:],predict_train2))


def generate_tableid(test_val_acc):

    id=[]
    num=0
    for i in range(len(test_val_acc)*2):
        if i%2==0:
           id.append('test_'+str(num)+'_val_error')
        else:
           id.append('test_' + str(num) + '_train_error')
           num+=1
    return id

id=generate_tableid(test_val_acc)
predict_test_error=model.predict(x_test_val)
predict_test_train_error=model2.predict(x_text_train)
len1=476
final=np.zeros(len1*2)
index1=0
index2=0
for i in range(len1*2):
    if(i%2==0):
        final[i]=predict_test_error[index1]
        index1+=1
    else:
        final[i] = predict_test_train_error[index2]
        index2+=1

y_=np.squeeze(np.asarray(final))
print((y_.shape))
df = pd.DataFrame({'id':id, 'Predicted':y_.reshape(-1,)})
df.to_csv('submission4.csv',index=False)

