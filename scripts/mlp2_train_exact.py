import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.optimizers import Adam  
import numpy as np  

from normalize import normalize01

tf.random.set_seed(1234)
np.random.seed(1234)

prm_load=0
prm_train=1
prm_save=1

#mlp_train的改进，结果更精确


#逐行归一化
def rownormal(X_train):
    for i in range(X_train.shape[0]):  
        X_train[i,:6]=      (X_train[i,:6]-np.min(X_train[i,:6]))/\
                    (np.max(X_train[i,:6])-np.min(X_train[i,:6]))
    return X_train
    
#dx/t , v , gt/2 ,gamma, v2
input_dim = 7

# maxe,mine
output_dim = input_dim + 2
input_shape = (input_dim,)  


ID_GAMMA=6
ID_INPUT_V2=7
ID_INPUT_V=4
ID_INPUT_GTD2=5

samplenum=1
samplenum=5
samplenum=12
samplenum=24
samplenum=48
samplenum=100
# samplenum=3000

epochs=10
# epochs=100
# epochs=300
epochs=1000



def ones_initializer(shape, dtype=None):  
    return tf.ones(shape, dtype=dtype)  


X_train = np.random.random((samplenum*10, input_dim))  # 48个样本（3个epoch * 16个批次/epoch） 
X_train=rownormal(X_train)


for i in range(X_train.shape[0]):

    #tune只有2个值
    # if(X_train[i,4]>0.5):
    #     X_train[i,4]=10
    # else:
    #     X_train[i,4]=0

    pass
    # X_train[i,4]=X_train[i,4]*5

    # X_train[i,4]=X_train[i,4]*20-10

    


X_train=X_train.astype(np.float32)
# y_train = np.random.random((samplenum*10, 4))  # 对应的48个目标值，每个值4个输出  
y_train=np.zeros((X_train.shape[0],output_dim))
# y_train=tf.convert_to_tensor(X_train)

# y_train=y_train.cpu().numpy()
y_train[:,:input_dim]=X_train
#y的第8个值放maxE,第9个值放min E
y_train[...,-2]=np.amax(y_train[...,0:4]+y_train[...,4][:, np.newaxis]+y_train[...,5][:, np.newaxis],axis=1)
y_train[...,-1]=np.amin(y_train[...,0:4]+y_train[...,4][:, np.newaxis]+y_train[...,5][:, np.newaxis],axis=1)


y_train=tf.convert_to_tensor(y_train)




cnt=0



def custom_loss(y_true, y_pred):#已经按batch划分了
    print(y_pred.shape)#batch_size * 4

    dot_product=tf.reduce_sum(y_pred[:,:4] * y_true[:,:4],axis=1)
    print(dot_product.shape)
    print(dot_product)




    # maxe=tf.reduce_sum(y_true[...,0:6],axis=1)
    # mine=tf.reduce_sum(y_true[...,4:6],axis=1)


    # maxe=tf.reduce_max(y_true[...,0:4],axis=1)+y_true[...,4]+y_true[...,5]
    # mine=tf.reduce_min(y_true[...,0:4],axis=1)+y_true[...,4]+y_true[...,5]

    maxe=y_true[...,-2]
    mine=y_true[...,-1]


  



    tune=y_true[...,ID_GAMMA]




    predictenergy=dot_product+y_true[...,ID_INPUT_V]+y_true[...,ID_INPUT_GTD2]


    # C1= y_true[...,ID_INPUT_GTD2]*2 + y_true[...,ID_INPUT_V]*2
    # C2= y_true[...,ID_INPUT_GTD2]* y_true[...,ID_INPUT_V]*2 + \
    #     y_true[...,ID_INPUT_V2]
    #            # y_true[...,ID_INPUT_GTD2]**2+\
    # predictenergy=dot_product**2+ dot_product*C1 + C2



    return tf.reduce_sum((  (maxe-mine)*tune+mine   -   predictenergy)**2)/batch_size

if(prm_load):
    tf.keras.models.load_model("./LinearMix.h5")
# 构建模型  
else:
    model = Sequential([  
        # Dense(64,activation='sigmoid',use_bias=True,  input_shape=input_shape), 
        # Dense(32,activation='sigmoid',use_bias=True),   


        Dense(64,activation='relu',use_bias=True,input_shape=input_shape),                               
        Dense(32,activation='relu',use_bias=True),    



        # Dense(output_dim,activation='softmax')


        Dense(output_dim,activation='sigmoid')
        #8e-5


        # Dense(output_dim,kernel_initializer=ones_initializer,activation='softmax') 
        #由于网络总是倾向于选择能量最低的模型 





        # Dense(output_dim,activation='relu') 

        # Dense(output_dim,activation='sigmoid')  

    ])  

    # 编译模型，这里我们假设是一个回归问题，使用均方误差（MSE）作为损失函数  
    model.compile(optimizer=Adam(learning_rate=2e-3),  
                loss=custom_loss)  
    
  

  
# 训练模型  
# 注意：这里的steps_per_epoch和epochs参数是可选的，因为当你直接传递X_train和y_train时，  
# Keras会根据X_train的形状自动推断出批次大小和epoch数量（如果批次大小是已知的）。  
# 但是，为了明确起见，我们可以在这里设置它们。  
# 由于我们每个批次送入16个样本，总共有48个样本，所以steps_per_epoch=48//16=3。  
# 但是，由于Keras在fit方法中可以自动处理这个问题（当batch_size未明确设置时，它默认为32或数据集大小的最小值），  
# 我们实际上可以省略steps_per_epoch和batch_size参数，除非我们有特定的需求。  
# 然而，为了教育目的，我们将在这里明确设置它们。  
batch_size = 32  
if(prm_train):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)  
  
# 注意：上面的代码实际上设置了每个epoch有3个批次（因为48个样本/16个批次/epoch=3），  
# 并且我们训练了3个epoch，所以总共训练了9个批次（3个epoch * 3个批次/epoch）。  
  
# 如果你想查看模型的总结信息  
model.summary()
for i in range(0,6):
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<test>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
    x_test=np.random.random((3, input_dim))

    # x_test=rownormal(x_test)


    if(i==1):
        x_test[:,ID_GAMMA]=0
    elif(i==2):
        x_test[:,ID_GAMMA]=1
    if(i==3):
        x_test[:,ID_GAMMA]=0.5
    if(i==4):
        x_test=np.random.random((21, input_dim))
        x_test=rownormal(x_test)
        x_test[0,ID_GAMMA]=0
        x_test[1,ID_GAMMA]=0.1
        x_test[2,ID_GAMMA]=0.2
        x_test[3,ID_GAMMA]=0.3
        x_test[4,ID_GAMMA]=0.4
        x_test[5,ID_GAMMA]=0.5
        x_test[6,ID_GAMMA]=0.6
        x_test[7,ID_GAMMA]=0.7
        x_test[8,ID_GAMMA]=0.8
        x_test[9,ID_GAMMA]=0.9
        x_test[10,ID_GAMMA]=1.0
        x_test[11,ID_GAMMA]=1.1
        x_test[12,ID_GAMMA]=1.2
        x_test[13,ID_GAMMA]=1.3
        x_test[14,ID_GAMMA]=1.4
        x_test[15,ID_GAMMA]=1.5
        x_test[16,ID_GAMMA]=1.6
        x_test[17,ID_GAMMA]=1.7
        x_test[18,ID_GAMMA]=1.8
        x_test[19,ID_GAMMA]=1.9
        x_test[20,ID_GAMMA]=2.0




    tune=x_test[...,ID_GAMMA]


    maxe=np.amax(x_test[...,:4],axis=1)+x_test[...,4]+x_test[...,5]
    mine=np.amin(x_test[...,:4],axis=1)+x_test[...,4]+x_test[...,5]
    # maxe=1
    # mine=0
    # maxe=np.sum(x_test[...,0:6],axis=1)
    # mine=np.sum(x_test[...,4:6],axis=1)

    maxe=np.amax(x_test[...,0:4]+x_test[...,4][:, np.newaxis]+x_test[...,5][:, np.newaxis],axis=1)
    mine=np.amin(x_test[...,0:4]+x_test[...,4][:, np.newaxis]+x_test[...,5][:, np.newaxis],axis=1)


    print('model energy--------------')
    print(x_test)
    print(x_test.shape)
    pre=model.predict(x_test)
    pre=pre[:,:4]
 

    print('--------------coff')
    print(pre)
    print('--------------pre strength:')

    pre_strength=\
            (np.sum(x_test[...,:4]*pre,axis=1)+\
                    x_test[...,4]+\
                    x_test[...,5])
    

    pre_strength=(pre_strength-mine)/(maxe-mine)


    print(pre_strength)

    print('--------------expected strength')

    expected_strength=tune
    print(expected_strength)

    print('------------------relative error ')
    print((pre_strength-expected_strength)/expected_strength)

if(prm_save):
    model.save("./LinearMix.h5")

def predictincconv_exact(d1,d2,d3,d4,v_t,gt_d2,partnum,tune):
    
    print('exact')

    x_test=np.array([d1,d2,d3,d4,v_t,gt_d2])
    print(x_test.shape)#6 3


    x_test=normalize01(x_test=x_test,rowwise=False)



    x_test=np.concatenate([x_test,np.array([[tune,tune,tune]])])#1 3
    print('[normalized]')
    print(x_test)
    # x_test=np.array([x_test])
    print(x_test.shape)

    # assert(False)
    pre=model.predict(x_test.T)

    return pre#3 * 9