import numpy as np  
# from normalize import normalize01
np.random.seed(1234)

from mlp2_train_exact___2 import predictincconv_exact,normalmethod,relumatrix

samplenum=3
x_raw=np.random.random((samplenum,6))*2-1

#01归一化
if(normalmethod==0):
    k=1/(np.amax(x_raw,axis=1)-np.amin(x_raw,axis=1))
    b=-np.amin(x_raw,axis=1)*k
    b=b[...,np.newaxis]


#-1,1归一化
elif(normalmethod==1):
    b=0
    k=1/np.amax(abs(x_raw),axis=1)
    
    # print(k)
k=k[...,np.newaxis]
print('[k]\t'+str(k))


def fangcheng(a,b,c,d):
    # print(a.shape)#31
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)
    delta=b*b-4*a*(c-d)
    if(delta.any()<0):
        assert(False)
    return (-b+np.sqrt(delta))/(2*a), \
           (-b-np.sqrt(delta))/(2*a)


fmin=[]
fmax=[]

#求出F理论上能达到的范围，这里是真实值
for i in range(x_raw.shape[0]):

    from util import relu

    a1=relu ( x_raw[i,0])+relu( x_raw[i,1])+relu( x_raw[i,2])+relu( x_raw[i,3])      +x_raw[i,4]+x_raw[i,5]
    a2=-relu(-x_raw[i,0])-relu(-x_raw[i,1])-relu(-x_raw[i,2])-relu(-x_raw[i,3])      +x_raw[i,4]+x_raw[i,5]


    # if(a1*a2<0):
    #     a1=max(a1**2,a2**2)
    #     a2=0
    # else:
    #     a1=max(a1**2,a2**2)
    #     a2=min(a1**2,a2**2)

    fmax.append(a1)
    fmin.append(a2)
    # return a1,a2

fmax=np.array(fmax)
fmin=np.array(fmin)






print('info')
# print(b.shape)#3 1
# print(k.shape)#3 1


print(x_raw)
x=k*x_raw+b

print(x)
print(x.shape) #3,6
# assert(False)

print('[f min,max]'+str([fmin,fmax]))
# print('[   e min,max]'+str([fmin**2,fmax**2]))

# needf=np.sqrt((fmax**2+fmin**2)/2)

#所需比例：

needratio=np.array([1,1.25,1.5])
prm_needratio=1

if(prm_needratio):#指定网络需要达到的能量比例
    needf=(fmax-fmin)*needratio+fmin
    print(fmax.shape)#3,
    print(needratio.shape)
    # assert(False)
# else:
#     needf=needf0    #直接指定网络需要达到的能量
#     needratio=(needf0-fmin)/(fmax-fmin)
#     if(needratio>2 ):
#         print('over capability')
#         needratio=1.5
#     elif(needratio<0):
#         print('over capability')
#         needratio=0.05

print('[need ratio]\t'+str(needratio))
print('[need F]\t'+str(needf))



#归一化后的能量范围
if(normalmethod==0):
    ffmin=x[:,4]+x[:, 5]
    ffmax=np.sum(x[:,:6],axis=1)
elif(normalmethod==1):
    ffmax=np.sum(relumatrix(x[...,0:4]               ),               axis=1)+x[...,4]+x[...,5]
    ffmin=np.sum(relumatrix(x[...,0:4],positive=False),               axis=1)+x[...,4]+x[...,5]

print(ffmin.shape)#3,
print(needf.shape)
# if(normalmethod==1):
#     needf=needf[:,np.newaxis]
print(k.shape)



# assert(False)
if(normalmethod==0):
    tune=(needf-ffmin)/(ffmax-ffmin)
elif(normalmethod==1):
    # print((needf[:,np.newaxis]*k)[:,0].shape)
    # print(needf.shape)#3,
    # assert(False)
    tune=(  (needf[:,np.newaxis]*k)[:,0]    -ffmin)/(ffmax-ffmin)
# print()


#所需能量E相对于EE的比例
print('[tune relative to FF]\t'+str(tune))
coff=predictincconv_exact(x_raw[:,0],x_raw[:,1],x_raw[:,2],x_raw[:,3],x_raw[:,4],x_raw[:,5],partnum=10,tune=tune)
print('--coff--')
print(coff.shape)#3 9
print(coff[:,:4])

#pass
f=coff[...,0]*x_raw[...,0]+\
  coff[...,1]*x_raw[...,1]+\
  coff[...,2]*x_raw[...,2]+\
  coff[...,3]*x_raw[...,3]+\
  x_raw[:,4]+x_raw[:,5]

f=f[:,np.newaxis]

m=(2+np.sum(coff[:,:4],axis=1))
m=m[:,np.newaxis]
#pass

# print(m.shape)#31
# print(f.shape)#31
# print(b.shape)#31
# print(k.shape)#31
# print(f[0])
# print(m[0])


# assert(False)

if(normalmethod==0):
    eesqr=k*f+m*b
elif(normalmethod==1):
    eesqr=k*f
#pass


ee2= (k*k)*(f*f+12*b*f+36*b*b)


# 它和ee计算结果一致，已验证
# eesqrsame=(coff[...,0]*x[...,0]+\
#         coff[...,1]*x[...,1]+\
#         coff[...,2]*x[...,2]+\
#         coff[...,3]*x[...,3]+\
#         x[:,4]+x[:,5])




print('[predict eesqr]\t'+str(eesqr))
# print('[predict eesqrsame)]\t'+str(eesqrsame))

#验证EE是不是真的达到了对网络所提出的要求(检查网络本身的可靠性，和原始数据无关)
print('[verify tune]:'+str((eesqr.T-ffmin)/(ffmax-ffmin)))

print('[predict ee2]\t'+str(ee2))

if(normalmethod==1):
    #验证F是不是符合原始数据的要求
    print('[F]\t'+str(f.T))
    print('[F ratio]\t'+str((f.T-fmin)/(fmax-fmin)))

print('[ffmin]\t'+str([ffmin]))
print('[ffmax]\t'+str([ffmax]))

print('[E from EE]\t')


alis=k*k
blis=2*k*m*b
clis=m*m*b*b
dlis=eesqr*eesqr
print(fangcheng(alis,blis,clis,dlis))


# print('[e min,max]\t'+str([x_raw[:,4]+x_raw[:,5],np.sum(x_raw[:4],axis=1)]))
