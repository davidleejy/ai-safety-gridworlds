import numpy as np
import torch
import torch_extras

B = torch.randint(low=0,high=5,size=(16,1,7,9),dtype=torch.long) # boards
orig = B.clone()
B = B.view(-1,1)
print(B.shape, B.max())
hot = torch_extras.one_hot((B.shape[0], 5), B)
print(hot.shape)
print(hot)
hot = hot.view(16,7,9,5)
hot = hot.permute(0,3,1,2) #bs x C x h x w
print(hot.shape)
# hot is use-able here.
R = torch.randint(low=0,high=5,size=(16,1,7,9), dtype=torch.long)
for i in range(0,hot.shape[0]):
    for h in range(0, hot.shape[2]):
        for w in range(0, hot.shape[3]):
            v = hot[i,:,h,w] # which is 1?
            a = np.where(v==1)[0][0]
            # print(a[0][0])
            R[i,0,h,w] = int(a)
assert torch.all(torch.eq(R, orig))


device_original = R.device
v=R.cpu()
v.to(device_original)

print(R.float().long().to('cuda:0'))
# R







# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
exit()

A = np.random.randint(low=0,high=100,size=(2,3,4))
print(A)

p = np.arange(A.shape[0])
p[np.newaxis,:,np.newaxis]
print(p.shape)

ixs1 = np.arange(A.shape[0])
ixs2 = np.arange(A.shape[1])
ixs3 = np.arange(A.shape[2])
print(ixs1, ixs2, ixs3)

ixs1 = np.expand_dims(ixs1,1)
ixs1 = np.expand_dims(ixs1,2)
print(ixs1.shape)

ixs2 = np.expand_dims(ixs2,0)
ixs2 = np.expand_dims(ixs2,2)
print(ixs2.shape)

ixs3 = np.expand_dims(ixs3,0)
ixs3 = np.expand_dims(ixs3,0)
print(ixs3.shape)

print('array indexing w/\n', A[ixs1,ixs2,ixs3])
print('diff', A - A[ixs1,ixs2,ixs3])

print('-'*9)

sixs = np.argsort(A, axis=-1)
print('sixs', sixs, sixs.shape)

print('sorted A diff', np.sort(A, axis=-1) - A[ixs1, ixs2, sixs])



#----
print('-'*9)


X = torch.randint(low=0,high=100,size=(2,3,4)) * -1
print('X', X)
entities = torch.randint(low=0,high=100,size=(2,3,4))
print('e', entities)

es, eo = torch.sort(entities, dim=-1)
print('es', es)
print(eo.shape)


tix1 = torch.tensor(ixs1,dtype=torch.int64)
tix2 = torch.tensor(ixs2,dtype=torch.int64)
tryxs = X[tix1, tix2, eo]
print('tryxs', tryxs)

#-----------
print('-'*9)
X = torch.randint(low=0,high=100,size=(2,1,4))
print(X)
rX = X.repeat(1,3,1)
print(rX, rX.shape)
print('indexing...')
print(rX[0,2,:])
print(X[0,0,:])
print(rX[1,2,:])
print(X[1,0,:])

i0 = torch.arange(end=6, dtype=torch.int64)
i0 = i0.view(i0.shape[0], 1, 1)
print(i0.shape)

#-----------
print('-'*9)
X = torch.randint(low=0,high=100,size=(2,3,4))
print(X.view(2,-1).shape)


