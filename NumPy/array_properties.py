import numpy as np 
 
a = np.arange(24)  
print (a.ndim)             # one d

b = a.reshape(2,4,3)  # three d
print (b.ndim)

print('------------------------------')


a = np.array([[1,2,3],[4,5,6]])  
print (a.shape)

a = np.array([[1,2,3],[4,5,6]]) 
a.shape =  (3,2)  
print (a)


print('------------------------------')

import numpy as np
 
# 默认为浮点数
x = np.zeros(5) 
print(x)
 
# 设置类型为整数
y = np.zeros((5,), dtype = int) 
print(y)
 
# 自定义类型
z = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])  
print(z)