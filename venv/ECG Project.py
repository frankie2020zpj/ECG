#%%

import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as n
from scipy.fftpack import fft,ifft
import numpy as np

#%% md

# * https://my.oschina.net/u/4633685/blog/4699630

#%%

# 数据读取
def read_ecg(file_name):
    alist = []
    with open(file_name, 'rb') as text:
        words = text.read(2)
        while words:
            words = text.read(2)
            num = int.from_bytes(words,byteorder='little',signed=True)
            alist.append(num)
    return alist

#%%

# 心跳数据读取
a01 = read_ecg(r'D:\Files\ECG/a01.dat')
len(a01)

#%%

# 压力数据读取
a01r = read_ecg(r'D:\Files\ECG/a01r.dat')
len(a01r)
a01r = a01r[:12000]

#%%

# 将压力数据分成4个sensor
sen1=[]
sen2=[]
sen3=[]
sen4=[]
for i in range(len(a01r)//4):
    sen1.append(a01r[i*4])
    sen2.append(a01r[i*4+1])
    sen3.append(a01r[i*4+2])
    sen4.append(a01r[i*4+3])

#%%

# 压力数据绘图
plt.figure(figsize=(20,6))
plt.plot(sen1,'g',alpha=0.5)
plt.plot(sen2,'b',alpha=0.5)
plt.plot(sen3,'r',alpha=0.5)
plt.plot(sen4,'y',alpha=0.5)
plt.show()

#%%

# 压力数据绘制子图
sen = [sen1,sen2,sen3,sen4]
fig=plt.figure(figsize=(20,8))
fig.subplots_adjust(wspace=0.1,hspace=0.2)
for i in range(1,5):
    alist = sen[i-1]
    ax1=fig.add_subplot(2,2,i)
    ax1.plot(alist,alpha=0.5,c='g')
    ax1.title.set_text(f'sen{i-1}')

#%%

# 心电数据过滤，去掉绝对值大于100的
alist = a01[:3000]
list_01 = []
for i in alist:
    if abs(i)<=100:
        list_01.append(i)

#%%

# 压力数据绘图
plt.figure(figsize=(20,6))
plt.plot(list_01,'g',alpha=0.5)
plt.plot(a01r,'b',alpha=0.5)
plt.show()

#%%

# 调用HP库进行R波位置定位
data = hp.scale_data(list_01)

# 自动获取R波位置
working_data, measures = hp.process(list_01, 500.0)
hp.plotter(working_data, measures)

#%%

# 获取R波对应的下标
peaklists = working_data['peaklist']
# 去头去尾 因为头尾数据不一定是完整的
peaklists = peaklists[1:-1]
print("所有R波对应的下标：", peaklists)

# 获取心拍

for i in peaklists:
    tem_data = data[i - 150:i + 150]
    plt.plot(tem_data)
    title = str(i)
    plt.title(title)
    plt.show()

from scipy.fft import fft, fftfreq
N = 6000
T = 1.0 / 100.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = a01
yf = fft(y)
xf = fftfreq(N, T)[:N//2]

plt.figure(figsize=(20,5))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.show()