from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score,recall_score, mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv("DL_Nganhang.csv",header = 0)
df = pd.DataFrame(df).fillna(0)

df.head(10)

#X1: Tuổi
#X2: Giới tính
#X4: Số ngày mở thẻ
#X5i: Số tiển debit quý i
#X6i: Số tiền crebit quý i
#X7i: Số lần giao dịch quý i      (i=1,2,3,4)
#X8i: Số debit quý i
#Amounti: Số dư cuối kì quý i
#Label: Nhãn lớp : churn=-1, non-churn=1

#Vẽ biểu đồ tương quan
import matplotlib.pyplot as plt
import seaborn as sns
# corrmat = df.corr()
# f, ax = plt.subplots(figsize=(20, 15))
# cols=df.columns.values
# cm = np.corrcoef(df.values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)
# plt.show()

# #Vẽ đồ thị biểu diễn ma trận tương quan
# mask=cm<=0.00
# cm[mask]=0
# import networkx as nx
# graph = nx.from_numpy_matrix(cm)
# label = dict()
# i = 0
# for c in cols:
#   label[i] = c
#   i += 1 
# nx.draw(graph, labels=label, node_size=700, alpha=0.8, size=(20,20))
# plt.savefig('custom.png', bbox_inches='tight')
# plt.show()
# n, bins, patches = plt.hist(x=df['Amount1'], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
# plt.text(23, 45, r'$\mu=15, b=3$')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# size, scale = 1000, 10
# commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)

# commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
#                    color='#607c8e')
# plt.title('Commute Times for 1,000 Commuters')
# plt.xlabel('Counts')
# plt.ylabel('Commute Time')
# plt.grid(axis='y', alpha=0.75)
# df['Amount1'].histogram() 
#print(gaussian_numbers)
#
plt.hist(df['Amount4'], 100, range=[0, 4000000], facecolor='gray', align='mid')
print(df['Amount4'].describe())
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
#Xây dựng ma trận xác suất chuyển từ quý i -> quý j
def P(X1, X2, k =3):
  shape=(3,3)
  P=np.zeros(shape)
  print(P)
  N=len(X1)
  #Lấy ngưỡng 
  p1=np.percentile(X1,80)
  print(p1)
  p2=np.percentile(X2,20)
  print(p2)
  #Hàng 1
  for i in range(N):
    if (X1[i]>=p1 and X2[i]>=p1):
      P[0][0]=P[0][0]+1
    if (X1[i]>=p1 and p2<=X2[i]<p1  ):
      P[0][1]=P[0][1]+1
    if (X1[i] >=p1 and X2[i]<p2):
      P[0][2]=P[0][2]+1
  N1=sum(P[0,:])
  P[0,:]=P[0,:]/N1
  #Hàng2
  for i in range(N):
    if(p2<=X1[i]<p1 and X2[i]>=p1):
      P[1][0]+=1
    if(p2<=X1[i]<p1 and p2<=X2[i]<p1):
      P[1][1]+=1
    if(p2<=X1[i]<p1 and X2[i]<p2):
      P[1][2]+=1
  N2=sum(P[1,:])
  P[1,:]=P[1,:]/N2
  #Hàng3
  for i in range(N):
    if(X1[i]<p2 and X2[i]>=p1):
      P[2][0]+=1
    if(X1[i]<p2 and p2<=X2[i]<p1):
      P[2][1]+=1
    if(X1[i]<p2 and X2[i]<p2):
      P[2][2]+=1
  N3=sum(P[2,:])
  P[2,:]=P[2,:]/N3
  #print(P)
  return P

#print("Ma trận xác suất chuyển từ quý 1 sang quý 2:")
p1=P(df['Amount1'],df['Amount2'])
#print(type(p1))
#print("")
#print("Ma trận xác suất chuyển từ quý 2 sang quý 3:")
#p2=P(df['Amount2'],df['Amount3'])
#print("")
#print("Ma trận xác suất chuyển từ quý 3 sang quý 4:")
# p3=P(df['Amount3'],df['Amount4'])
# p = (p1+p2+p3)/3

# p = np.mean( np.array([ p1, p2, p3]), axis=0 )
# print(p)

# Calculate CLV


# marginProfit = 3.52/400
# cost = 90000/4
# d = 4.80/400

# def calculateProfit(amount):
#   return marginProfit*amount - cost

# profit = []
# for i in range(len(df)): 
#   amount = df['Amount4'][i]
#   profit.append(calculateProfit(amount))

# df.insert(loc=0, column='profit', value = profit)
# print(df['profit'].describe())
# # #Mean of state
# p41=np.percentile(df['Amount4'],80)
# p42=np.percentile(df['Amount4'],20)

# #High
# meanstate1 = 0
# n1 = 0
# #Medium
# meanstate2 = 0
# n2 = 0
# #Low
# meanstate3 = 0
# n3 = 0
# for i in range(len(df)):
#   if (df['Amount4'][i]>=p41):
#     n1 += 1
#     meanstate1 += df['profit'][i]
#   if (p42<=df['Amount4'][i]<p41 ):
#     n2 += 1
#     meanstate2 += df['profit'][i]
#   if (df['Amount4'][i]<p42):
#     n3 += 1
#     meanstate3 += df['profit'][i]

# meanstate1 = meanstate1/n1
# meanstate2 = meanstate2/n2
# meanstate3 = meanstate3/n3

# profit = [meanstate1, meanstate2, meanstate3]

# #Predict
# profit5 = np.dot(profit, p)
# profit6 = np.dot(profit5, p)
# profit7 = np.dot(profit6, p)
# profit8 = np.dot(profit7, p)
# profit9 = np.dot(profit8, p)
# profit10 = np.dot(profit9, p)
# profit11 = np.dot(profit10, p)
# profit12 = np.dot(profit11, p)

# def caculateCLV(profit5, profit6, profit7, profit8, profit9, profit10, profit11, profit12):
#   return profit5/(1+d) + profit6/(1+d)**2 + profit7/(1+d)**3 + profit8/(1+d)**4 + profit9/(1+d)**5 + profit10/(1+d)**6 + profit11/(1+d)**7 + profit12/(1+d)**8

# #CLV State High
# CLV1 = caculateCLV(profit5[0], profit6[0], profit7[0], profit8[0], profit9[0] , profit10[0], profit11[0], profit12[0])
# #CLV State Medium
# CLV2 = caculateCLV(profit5[1], profit6[1], profit7[1], profit8[1], profit9[1] , profit11[1], profit11[1], profit12[1])
# #CLV State Low
# CLV3 = caculateCLV(profit5[2], profit6[2], profit7[2], profit8[2], profit9[2] , profit12[2], profit11[2], profit12[2])
# print(CLV1, CLV2, CLV3)
