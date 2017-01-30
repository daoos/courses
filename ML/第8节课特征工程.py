from sklearn.feature_selection import RFE                #包裹型特征选择
from sklearn.preprocessing import StandardScaler    #数据标准化
from sklearn.cross_validation import train_test_split  #交叉验证
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.datasets import load_boston     
import numpy as np
from sklearn.linear_model import Ridge                   #L2正则化
from sklearn.linear_model import Lasso                   #L1正则化

#数据导入
boston=load_boston()                        
scaler = StandardScaler()                                        #数据标准化
X=scaler.fit_transform(boston.data)                        #特征变量的数据
y=boston.target                                                      #结果-->房价
names=boston.feature_names                                  #特征名

#算法拟合
lr=LinearRegression()              #线性回归算法
rfe=RFE(lr,n_features_to_select=1) 
rfe.fit(X,y)                              #拟合数据

print("原有特征名:")
print("\t",list(names))
print("排序后的特征名:")
print("\t",sorted(zip(map(lambda x: round(x,4),rfe.ranking_),names))) #对特征进行排序

#提取排序后的属性在原属性列的序列号
rank_fea=sorted(zip(map(lambda x: round(x,4),rfe.ranking_),names))   #排序好的特征
rank_fea_list=[]                                              #用来装排序的特征的属性名
for i in rank_fea:
	rank_fea_list.append(i[1])	
index_list=[0]*13                                         #记录特征属性名对应原属性names的序列号
for j,i in enumerate(rank_fea_list):
	index=list(names).index(i) #获取序列号
	index_list[j]=index
print("排序后特征对应原特征名的序列号：")
print("\t",index_list)
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
'''
#如果想要看一看每个特征与结果之间的散点分布情况的话，请把''' ''''去掉即可，即把注释符号去掉
#给予排序号的每个特征和结果画图，看看每个特征和结果之间的关系
for i in index_list:                                #共有13个特征，所以画13张图
    plt.figure(names[i])            #每张图以该特征为名
    plt.scatter(X[:,i],y)             #画出散点图
    plt.xlabel(names[i])
    plt.ylabel("price house")
'''
#-------------------------------------------问题是：-------------------------------------------------------------------------------------------------------
#1.第三张图，即CHAS属性对应图，这样的分布也可以用线性回归算法吗？这是不是说只有CHAS只能取两个值0，1？
#2从图中可以看到前六张图都是比较符合线性回归的，能不能直接抽取这前6个特征向量出来直接用来拟合LinearReggression算法？（或者是排序好的前几个特征）
#3.如果用L1正则化和L2正则化方法，与包裹型特征选择的方法的比较？
#4.用什么参数去评估模型的优劣？卡方值
#验证一下-----------------------------------------------------------------下面是验证------------------------------------------------------------------------

#提取排名前n个特征的算法拟合
print("提取排序后的前n个特征向量进行训练:")
for time in range(2,13):
    X_exc=np.zeros((X.shape[0],time))     #把排序好前六个特征向量提取出来,放在X—exc矩阵里
    for  j,i in enumerate(index_list[:time]):
        X_exc[:,j]=X[:,i]
        
    X_train1,X_test1,y_train1,y_test1=train_test_split(X_exc,y)
    lr1=LinearRegression()
    lr1.fit(X_train1,y_train1)
    print("\t提取{0}个的特征-->R方值\t".format(time),lr1.score(X_test1,y_test1))
print()

#原数据全部特征拟合
print("全部特征向量进行训练：")
X_train_raw,X_test_raw,y_train_raw,y_test_raw=train_test_split(X,y)
lr_raw=LinearRegression()
lr_raw.fit(X_train_raw,y_train_raw)
print("\t全部特征---->R方值\t",lr_raw.score(X_test_raw,y_test_raw))
print()

#只提取一个特征向量
print("只提取一个特征向量进行训练：")
for i in index_list:
    X2=np.zeros((X.shape[0],1))
    X2[:,0]=X[:,index_list[i]]
    X_train2,X_test2,y_train2,y_test2=train_test_split(X2,y)
    lr2=LinearRegression()
    lr2.fit(X_train2,y_train2)
    print("\t特征",names[i],"---->R方值","\t",lr2.score(X_test2,y_test2))
print()

#采取L1正则化的方法
print("采取L1正则化的方法:")
lasso= Lasso(alpha=0.3)        #alpha参数由网友用网格搜索方法确定下来的
lasso.fit(X_train_raw,y_train_raw)
print("\tL1正则化特征---->R方值\t",lasso.score(X_test_raw,y_test_raw))
print()

#采取L2正则化的方法
print("采取L2正则化的方法")
ridge = Ridge(alpha=10)         #alpha参数由网友用网格搜索方法确定下来的
ridge.fit(X_train_raw,y_train_raw)
print("\tL2正则化特征---->R方值\t",ridge.score(X_test_raw,y_test_raw))

'''
plt.show()        #显示图片
'''
#-----------------------------------------------------------我的看法--------------------------------------------------
#经过多次实验后发现：
#1.在本例中，用L2正则化方法是效果最好的，不仅R方值较高，而且R方值变化幅度不大
#2.只采用一个特征向量的话，效果非常差，即使是排序第一的特征值。
#3.用排序后的前n个特征一起训练算法，效果好像没有全部特征一起训练算法那么好。那么RFE算法（包裹型特征选择）在本例的作用是？？

#-------------------------------------------------------------疑问-----------------------------------------------------
#1.R-方值在65%-75%的是什么概念？评估一个模型的好坏，标准是什么？是R方值大于某个参数？还是与其他预测房价的系统比吗？
#2.在本例中采用的交叉验证方法合理吗? (交叉验证的默认值是----->训练集：测试集=3:1)
#3.特征删选的话，可能会把相关联的特征给剔除掉，造成很大的影响，为了避免这种情况，我该怎么办？不可能两两特征去判断的吧？
#4.在本数据中，13个特征应该还不算高维吧？所以我觉得保留全部特征去训练算法也是没有问题吧
#5.如果拿到一个多维（假设维数不是很高的情况，比如20个）数据集，特征向量X={x0,x1,x2,...xn},结果是y,目前还不知道大概是符合什么分布，下面的思路合理吗：？
#      a.从X={x0,x1,x2,x2....xn}，逐一挑取xi(i=0到i=n),用matplotlib画出xi和y的散点分布图
#      b.观察每个特征和结果的分布图，判断大概是线性相关，还是线性不相关？？
#      c.如果大部分特征与结果成相关分布的趋势，把这些特征保留下来，然用用这些特征用来进行训练算法，采用线性回归或者其他线性回归模型
#      d.如果大部分特征与结果无相关趋势的话，在考虑其他模型吧。


