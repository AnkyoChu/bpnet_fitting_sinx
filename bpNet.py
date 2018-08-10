'''
author：AnkyoChu
date:18/8/10
python 3.5
program:backpropagation neural network
net_size:1*10*10*10*1
         input+(hidden_layer*3)+output
Iteration:10000 (Changeable as you wish)
aim：sin(x)(Changeable as you wish)
'''

import numpy
import random
import math
import matplotlib.pyplot as plt

#感觉sigmod函数有点麻烦，选了个简单的整流函数
def tanh(x):
    return numpy.tanh(x)

#在backward的时候要用到，正弦双曲函数的导数
def tanh_derivate(x):
    return 1-numpy.tanh(x)**2

#在[a,b]中生成随机数
#用来选取初始的weight和bias
def rand(a,b):
    return a+(b-a)*random.random()

#初始化二维数组
def Matrix(m,n,value=0.0):
    matrix=[]
    for x in range(m) :
        matrix.append([value]*n)
    return matrix

class bpNet:
    def __init__(self):
        #数目
        self.input_n=0
        self.hidden1_n=0
        self.hidden2_n=0
        self.hidden3_n=0
        self.output_n=0

        #神经元
        self.cell_input=[]
        self.cell_hidden1=[]
        self.cell_hidden2=[]
        self.cell_hidden3=[]
        self.cell_output=[]

        #weight值
        #从hidden1开始是因为把weight值看作连接在下一个输出层上
        self.w_hidden1=[]
        self.w_hidden2=[]
        self.w_hidden3=[]
        self.w_output=[]

        #类似adam的方法，为了使gradient decent的时候不在local min处停下
        #给了一个运动变量
        self.correct_input=[]
        self.correct_hidden1=[]
        self.correct_hidden2=[]
        self.correct_hidden3=[]

    #初始化
    def set(self,ni,nh1,nh2,nh3,no):
        self.input_n=ni+1
        self.output_n=no
        self.hidden1_n=nh1+1
        self.hidden2_n=nh2+1
        self.hidden3_n=nh3+1

        self.cell_input=[1.0]*self.input_n
        self.cell_output=[1.0]*self.output_n
        self.cell_hidden1=[1.0]*self.hidden1_n
        self.cell_hidden2=[1.0]*self.hidden2_n
        self.cell_hidden3=[1.0]*self.hidden3_n

        self.w_hidden1=Matrix(self.input_n,self.hidden1_n)
        self.w_hidden2=Matrix(self.hidden1_n,self.hidden2_n)
        self.w_hidden3=Matrix(self.hidden2_n,self.hidden3_n)
        self.w_output=Matrix(self.hidden3_n,self.output_n)

        self.b_hidden1=Matrix(self.hidden1_n,1)
        self.b_hidden2=Matrix(self.hidden2_n,1)
        self.b_hidden3=Matrix(self.hidden3_n,1)
        self.b_output=Matrix(self.output_n,1)

        for i in range(self.input_n):
            for j in range(self.hidden1_n):
                self.w_hidden1[i][j]=rand(-1,1)
        for i in range(self.hidden1_n):
            for j in range(self.hidden2_n):
                self.w_hidden2[i][j]=rand(-1,1)
        for i in range(self.hidden2_n):
            for j in range(self.hidden3_n):
                self.w_hidden3[i][j] = rand(-1, 1)
        for i in range(self.hidden3_n):
            for j in range(self.output_n):
                self.w_output[i][j]=rand(-1,1)

        for i in range(self.hidden1_n):
            self.b_hidden1[i]=rand(-1,1)
        for i in range(self.hidden2_n):
            self.b_hidden2[i]=rand(-1,1)
        for i in range(self.hidden3_n):
            self.b_hidden3[i]=rand(-1,1)
        for i in range(self.output_n):
            self.b_output[i]=rand(-1,1)

        self.correct_input=Matrix(self.input_n,self.hidden1_n)
        self.correct_hidden1=Matrix(self.hidden1_n,self.hidden2_n)
        self.correct_hidden2=Matrix(self.hidden2_n,self.hidden3_n)
        self.correct_hidden3=Matrix(self.hidden3_n,self.output_n)

    def predict(self,input_list):
        for i in range(self.input_n-1):
            self.cell_input[i]=input_list[i]

        for i in range(self.hidden1_n):
            summ=0.0
            for j in range(self.input_n):
                summ+=self.cell_input[j]*self.w_hidden1[j][i]
            self.cell_hidden1[i]=tanh(summ-self.b_hidden1[i])

        for i in range(self.hidden2_n):
            summ=0.0
            for j in range(self.hidden1_n):
                summ+=self.cell_hidden1[j]*self.w_hidden2[j][i]
            self.cell_hidden2[i]=tanh(summ-self.b_hidden2[i])

        for i in range(self.hidden3_n):
            summ = 0.0
            for j in range(self.hidden2_n):
                summ += self.cell_hidden2[j] * self.w_hidden3[j][i]
            self.cell_hidden3[i] = tanh(summ - self.b_hidden3[i])

        for i in range(self.output_n):
            summ=0.0
            for j in range(self.hidden3_n):
                summ+=self.cell_hidden3[j]*self.w_output[j][i]
            self.cell_output[i]=tanh(summ-self.b_output[i])

        return self.cell_output[:]

    def backpropagation(self,case,label,learing_rate,correct):
        self.predict(case)
        out_d=[0.0]*self.output_n
        for i in range(self.output_n):
            error=label[i]-self.cell_output[i]
            out_d[i]=tanh_derivate(self.cell_output[i])*error

        h1_d=[0.0]*self.hidden1_n
        h2_d=[0.0]*self.hidden2_n
        h3_d=[0.0]*self.hidden3_n
        for i in range(self.hidden3_n):
            error=0.0
            for j in range(self.output_n):
                error+=out_d[j]*self.w_output[i][j]
            h3_d[i]=tanh_derivate(self.cell_hidden3[i])*error

        for i in range(self.hidden3_n):
            for j in range(self.output_n):
                change=out_d[j]*self.cell_hidden3[i]
                self.w_output[i][j]+=learing_rate*change+correct*self.correct_hidden3[i][j]
                self.correct_hidden3[i][j]=change

        for i in range(self.hidden2_n):
            for j in range(self.hidden3_n):
                change=h3_d[j]*self.cell_hidden2[i]
                self.w_hidden3[i][j]+=learing_rate*change+correct*self.correct_hidden2[i][j]
                self.correct_hidden2[i][j]=change

        for i in range(self.hidden1_n):
            for j in range(self.hidden2_n):
                change = h2_d[j] * self.cell_hidden1[i]
                self.w_hidden2[i][j] += learing_rate * change + correct * self.correct_hidden1[i][j]
                self.correct_hidden1[i][j] = change

        for i in range(self.input_n):
            for j in range(self.hidden1_n):
                change=h1_d[j]*self.cell_input[i]
                self.w_hidden1[i][j]+=learing_rate *change+correct*self.correct_input[i][j]
                self.correct_input[i][j]=change

        for i in range(self.output_n):
            self.b_output[i]-=(learing_rate*out_d[i])
        for i in range(self.hidden3_n):
            self.b_hidden3[i]-=(learing_rate*h3_d[i])
        for i in range(self.hidden2_n):
            self.b_hidden2[i]-=(learing_rate*h2_d[i])
        for i in range(self.hidden1_n):
            self.b_hidden1[i]-=(learing_rate*h1_d[i])

        error=0.0
        for i in range(len(label)):
            error=0.5*(label[i]-self.cell_output[i])**2

        return error

    def train(self,cases,labels,repeats,learing_rate,correct):
        for i in range(repeats):
            error=0.0
            for j in range(len(cases)):
                label=labels[j]
                case=cases[j]
                error+=self.backpropagation(case,label,learing_rate,correct)

    def test(self):
        cases=[]
        for i in range(0,21,1):
            cases.append([i*math.pi/10])
            labels=numpy.sin(cases)
        self.set(1,10,10,10,1)
        self.train(cases,labels,10000,0.05,0.1)

        #画训练点的拟合图
        test = []  # 训练范围外的数据
        yables = []

        for i in range(0, 201, 1):
            test.append([i * math.pi / 100])
        for case in test:
            yables.append(self.predict(case))

        x = numpy.arange(0.0, 2.0, 0.01)
        plt.figure()
        l1, = plt.plot(x * math.pi, numpy.sin(x * math.pi), color='red')
        l2, = plt.plot(test, yables, color='green')
        plt.legend(handles=[l1, l2, ], labels=['original', 'test predict'], loc='best')
        plt.xticks([0, numpy.pi / 2, numpy.pi, 3 * numpy.pi / 2, 2 * numpy.pi],
                   [r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+\pi*3/2$', r'$+\pi*2$'])
        plt.show()

        #画测试点的拟合
        zables = []
        for a in cases:
            zables.append(self.predict(a))
        plt.figure()
        l3, = plt.plot(x * math.pi, numpy.sin(x * math.pi), color='red')
        l4, = plt.plot(cases, zables, color='green')
        plt.legend(handles=[l3, l4, ], labels=['original', 'train predict'], loc='best')
        plt.xticks([0, numpy.pi / 2, numpy.pi, 3 * numpy.pi / 2, 2 * numpy.pi],
                   [r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+\pi*3/2$', r'$+\pi*2$'])
        plt.show()

if __name__ == '__main__':
    #稳定随机数
    random.seed(0)
    nn=bpNet()
    nn.test()