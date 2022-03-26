# Bayes手写数字识别，基于最小风险分类
import os
import cv2
import csv
import sys
import pandas as pd
# 1.样本特征，不需预处理
# 2.图像压缩表示，28*28——7*7，存储于(picture_bayes(i).txt)
# 3.根据数字类别分别存储于不同的txt
# 计算Pj(Wi)，在此基础上计算P(X/Wi)
# 返回一个double类型数据，为在某一类数字（0-9）的特征空间中出现X的概率
# 某类数字的特征空间中出现该样本X的概率

f_sum = open('sum.csv','r')
f_txt = open('TXT.csv','r')
f_number = open('number.csv','r')
f_sum_read =f_sum.read()
f_txt_read =f_txt.read()
f_number_read =f_number.read()
def Class_conditional_pro(test_str, n):
    m = 0
    file = open(
        "D:/file/work_space/deep_leaning/bytes/mnist_data_jpg/mnist_train_jpg_60000."
        + str(n) + ".txt", 'r')      # 只读模式
    p_x_wi = 1.0
    for i in range(len(test_str)):
        sum_pj_wi = 0
        while True:
            line = file.readline()
            m +=1
            if not line:  # 已读完整个文档，光标返回开头，结束此次匹配
                file.seek(0)  # 移动指针
                break
            train_str = line[-197:-1]
            sum_pj_wi += eval(train_str[i])
        if eval(test_str[i]) == 1:
            p_x_wi *= (sum_pj_wi + 1) / (m + 2)
            m=0
        else:
            p_x_wi *= 1 - (sum_pj_wi + 1) / (m + 2)
            m=0
    file.close()
    return p_x_wi
#######################
# 实现图像压缩，28*28——14*14，并以196数据流返回，img_path--图片路径
# 划分为14*14的像素矩阵，其中2个及以上的像素点超过127即记为1，反之为0
# 参数img_path，必须为完成的图片路径
#######################
def Image_Compression(img_path):
    # 数据按行存储
    img_str = ""        # 存数据流
    img = cv2.imread(img_path)
    # print("图像的形状,返回一个图像的(行数,列数,通道数):", img.shape)
    x = y = 0           # img像素点坐标表示img[x][y]
    for k in range(1, 197):      # k的范围1-49
        totle_img = 0           # 统计满足要求的像素点数目
        for i in range(2):
            for j in range(2):
                if img[x + i - 1][y + j - 1][0] > 127:
                    totle_img += 1
        y = (y + 2) % 28
# 一个矩形阵中包含4个像素点，其中像素值大于127的点超过2个记为1，反之记为0
        if totle_img >= 2:
            img_str += '1'
        else:
            img_str += '0'
        if k % 14 == 0:      # 控制x,y的变化
            x = x + 2
            y = 0
    return img_str
file_bayes=[]
for i in f_txt.readlines():
    file_bayes.append(int(i))
sum_num = int(f_sum_read)
Priori_pro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m=0
f2 = open("D:/file/work_space/deep_leaning/bytes/朴素贝叶斯/number.csv",'r')
for i in f2.readlines():
    Priori_pro[m] = int(i)
    m+=1
# 3.bayes概率计算，先验概率、类条件概率、后验概率
# 先验概率--Priori_pro
for i in range(10):
    Priori_pro[i] =float(Priori_pro[i])/float(sum_num)
# 类条件概率--Class_pro && 后验概率--Posterior_pro
Class_pro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # 某个样本的类条件概率
Posterior_pro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # 某个样本的后验概率
Correct_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 识别率
root_dir = \
    "D:/file/work_space/deep_leaning/bytes/mnist_data_jpg/mnist_test_jpg_10000"          # 测试test-images数据集
j=0
ii=[0,0,0,0,0,0,0,0,0,0]
jj=[0,0,0,0,0,0,0,0,0,0]
list_0 = []
#R[][]存储决策表，设置主对角线元素为0，其余元素为1，
# 则可以实现使用最小风险决策与朴素贝叶斯决策输出结果一致
R=[]
for i in range(10):
    R.append([])
    for j in range(10):
        if i==j:
            R[i].append(0)
            continue
        R[i].append(1)
for fl in os.listdir(root_dir):  # 循环处理所有test图片
    j=j+1
    # 通过调节n可以调节测试间隔
    n = 100
    if j%n!=0:
        continue
    else:
        class_pic = 0   # 比较得出后验概率最大的图片类别
        max_pxh = 0.00     # 最大的后验概率
        px_h_deno = 0.00   # 后验概率中的分母
        sum = 0.00
        sum_1 = []
        test_img_str = Image_Compression(root_dir + '/' + fl)
        for i in range(10):         # 计算类条件概率
            Class_pro[i] = Class_conditional_pro(test_img_str, i)
            px_h_deno += Priori_pro[i] * Class_pro[i]
        for i in range(10):         # 计算后验概率
            Posterior_pro[i] = Priori_pro[i] * Class_pro[i] / px_h_deno
        #求出最小风险
        for i in range(10):
            for j in range(10):
                sum += Posterior_pro[j]*R[j][i]
            sum_1.append(sum)
            sum = 0
        class_pic = sum_1.index(min(sum_1))
        sum_1 = []
        if class_pic == eval(fl[0]):
            print("测试数字：", fl[0:-4], "  --  识别出来的结果：", class_pic)  # 数字的识别结果
            Correct_rate[class_pic] += 1
            ii[eval(fl[0])]+=1
            jj[eval(fl[0])]+=1
        else:
            print("测试数字：", fl[0:-4], "  --  识别出来的结果：", class_pic, "识别错误！！！")
            ii[eval(fl[0])]+=1
print("------------------------------------------------")
print(ii)
for i in range(10):
    if ii[i] ==0:
        print("数字"+str(i)+"未参加检测")
        i+=1
        continue
    print("数字{:d} 正确率={:.2f}%,错误率 = {:.2f}%".format(i,jj[i]/ii[i]*100,(100-(jj[i]/ii[i])*100)))
print("success！")

