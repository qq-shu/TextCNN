#统计每个样本含有的序列长度
file=open('train-ops.txt')
ops=[]
import pandas as pd
for line in file.readlines():
#     print(line)
    curLine=line.strip().split(" ")
#     floatLine=list(map(float,curLine))
    ops.append(curLine[:])
print(len(ops))
print(ops[7940])
print(len(ops[7940]))

#删除操作码数为0的样本
blankline = []
def tong1(filename):
    with open(filename, 'r') as f:
        num = 0
        for line in f:
            num += 1
            if len(line) == 1:
                blankline.append(num)
              #  print(num)  
        print('%s' % num)
file1 = 'train-ops.txt'
value = num = 0
tong1(file1)
print(blankline)
print(len(blankline))
#删除train-ops.txt文件的空行
with open("train-ops.txt","r",encoding="utf-8") as f:
    lines = f.readlines()
    #print(lines)
with open("del-train-ops.txt","w",encoding="utf-8") as f_w:
    for line in lines:
        if len(line) ==1 :
            continue
        f_w.writelines(line)