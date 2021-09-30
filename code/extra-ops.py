#!python
import re
import pandas as pd
from collections import *
# 从.asm文件获取Opcode序列
def getOpcodeSequence(filename):
    opcode_seq = []
    p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    with open(filename,encoding='ISO-8859-1') as f:
        for line in f:
            if line.startswith(".text"):
                m = re.findall(p,line)
                if m:
                    opc = m[0][1]
                    if opc != "align":
                        opcode_seq.append(opc)
    return opcode_seq

basepath = "C:/Users/12418/Documents/WeChat Files/wxid_f8v2we60kk4a22/FileStorage/File/2020-07/恶意代码/wuyunxunlianshuju/"
subtrain = pd.read_csv('subtrainLabels.csv')
# 样例有900个文件
ops = []
for sid in subtrain.Id:
    filename = basepath + sid + ".asm"
    print(filename)
    ops1 = getOpcodeSequence(filename)
    ops.append(ops1)
print(len(ops))     #得到所有文件的操作码长度为900

# 将操作码保存为txt文件
def text_save(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")
text_save('opcode.txt', ops)