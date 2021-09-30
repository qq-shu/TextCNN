# coding=gbk
#!python
import re
import pandas as pd
from collections import *  
def getMajorBlock(filename):
    result=[]
    temp=[]
    p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    with open(filename,encoding='ISO-8859-1') as f:
        for line in f:
            if line.startswith(".text"):
                m = re.findall(p,line)
                if m:
                    opc = m[0][1]
                    if opc != "align":
                        temp.append(opc)
                if("CODE XREF" in line):
                    if("call" in temp):
                        for item in temp:
                            result.append(item)
                    temp=[]
    return result
  
basepath = "C:/Users/12418/Documents/WeChat Files/wxid_f8v2we60kk4a22/FileStorage/File/2020-07/恶意代码/malware-classification/train/"


subtrain = pd.read_csv('trainLabels.csv')


ops = []
for sid in subtrain.Id:
    filename = basepath + sid + ".asm"
    print(filename)
    ops1 = getMajorBlock(filename)
    ops.append(ops1)
print(len(ops))

def text_save(filename, data):
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Saved file successfully")
text_save('major-trian-ops.txt', ops)