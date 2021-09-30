import re
from collections import *
import os
import pandas as pd
#从.asm文件获取opcode序列
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

def train_opcode_lm(ops, order=4):
    lm = defaultdict(Counter)
    prefix = ["~"] * order
    prefix.extend(ops)
    data = prefix
    for i in range(len(data)-order):
        history, char = tuple(data[i:i+order]), data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.iteritems()]
    outlm = {hist:chars for hist, chars in lm.iteritems()}
    return outlm

#根据opcode序列，统计对应的n-gram
def getOpcodeNgram(ops, n=3):
    opngramlist = [tuple(ops[i:i+n]) for i in range(len(ops)-n)]
    opngram = Counter(opngramlist)
    return opngram

basepath = "C:/Users/12418/Documents/WeChat Files/wxid_f8v2we60kk4a22/FileStorage/File/2020-07/恶意代码/wuyunxunlian/"
map3gram = defaultdict(Counter)
subtrain = pd.read_csv('subtrainLabels.csv')
count = 1
for sid in subtrain.Id:
    print ("counting the 3-gram of the {0} file...".format(str(count)))
    count += 1
    filename = basepath + sid + ".asm"
    ops = getOpcodeSequence(filename)
    op3gram = getOpcodeNgram(ops)
    map3gram[sid] = op3gram

cc = Counter([])
for d in map3gram.values():
    cc += d
selectedfeatures = {}
tc = 0
for k,v in cc.items():
    if v >= 500:             #500表示一个gram出现的次数('mov', 'mov', 'mov'): 380,380<500（小于，不选）用于特征筛选
        selectedfeatures[k] = v            #筛选过的特征{('mov', 'mov', 'mov'): 380，其中k为('mov', 'mov', 'mov')，v为380
        print (k,v)
        tc += 1                            #满足v>=500的特征个数
dataframelist = []
for fid,op3gram in map3gram.items():            #fid:0A32eTdBKayjCWhZqDOQ  op3gram为对应的n-gram
    standard = {}
    standard["Id"] = fid
    for feature in selectedfeatures:
        if feature in op3gram:
            standard[feature] = op3gram[feature]
        else:
            standard[feature] = 0
    dataframelist.append(standard)
df = pd.DataFrame(dataframelist)
df.to_csv("3gramfeature.csv",index=False)
