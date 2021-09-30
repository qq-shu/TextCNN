file=open('train-ops.txt')
ops=[]
import pandas as pd
for line in file.readlines():
#     print(line)
    curLine=line.strip().split(" ")
#     floatLine=list(map(float,curLine))
    ops.append(curLine[:])
print(len(ops))

# print(ops)
length = []
for i in ops:
    length.append(len(i))
    #print(len(i))
# print(len(ops))
#print(length)
name = ["length"]
tt = pd.DataFrame(columns=name,data=length)
tt.to_csv('train-length.csv')