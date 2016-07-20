import numpy as np

fr = open('E:/tianchi/test_anaylsis/data/artist_all_play.txt','r')
fw = open('E:/tianchi/test_anaylsis/data/artist_all_std.txt','w')

line = fr.readline()
while line:
    name = line
    fw.write(name)
    play = fr.readline().strip('\n').split(',')
    iplay = map(int,play)
    temp = []
    for i in iplay:
        temp.append(i)
    sumplay = np.sum(temp)
    sqrtplay = str(np.sqrt(sumplay))
    fw.write(sqrtplay+'\n')
    line = fr.readline()
fr.close()
fw.close()
    
