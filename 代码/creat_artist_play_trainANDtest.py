START_UNIX 	=1425139200
DAY_SECOND 	=86400
import time
def date2Num(date):
    return (int(time.mktime(time.strptime(date,'%Y%m%d')))-START_UNIX)//DAY_SECOND  #计算第几天
Days = (int(time.mktime(time.strptime('20150630','%Y%m%d')))-START_UNIX)//DAY_SECOND
with open('E:/tianchi/test_anaylsis/artist2.txt','r') as fr:
    play = fr.readline().strip('\n').split(',')
    train = play[:Days]
    test = play[Days:]
with open('E:/tianchi/test_anaylsis/artist2_train.txt','w') as fw:
    write_train = ','.join(train)
    fw.write(write_train)
with open('E:/tianchi/test_anaylsis/artist2_test.txt','w') as fw2:
    write_test = ','.join(test)
    fw2.write(write_test)
fr.close()
fw.close()
fw2.close()
