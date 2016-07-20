fr = open('E:/tianchi/test_anaylsis/artist_all_innormal.txt','r')
fw = open('E:/tianchi/test_anaylsis/artist_all_innormal_weekend.txt','w')
def create_week(start,lenght):
    week = []
    w = start
    for i in range(0,lenght):
        if w == 7:
            ws = str(w)
            week.append(ws)
            w = 1
        elif w < 7:
            ws = str(w)
            week.append(ws)
            w = w+1
    return week
def ifweekend(week,lenght):
    weekend = []
    for i in week:
        if i == '6' or i == '7':
            weekend.append('1')
        else :
            weekend.append('0')
    return weekend

line1 = fr.readline()
while line1:
    name = line1
    fw.write(name)
    line2 = fr.readline()
    fw.write(line2)
    play = line2.strip('\n').split(',')
    lenght = len(play)
    week = create_week(7,lenght)
    weekend = ifweekend(week,lenght)
    fw.write(','.join(weekend)+"\n")
    print ('ok')
    line1 = fr.readline()

fr.close()
fw.close()
