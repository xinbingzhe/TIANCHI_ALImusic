# -*- coding: utf-8 -*-


import os,sys,time

import csv
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

CURRENT_PATH=sys.path[0]
ARTIST_FOLDER=os.path.join(CURRENT_PATH,'artist')//拼接路径

ARTIST=os.path.join(CURRENT_PATH,'mars_tianchi_songs.csv')
SONGS=os.path.join(CURRENT_PATH,'mars_tianchi_user_actions.csv')

SONG_P_D_C=os.path.join(CURRENT_PATH,'song_p_d_c.txt')
ARTIST_P_D_C=os.path.join(CURRENT_PATH,'artist_p_d_c.txt')

SONG_FAN=os.path.join(CURRENT_PATH,'song_fan.txt')
ARTIST_FAN=os.path.join(CURRENT_PATH,'artist_fan.txt')

DAYS=183		#HOW MANY DAYS YOU WANT TO RECORD.


START_UNIX 	=1425139200
DAY_SECOND 	=86400



'''
date:
    %Y%m%d 20150301
'''
def date2Num(date):
    return (int(time.mktime(time.strptime(date,'%Y%m%d')))-START_UNIX)//DAY_SECOND  #计算第几天

class artist(object):
    #ARTIST_PLAY_FOLDER=''
    #ARTIST_FAN_FOLDER=''   only one picture, so just palce it in the root folder.
    SONG_PLAY_FOLDER = ""
    SONG_FAN_FOLDER = ""

    ARTIST_ID = ""
    FPATH = ""

    def __init__(self,artist_id):
        self.ARTIST_ID = artist_id

        self.FPATH = os.path.join(ARTIST_FOLDER, artist_id)#一个艺人一个文件
        if not os.path.exists(self.FPATH):
            os.mkdir(self.FPATH)

        self.SONG_PLAY_FOLDER = os.path.join(self.FPATH, "song_play")
        self.SONG_FAN_FOLDER = os.path.join(self.FPATH, "song_fan")
        if not os.path.exists(self.SONG_PLAY_FOLDER):
            os.mkdir(self.SONG_PLAY_FOLDER)
        if not os.path.exists(self.SONG_FAN_FOLDER):
            os.mkdir(self.SONG_FAN_FOLDER)
                
    def plot_artist_play(self):#画 艺人的播放量图
        ylabel = "count"
        xlabel = "days"
        with open(ARTIST_P_D_C, "r") as fr:
            artist_id = fr.readline().strip("\n")#去掉\n
            while artist_id:
                play = list(map(int, fr.readline().strip("\n").split(","))) #读入播放量，把后面的每个元素都变成int
                download = list(map(int, fr.readline().strip("\n").split(",")))
                collect = list(map(int, fr.readline().strip("\n").split(",")))

                if artist_id==self.ARTIST_ID:
                    p = plt.plot(play, "bo", play, "b-", marker="o")
                    d = plt.plot(download, "ro", download, "r-", marker="o")
                    c = plt.plot(collect, "go", collect, "g-", marker="o")
                    plt.legend([p[1], d[1],c[1]], ["play", "download","collect"])#图例
                    plt.title(artist_id)

                    plt.xlabel(xlabel)#paint x
                    plt.ylabel(ylabel)#paint y
                    plt.savefig(os.path.join(self.FPATH, "play.png")) #save fig
                    plt.clf()#清除图形
                    break
                artist_id = fr.readline().strip("\n")
		
    def plot_artist_fan(self): #画艺人每天的用户数量
        ylabel = "count"
        xlabel = "days"
        with open(ARTIST_FAN, "r") as fr:
            artist_id=fr.readline().strip("\n")
            while artist_id:
                fan = list(map(int, fr.readline().strip("\n").split(",")))
                if artist_id==self.ARTIST_ID:
                    f = plt.plot(fan, "bo", fan, "b-", marker="o")
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.title(artist_id)
                    plt.legend([f[1]], ["artist fans"])
                    plt.savefig(os.path.join(self.FPATH, "fan.png"))
                    plt.clf()
                    break
                artist_id=fr.readline().strip("\n")
                
    def plot_song_play(self):#歌曲播放量图
        ylabel = "count"
        xlabel = "days"
        songs = self.getSongsListByArtist_id()
        with open(SONG_P_D_C, "r") as fr:
            songs_id = fr.readline().strip("\n")
            while songs_id:
                play = list(map(int, fr.readline().strip("\n").split(",")))
                download = list(map(int, fr.readline().strip("\n").split(",")))
                collect = list(map(int, fr.readline().strip("\n").split(",")))
                if songs_id in songs:
                    p = plt.plot(play, "bo", play, "b-", marker="o")
                    d = plt.plot(download, "ro", download, "r-", marker="o")
                    c = plt.plot(collect, "go", collect, "g-", marker="o")
                    plt.legend([p[1], d[1],c[1]], ["play", "download","collect"])
                    plt.title(songs_id)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.savefig(os.path.join(self.SONG_PLAY_FOLDER, songs_id+".png"))
                    plt.clf()
                songs_id=fr.readline().strip("\n")
                
    def plot_song_fan(self):#歌曲的播放用户图
        ylabel = "count"
        xlabel = "days"
        songs = self.getSongsListByArtist_id()
        with open(SONG_FAN, "r") as fr:
            songs_id = fr.readline().strip("\n")
            while songs_id:
                fan = list(map(int, fr.readline().strip("\n").split(",")))
                if songs_id in songs:
                    f = plt.plot(fan, "bo", fan, "b-", marker="o")
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.title(songs_id)
                    plt.legend([f[1]], ["song fans"])
                    plt.savefig(os.path.join(self.SONG_FAN_FOLDER, songs_id+".png"))
                    plt.clf()
                songs_id=fr.readline().strip("\n")
                
    """
    {songs_id:bool,songs_id:bool,...,songs_id:bool}
    """
    def getSongsListByArtist_id(self): #歌曲字典
        songs = {}
        with open(ARTIST) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            for row in spamreader:
                if self.ARTIST_ID==row[1]:
                    songs[row[0]] = True
        return songs


"""
GOT THE 'PLAY','DOWNLOAD' AND 'COLLECT' TIMES IN 'mars_tianchi_user_action.csv' FILE FOR EVERY DAY.
THEN GOT A 'SONGS' STRUCTURE.
LAST WRITE 'SONGS' INTO SONG.TXT FILE.
SONGS={'SONGS_ID':[ , ,..., ],[[],[],[]...[]],[[],[],[]...[]],'SONGS_ID':[[],[],[]...[]],[[],[],[]...[]],[[],[],[]...[]],'SONGS_ID':[[],[],[]...[]],[[],[],[]...[]],[[],[],[]...[]]...}
user={songs_id:[{},{},{},...,{}],songs_id:[{},{},{},...,{}],songs_id:[{},{},{},...,{}]}
"""
def ifNoSongTXT():
    user = {}
    songs = {} #songs 字典中每个索引有个三维列表，每维列表有days个值 SONGS={'SONGS_ID':[ , ,..., ],[ , ,..., ],[ , ,..., ]],...}
    with open(SONGS) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for row in spamreader:
            if row[1] not in songs:
                songs[row[1]] = [[0 for i in range(DAYS)] for j in range(3)]#产生一个三维列表，每维产生DAYS个0，先执行前面的列表解析
            songs[row[1]][int(row[3])-1][date2Num(row[4])] += 1 #收集日加1      
            
            if row[3] == "1":
                if row[1] not in user:
                    user[row[1]] = [{} for i in range(DAYS)]# 产生DAYS个字典索引
                user[row[1]][date2Num(row[4])][row[0]] = True

    with open(SONG_P_D_C, "w") as fw:  
        for i in songs:
            fw.write(i+"\n")
            fw.write(",".join(str(x) for x in songs[i][0])+"\n")
            fw.write(",".join(str(x) for x in songs[i][1])+"\n")
            fw.write(",".join(str(x) for x in songs[i][2])+"\n")

    with open(SONG_FAN, "w") as fw:
        for i in user:
            fw.write(i+"\n")
            fw.write(",".join(str(len(x)) for x in user[i])+"\n")


"""
BEFORE RUN THIS CODE,PLEASE RUN 'ifNoSongTxt' FIRSTLY!
ARTIST={'ARTIST_ID':[[],[],[]...[]],[[],[],[]...[]],[[],[],[]...[]],'ARTIST_ID':[[],[],[]...[]],[[],[],[]...[]],[[],[],[]...[]],'ARTIST_ID':[[],[],[]...[]],[[],[],[]...[]],[[],[],[]...[]]...}
user={[{user_id:bool...},{user_id:bool...},...,{user_id:bool...}]}
"""
def ifNoArtistTXT():
    user={}
    artist={}
    index={}
    with open(ARTIST) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for row in spamreader:
            index[row[0]] = row[1] #用户与艺人对应
            if row[1] not in artist:
                artist[row[1]] = [[0 for i in range(DAYS)] for j in range(3)]#产生一个三维列表，每维产生DAYS个0，先执行前面的列表解析
                user[row[1]] = [{} for i in range(DAYS)]

    with open(SONG_P_D_C, "r") as fr:
        songs_id=fr.readline().strip("\n")
        while songs_id:
            temp=[]
            play = list(map(int, fr.readline().strip("\n").split(",")))
            download = list(map(int, fr.readline().strip("\n").split(",")))
            collect = list(map(int, fr.readline().strip("\n").split(",")))
            temp.append(play)
            temp.append(download)
            temp.append(collect)
            t=artist[index[songs_id]]#艺人
            for i in range(3):
                for j in range(DAYS):
                    t[i][j]+=temp[i][j]#对每天操作的歌曲总数求和
            artist[index[songs_id]]=t#再赋值回去
            songs_id=fr.readline().strip("\n")

    with open(ARTIST_P_D_C, "w") as fw:
        for i in artist:
            fw.write(i+"\n")
            fw.write(",".join(str(x) for x in artist[i][0])+"\n")
            fw.write(",".join(str(x) for x in artist[i][1])+"\n")
            fw.write(",".join(str(x) for x in artist[i][2])+"\n")

    with open(SONGS) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for row in spamreader:
            if row[1] in index:
                user[index[row[1]]][date2Num(row[4])][row[0]]=True

    with open(ARTIST_FAN, "w") as fw:
        for i in user:
            fw.write(i+"\n")
            fw.write(",".join(str(len(x)) for x in user[i])+"\n")

def testForSongTXT():
    count=0
    with open(SONG_P_D_C, "r") as fr:
        songs_id=fr.readline().strip("\n")
        while songs_id:
            temp=[]
            play=list(map(int, fr.readline().strip("\n").split(",")))
            download=list(map(int, fr.readline().strip("\n").split(",")))
            collect=list(map(int, fr.readline().strip("\n").split(",")))
            for i in play:
                count+=i
            for i in download:
                count+=i
            for i in collect:
                count+=i
            songs_id=fr.readline().strip("\n")
    print(count)    #5652232

	
if __name__ == "__main__":
    ifNoSongTXT()
    ifNoArtistTXT()
    a = artist("0c80008b0a28d356026f4b1097041689")
    a.plot_artist_play()
    a.plot_artist_fan()
    a.plot_song_play()
    a.plot_song_fan()
    #total time 371.4s
