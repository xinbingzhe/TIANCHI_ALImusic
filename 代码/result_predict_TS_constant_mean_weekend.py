from sklearn.linear_model import BayesianRidge
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.ensemble.forest import RandomForestRegressor
import statsmodels.api as sm

### err var ###
def rmse(y_test, y_predict):  
    #return sp.sqrt(sp.mean((y_test - y) ** 2))  
     return np.mean((y_test - y_predict) ** 2)

### polynomial ###
def polynomial(x_train,y_train,x_test,degree):  
     clf = Pipeline([('poly', PolynomialFeatures(degree=degree)),(' Ridge',  Ridge(fit_intercept=True))])  
     clf.fit(x_train, y_train)  
     y_predict = clf.predict(x_test)
     
     #y_predict_list.append(y_predict)
     #rmsevalue = rmse(y_test,y_predict)      
     return y_predict
### RandomForest ###
def RandomForest(x_train,y_train,x_test,degree):    
     params = {'n_estimators': 1000, 'max_depth': degree, 'min_samples_split': 1,'warm_start':True}
     clf = RandomForestRegressor(**params)
     clf.fit(x_train, y_train)          
     y_predict = clf.predict(x_test)
     #plt.plot(x_test,y_predict,color='red')
     return y_predict
### timeseries ###
def TS(dta,p,q):
    arma_mod2 = sm.tsa.ARMA(dta,(p,q)).fit()
    predict_pandas = arma_mod2.predict('20150831', '20151029', dynamic=True)
    
    y_predict = predict_pandas.values
    plt.plot(predict_pandas,color='red')
    return y_predict
### constant ###
def constant_value(play):
    temp = []
    for i in range(0,60):
        temp.append(play)
    y_predict = temp
    return y_predict
### mean_value ###
def mean_value(y_train,degree):
    value = y_train[-degree:]
    mean_value = int(np.mean(value))
    temp = []
    for i in range(0,60):
        temp.append(mean_value)
    y_predict = temp
    return y_predict
### create datelist ###
import datetime
def datelist(start, end):
    start_date = datetime.date(*start)
    end_date = datetime.date(*end)

    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result

### create_week ###
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
        if i == 6 or i ==7:
            weekend.append(1)
        else :
            weekend.append(0)
    return weekend

fr1 = open('E:/tianchi/season2(1)/p2/data/p2_artist_play_week.txt','r')
fr2 = open('E:/tianchi/season2(1)/p2/data/model/train_model_with_p1.txt','r')
fw = open('E:/tianchi/season2(1)/p2/data/result/predict_train_model_with_p1.csv','w')
line1 = fr1.readline()
line2 = fr2.readline()
#i = 0
datelist = datelist((2015,9,1), (2015,10,30))
while line1:
    name = line1
    #fw.write(name)
    ### create y_train ###
    y_train = fr1.readline().strip('\n').split(',')
    y_train = map(float,y_train)
    temp = []
    for i in y_train:
        temp.append(i)
    
    y_train = np.array(temp)
    days = y_train.shape
    days = days[0]
    ### create X_train ###
    x_train = np.arange(0,days)

    w= fr1.readline().strip('\n').split(',')
    w = map(int,w)
    tempw = []
    for i in w:
        tempw.append(i)
    weekend = np.array(tempw)

    X_train = []
    for (i,j) in zip(x_train,weekend):
        X_train.append([i,j])
    
    ### create X_test ###
    x_test = np.arange(days,days+60)
    week_test = create_week(2,60)
    weekend_test = ifweekend(week_test,60)
    X_test = []
    for (i,j) in zip (x_test,weekend_test):
            X_test.append([i,j])
    
    ### useModel and degree ###
    union = line2.strip('\n').split(',')
    artist = union[0]
    model = union[1]
    degree = int(union[2])
    
    ### fit and predict ###
    y_predict = []
    
    if model == 'p':
        y_predict = polynomial(X_train,y_train,X_test,degree)
        plt.scatter(x_train,y_train)
        plt.plot(x_test,y_predict,color='red')
    elif model =='r':
        y_predict = RandomForest(X_train,y_train,X_test,degree)
        plt.scatter(x_train,y_train)
        plt.plot(x_test,y_predict,color='red')
    elif model =='ts':
        dta = pd.Series(temp)
        dta.index = pd.Index(pd.date_range('20150301', periods=days))
        y_predict =  TS(dta,1,0)
        plt.scatter(dta.index,dta.values)
        #plt.plot(x_test,y_predict,color='red')
    elif model =='cons':
        y_predict = constant_value(degree)
        plt.scatter(x_train,y_train)
        plt.plot(x_test,y_predict,color='red')
    elif model =='mean':
        y_predict = mean_value(y_train,degree)
        plt.scatter(x_train,y_train)
        plt.plot(x_test,y_predict,color='red')
    else :
        print('failed')
        break
     
   
    ### output predict ###
    for (value,date) in zip(y_predict,datelist):
         valuei = int(value) 
         values = str(valuei)
         fw.write(artist+","+values+","+date+"\n")
    
    
    ### plot ###
     
    #plt.scatter(x_train,y_train)
    
    plt.grid()
    path = "E:/tianchi/season2(1)/p2/result_train_p1_fig/"+name.strip('\n')    
    plt.savefig(path+".png")
    plt.clf()#清除图像，所有的都画到一起了
    line1 = fr1.readline()
    line2 = fr2.readline()

    #i = i+1

fr1.close()
fr2.close()
fw.close()

    
    
