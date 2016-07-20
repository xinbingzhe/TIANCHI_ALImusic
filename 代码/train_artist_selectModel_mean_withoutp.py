from sklearn.linear_model import BayesianRidge
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.ensemble.forest import RandomForestRegressor

from scipy import  stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

### err var ###
def rmse(y_test, y_predict):  
    #return sp.sqrt(sp.mean((y_test - y) ** 2))  
     return np.mean((y_test - y_predict) ** 2)

### polynomial ###
'''def polynomial(x_train,y_train,x_test,y_test):
     degree = [1,2,3]
     result = {}
     rmse_list = []
     for d in degree:      
          clf = Pipeline([('poly', PolynomialFeatures(degree=d)),(' Ridge',  Ridge(fit_intercept=True))])  
          clf.fit(x_train[:, np.newaxis], y_train)  
          y_predict = clf.predict(x_test[:, np.newaxis])
          #y_predict_list.append(y_predict)
          rmsevalue = rmse(y_test,y_predict)
          result[rmsevalue] = [y_predict,d]
          rmse_list.append(rmsevalue)
     rmseMin = min(rmse_list)          
     return rmseMin,result[rmseMin]'''
### RandomForest ###
def RandomForest(x_train,y_train,x_test,y_test):
     degree = [1,2,3,4,7]
     result = {}
     rmse_list = []
     for d in degree:
          params = {'n_estimators': 1000, 'max_depth': d, 'min_samples_split': 1,'warm_start':True}
          clf = RandomForestRegressor(**params)
          clf.fit(x_train[:, np.newaxis], y_train)
          y_predict = clf.predict(x_test[:, np.newaxis])
          rmsevalue = rmse(y_test,y_predict)
          result[rmsevalue] = [y_predict,d]
          rmse_list.append(rmsevalue)
     rmseMin = min(rmse_list)     
     return rmsevalue,result[rmseMin]
### mean ###
def mean(x_train,y_train,x_test,y_test):
     degree = [10,15,20]
     result = {}
     rmse_list = []
     for d in degree:
        value = y_train[-d:]
        mean_value = int(np.mean(value))    	  
        temp = []
        for i in range(0,16):
            temp.append(mean_value)
        y_predict = temp
        rmsevalue = rmse(y_test,y_predict)
        result[rmsevalue] = [y_predict,d]
        rmse_list.append(rmsevalue)
     rmseMin = min(rmse_list)
     return rmsevalue,result[rmseMin]
def TS(dta,p,q):
    arma_mod2 = sm.tsa.ARMA(dta,(p,q)).fit()
    predict_pandas = arma_mod2.predict('20150815', '20150830', dynamic=True)
    
    y_predict = predict_pandas.values
    #plt.plot(predict_pandas,color='red')
    return y_predict
def ts(dta,y_test):
    y_predict = TS(dta,1,0)
    rmsevalue = rmse(y_test,y_predict)
    return rmsevalue,y_predict

    
### selectModel ###   
def selectModel(rmsevalue_p,rmsevalue_r):
    if rmsevalue_p < rmsevalue_r:
        return rmsevalue_p,"p"
    elif rmsevalue_p >= rmsevalue_r:
        return rmsevalue_r,"r"
    
    
fr1 = open('E:/tianchi/season2(1)/p2/data/p2_new_artist_all_train.txt','r')
fr2 = open('E:/tianchi/season2(1)/p2/data/p2_new_artist_all_test.txt','r')
#fw = open('E:/tianchi/test_anaylsis/artist_all_predict_combine2.txt','w')
fw2 = open('E:/tianchi/test_anaylsis/train_Model1_withoutp.txt','w')
line1 = fr1.readline()
line2 = fr2.readline()
#i = 0
while line1:
    name = line1

    ### create y_train ###
    y_train = fr1.readline().strip('\n').split(',')
    y_train = map(float,y_train)
    temp = []
    for i in y_train:
        temp.append(i)
    y_train = np.array(temp)
    dta = pd.Series(temp)  #ts
    ts_y_train = np.array(temp)  #ts             
    days = y_train.shape
    days = days[0]            
    ### create x_train ###
    x_train = np.arange(0,days)
    dta.index = pd.Index(pd.date_range('20150721', periods=days))
    ### create y_test ###
    y_test = fr2.readline().strip('\n').split(',')
    y_test = map(float,y_test)
    temp = []
    for i in y_test:
        temp.append(i)
    y_test  = np.array(temp)
    days2 = y_test.shape
    days2 = days2[0]

    ### create x_test ###
    x_test = np.arange(days,days+days2)


    ### polynomianl ###
    #rmsevalue_p,y_predict_p_d = polynomial(x_train,y_train,x_test,y_test)
	
    ### RandomForest ###
    rmsevalue_r,y_predict_r_d = RandomForest(x_train,y_train,x_test,y_test)
    
    ### mean ###
    rmsevalue_m,y_predict_m_d = mean(x_train,y_train,x_test,y_test)

    ### timeseries ###
     
    rmsevalue_t,y_predict_t_d = ts(dta,y_test)
   
    ### selectModel ###

    model_degree = {'r':str(y_predict_r_d[1]),'m':str(y_predict_m_d[1]),'ts':'0'}
    rmsemin = {rmsevalue_r:'r',rmsevalue_m:'m',rmsevalue_t:'ts'}
    rmse_min = min(rmsemin.keys())
    min_model = rmsemin[rmse_min]
    degree = model_degree[min_model]                                                 
    fw2.write(name.strip("\n")+","+min_model+","+degree+"\n")
   
    ''''if rmsevalue_p < rmsevalue_r and np.mean(y_predict_p_d[0])>0:
   	    if rmsevalue_p<rmsevalue_m and np.mean(y_predict_m_d[0])>0:
   	   	   degree = str(y_predict_p_d[1])
        	   fw2.write(name.strip("\n")+","+"p"+","+degree+"\n")
          elif rmsevalue_m<rmsevalue_r and np.mean(y_predict_m_d[0])>0: 
         	 degree = str(y_predict_m_d[1])
        	  fw2.write(name.strip("\n")+","+"m"+","+degree+"\n")
         elif np.mean(y_predict_r_d[0])>0:
        	degree = str(y_predict_r_d[1])
        	 fw2.write(name.strip("\n")+","+"r"+","+degree+"\n")
         else :
        	print("failed1")
        elif rmsevalue_p >= rmsevalue_r and np.mean(y_predict_r_d[0])>0:
   	    if rmsevalue_r<rmsevalue_m and np.mean(y_predict_r_d[0])>0:
   	   	   degree = str(y_predict_r_d[1])
        	   fw2.write(name.strip("\n")+","+"r"+","+degree+"\n")
         elif rmsevalue_p<= rmsevalue_m and np.mean(y_predict_p_d[0])>0: 
         	 degree = str(y_predict_p_d[1])
        	  fw2.write(name.strip("\n")+","+"p"+","+degree+"\n")
         elif np.mean(y_predict_m_d[0])>0:
        	degree = str(y_predict_m_d[1])
        	 fw2.write(name.strip("\n")+","+"m"+","+degree+"\n")
         else :
        	print("failed2")'''
    #rmsevalue,model = selectModel(rmsevalue_p,rmsevalue_r)


    ### plot ###
    #plt.scatter(x_train,y_train)
    #plt.plot(x_test,y_predict_p_d[0],color='red')
    #plt.plot(x_test,y_predict_r_d[0],color='green')
    #plt.plot(x_test,y_predict_m_d[0],color='blue')
    #plt.scatter(x_test,y_test,color='yellow')
    #path = "E:/tianchi/test_anaylsis/artist_all_predict_figs/"+name.strip('\n')    
    #plt.savefig(path+".png")
    #plt.clf()#清除图像，所有的都画到一起了

    line1 = fr1.readline()
    line2 = fr2.readline()
    print(ok)
    #i = i+1

fr1.close()
fr2.close()
fw2.close()

    
    
