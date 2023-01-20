import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import sklearn.metrics as skmet
from sklearn.preprocessing import MinMaxScaler


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1950))
    return f
def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


df = pd.read_csv('API.csv', skiprows=(4))
namew=['CO2 emissions from solid fuel consumption (kt)','CO2 emissions from liquid fuel consumption (kt)']
name=['CO2 emissions from solid fuel consumption (kt)','CO2 emissions from liquid fuel consumption (kt)','CO2 emissions from gaseous fuel consumption (kt)']
data=df.loc[df['Indicator Name'].isin(name)]
dataa=df.loc[df['Indicator Name'].isin(namew)]
namee=['India']
nameq=['India','Pakistan']
data=data.loc[data['Country Name'].isin(namee)]
dataa=dataa.loc[dataa['Country Name'].isin(nameq)]
#columns
cols = ['Country Name', 'Indicator Name',
        '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967','1968','1969',
           '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016','2017','2018','2019','2020','2021']

#this line reorients the dataframe


data = data[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(1)
dataq = dataa[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(0)

df_fit = dataq[['India','Pakistan']].copy()
scaler = MinMaxScaler()
scaler=scaler.fit(df_fit[['India']])
df_fit['India']=scaler.transform(df_fit[['India']])
scaler=scaler.fit(df_fit[['Pakistan']])
df_fit['Pakistan']=scaler.transform(df_fit[['Pakistan']])
print(df_fit)
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))
    
    

kmeans = cluster.KMeans(n_clusters=5)

kmeans.fit(df_fit) # fit done on country name and level_1 as a year

labels = kmeans.labels_
cen= kmeans.cluster_centers_
df_fit['cluster']=labels
#clusters

d1=df_fit[df_fit.cluster==0]
d2=df_fit[df_fit.cluster==1]
d3=df_fit[df_fit.cluster==2]
d4=df_fit[df_fit.cluster==3]
d5=df_fit[df_fit.cluster==4]

plt.scatter(d1[["India"]], d1[["Pakistan"]] )
plt.scatter(d2[["India"]], d2[["Pakistan"]] )
plt.scatter(d3[["India"]], d3[["Pakistan"]])
plt.scatter(d4[["India"]], d4[["Pakistan"]])
plt.scatter(d5[["India"]], d5[["Pakistan"]])
plt.scatter(cen[:, 0], cen[:, 1], s=100, c='black', label = 'Centroids')
plt.legend(["Cluster0","Cluster1","Cluster2","Cluster3","Cluster4", "Centroid"])
plt.title("India vs Pakistan CO2 emissions liquid fuel consumption (kt)")
plt.xlabel("C02 Emission liquid fuel")
plt.ylabel("C02 Emission liquid fuel")


year=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967','1968','1969',
           '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016']

data["Year"]=year

data["Year"]=pd.to_numeric(data["Year"])
#fitting

#exponential function
popt, covar = opt.curve_fit(exp_growth, data["Year"],data["CO2 emissions from solid fuel consumption (kt)"], p0=[4e8, 0.02])



data["pop_exp"] = exp_growth(data["Year"], *popt)
plt.figure()
plt.plot(data["Year"], data["CO2 emissions from solid fuel consumption (kt)"], label="CO2 emissions from solid fuel consumption (kt)")
plt.plot(data["Year"], data["pop_exp"], label="fit")
plt.legend()
plt.title("CO2 solid fuel consumption fit ")
plt.xlabel("year")
plt.ylabel("CO2 emissions from solid fuel consumption (kt)")
plt.show()


"""
#logistic function
popt, covar = opt.curve_fit(logistics, data["Year"],data["CO2 emissions from solid fuel consumption (kt)"], p0=(2e9, 0.05, 1990.0))

data["pop_expp"] = logistics(data["Year"], *popt)
plt.figure()
plt.plot(data["Year"], data["CO2 emissions from solid fuel consumption (kt)"], label="CO2 emissions from solid fuel consumption (kt)")
plt.plot(data["Year"], data["pop_expp"], label="fit")
plt.legend()
plt.title("CO2 solid fuel consumption fit logistic ")
plt.xlabel("year")
plt.ylabel("CO2 emissions from solid fuel consumption (kt)")
plt.show()

"""

#exponential function with highlighted graph
popt, covar = opt.curve_fit(exp_growth, data["Year"],data["CO2 emissions from solid fuel consumption (kt)"], p0=(4e8, 0.02))
sigma = np.sqrt(np.diag(covar))

low, up = err_ranges(data["Year"], exp_growth, popt, sigma)

data["pop_log"] = exp_growth(data["Year"], *popt)
plt.figure()
plt.plot(data["Year"], data["CO2 emissions from solid fuel consumption (kt)"], label="CO2 emissions from solid fuel consumption (kt)")
plt.plot(data["Year"], data["pop_log"], label="forecast")
plt.fill_between(data["Year"], low, up, alpha=0.7,color="yellow")
plt.legend()
plt.title("CO2 solid fuel consumption fit ")
plt.xlabel("year")
plt.ylabel("CO2 emissions from solid fuel consumption (kt)")
plt.show()

#Forcasted population
print("Forcasted population")
low, up = err_ranges(2020, exp_growth, popt, sigma)
print("2020 between ", low, "and", up)
low, up = err_ranges(2030, exp_growth, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err_ranges(2040, exp_growth, popt, sigma)
print("2040 between ", low, "and", up)

#forecasted population with mean
print("Forcasted population")
low, up = err_ranges(2020, exp_growth, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2020:", mean, "+/-", pm)
low, up = err_ranges(2030, exp_growth, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err_ranges(2040, exp_growth, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)

#heatmap for india with 3 indicators
plt.figure(figsize=(8,6))
heat_data=data[["CO2 emissions from solid fuel consumption (kt)",'CO2 emissions from liquid fuel consumption (kt)','CO2 emissions from gaseous fuel consumption (kt)']]
heat_data.rename(columns = {'CO2 emissions from solid fuel consumption (kt)':'CO2 emissions Solid fuel (kt)', 'CO2 emissions from liquid fuel consumption (kt)':'CO2 emissions liquid fuel (kt)',
                              'CO2 emissions from gaseous fuel consumption (kt)':'CO2 emissions gaseous fuel (kt)'}, inplace = True)
corr=heat_data.corr()
ax = plt.axes()
sns.heatmap(data=corr, annot=True,cmap="turbo_r",ax = ax)

plt.title('India', fontsize = 24) # title with fontsize 20


#heatmap for india with 3 indicators
plt.figure(figsize=(8,6))

heat_data=data[["CO2 emissions from solid fuel consumption (kt)",'CO2 emissions from liquid fuel consumption (kt)','CO2 emissions from gaseous fuel consumption (kt)']]
heat_data.rename(columns = {'CO2 emissions from solid fuel consumption (kt)':'CO2 emissions Solid fuel (kt)', 'CO2 emissions from liquid fuel consumption (kt)':'CO2 emissions liquid fuel (kt)',
                              'CO2 emissions from gaseous fuel consumption (kt)':'CO2 emissions gaseous fuel (kt)'}, inplace = True)
corr=heat_data.corr()
ax = plt.axes()
sns.heatmap(data=corr, annot=True,cmap="turbo_r",ax = ax)

plt.title('India', fontsize = 24) # title with fontsize 20



#heatmap for pakistan with 3 indicators
namee=['CO2 emissions from solid fuel consumption (kt)','CO2 emissions from liquid fuel consumption (kt)','CO2 emissions from gaseous fuel consumption (kt)']
dataa=df.loc[df['Indicator Name'].isin(namee)]

namepak=['Pakistan']
dataa=dataa.loc[dataa['Country Name'].isin(namepak)]
#columns
colls = ['Country Name', 'Indicator Name',
        '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967','1968','1969',
           '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016','2017','2018','2019','2020','2021']

#this line reorients the dataframe


df_dataa = dataa[colls].set_index(['Country Name', 'Indicator Name']).stack().unstack(1)


plt.figure(figsize=(8,6))
heat_dataa=df_dataa[["CO2 emissions from solid fuel consumption (kt)",'CO2 emissions from liquid fuel consumption (kt)','CO2 emissions from gaseous fuel consumption (kt)']]
heat_dataa.rename(columns = {'CO2 emissions from solid fuel consumption (kt)':'CO2 emissions Solid fuel (kt)', 'CO2 emissions from liquid fuel consumption (kt)':'CO2 emissions liquid fuel (kt)',
                              'CO2 emissions from gaseous fuel consumption (kt)':'CO2 emissions gaseous fuel (kt)'}, inplace = True)
cor=heat_dataa.corr()
ax = plt.axes()
sns.heatmap(data=cor, annot=True,cmap="YlGnBu",ax = ax)
plt.title('Pakistan', fontsize = 24) # title with fontsize 24