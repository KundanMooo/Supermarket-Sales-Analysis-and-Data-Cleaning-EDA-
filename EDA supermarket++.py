import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('D:/DS/resume projects/supermarket EDA/columns_description.csv')
dfdict=pd.read_csv('D:/DS/resume projects/supermarket EDA/data_dict.csv')

#Glancing data 
df.head()
dfdict

#overviewing data
df.style.background_gradient(cmap='cool')
df.describe().style.background_gradient(cmap='hot')
df.info()

# cleaning data 
f=(df.isna().mean()*100).sort_values(ascending=False)
sns.barplot(x=f.index,y=f.values)
plt.xticks(rotation=90)
plt.show()

#orignal data in df and manupulation in sdf
sdf=df

#Rating is summeticaly distributed replacing null with mean
b=df.Rating.value_counts()
sns.kdeplot(b.values)
plt.xticks(rotation=90)
plt.show()

sdf['Rating']=df['Rating'].fillna(df.Rating.mean())

# filling null of tax column by calculating it from total column
sdf['Tax 5%']=df['Tax 5%'].fillna(df['Total']*.05)


# filling null values in categorical column by mode 
def repmode(col):
    sdf[col]=df[col].fillna(df[col].mode()[0])

repmode('Product line')
repmode('Gender')
repmode('Customer type')
repmode('Payment')

# Branch column need to be cleaned
f=(df.isna().mean()*100).sort_values(ascending=False)
sns.barplot(x=f.index,y=f.values)
plt.xticks(rotation=90)
plt.show()

sdf.groupby(['City','Branch'])['Invoice ID'].count()
sdf.loc[(sdf['Branch'].isnull()) & (sdf['City'] == 'Mandalay'), 'Branch'] = 'B'
sdf.loc[(sdf['Branch'].isnull()) & (sdf['City'] == 'Naypyitaw'), 'Branch'] = 'C'
sdf.loc[(sdf['Branch'].isnull()) & (sdf['City'] == 'Yangon'), 'Branch'] = 'A'


sdf.info()

#correcting datatype of time,date column
sdf['Date']=sdf['Date'].astype('datetime64')
sdf['Time']=sdf['Time'].astype('datetime64')
sdf.info()
sdf['Month']=sdf['Date'].dt.month
sdf['Year']=sdf['Date'].dt.year
sdf['Day']=sdf['Date'].dt.day
sdf['Weekday']=sdf['Date'].dt.day_name()
sdf['Qruarter']=sdf['Date'].dt.quarter
#df['week_number'] = df['Date'].dt.week
df['annual_week_number'] = df['Date'].dt.isocalendar().week

sdf['Hour']=sdf['Time'].dt.hour
sdf['Minute']=sdf['Time'].dt.minute

#EDA 

#1
sns.violinplot(y=sdf['Unit price'])
plt.show()

#2
def fun(col):
    fig,ax=plt.subplots(4,1,figsize=(7,5))
    sns.swarmplot(data=sdf, x=col, ax=ax[0])
    ax[0].set_title('swarmplot of '+col)

    sns.stripplot(data=sdf, x=col, ax=ax[1])
    ax[1].set_title('stripplot of '+col)
    
    sns.violinplot(data=sdf, x=col, ax=ax[2])
    ax[2].set_title('violinplot of '+col)
    
    sns.kdeplot(sdf[col],ax=ax[3])
    ax[3].set_title('kdeplot of '+col)
    plt.tight_layout()
    plt.show()

fun('Unit price')
fun('Quantity')

#3
sdf.Gender.value_counts().plot.pie(autopct='%1.2f%%')

#4
sns.heatmap(sdf[['Unit price','Quantity','Tax 5%','Total','gross income']].corr(),cmap='cool')

#5
#plt.pie(labels=sdf.Branch,value=sdf.Total)
h=sdf.groupby(['Branch'])['Total'].sum()
plt.pie(h.values,labels=h.index,autopct='%1.1f%%')
plt.title('Toatal sales from each branch')
plt.show()

#6 Univariate analysis 
def ubit(col):
    
    a=(sdf.groupby(sdf[col])['Total'].sum()).sort_values(ascending=False)
    b=(sdf.groupby([sdf[col]])['Total'].mean()).sort_values(ascending=False)
    c=(sdf.groupby([sdf[col]])['Rating'].mean()).sort_values(ascending=False)
    d=(sdf.groupby([sdf[col]])['Quantity'].mean()).sort_values(ascending=False)
    
    
    fig,ax=plt.subplots(2,2,figsize=(15,5))
        
    sns.lineplot(x=a.index, y=a.values, ax=ax[0, 0])
    ax[0,0].set_title('Total sales by'+col)
    
    ax[0,1].bar(height=b.values,x=b.index)
    ax[0,1].set_title('Mean sales by'+col)
    
    ax[1,0].barh(y=c.index, width=c.values)
    ax[1,0].set_title('Mean Rating by'+col)
    
    ax[1,1].barh(y=d.index, width=d.values)
    ax[1,1].set_title('Mean Rating by'+col)
    
    plt.tight_layout()
    
ubit('Hour')

#7 Bivariate analysis 
kk=sdf[['Unit price','Quantity','Tax 5%','Total','gross income']].corr()
sm=kk.corr()
sm=np.tril(sm)
sns.heatmap(sm)