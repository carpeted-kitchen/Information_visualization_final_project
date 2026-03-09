import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go


import math

data = pd.read_csv("Data.Gov+-+FY25+Q1.csv",dtype = dict({"Primary Export Product NAICS/SIC code" : np.str_, "Multiyear Working Capital Extension": np.str_, "Fiscal Year": np.str_}))
print(data.head(8).to_string)

data.dropna(inplace=True, subset = ['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount', 'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount', 'Fiscal Year'])
#create a pie chart showing each country
#print(data.info())
first_yr = data[(data['Fiscal Year'] == '2007')]
prim_exp_df = pd.DataFrame(first_yr.loc[:,'Primary Exporter State Name'].value_counts())

print(prim_exp_df.head())

valcounts = data['Country'].value_counts().dropna()
new_pd = data
cntry = pd.DataFrame(data["Country"].value_counts())
#Track number of "Other Countries" instances
other_country_count = 0

cntry_clean = cntry.dropna()
cntry_clean.drop("Multiple - Countries", inplace=True)

other_country_list1 = []
for i in cntry_clean.index:
    count = cntry_clean.loc[i, "count"]
    #print(f"{i}: {count}")
    if  count <=350:
        other_country_count += count
        cntry_clean.drop(i, inplace=True)
        other_country_list1.append(i)
#Add "Other Countries" row
print("Other countries: ", other_country_count)
cntry_clean.at["Other Countries", 'count'] = other_country_count

#group by country
cntry_loan_sum = data.loc[:, ['Country', 'Approved/Declined Amount']].groupby('Country').sum()
#Drop NaN values
cntry_loan_sum_clean= cntry_loan_sum.dropna(subset=['Approved/Declined Amount'])
cntry_loan_sum_clean.fillna(0)
cntry_loan_sum_clean.drop("Multiple - Countries", inplace=True)

#Remove outliers by shaving off the bottom 30% of total loan amounts
max_loan_amt = cntry_loan_sum_clean['Approved/Declined Amount'].max()
other_cntry_loan_total = 0
other_countries = []
loan_cntry_avg = cntry_loan_sum_clean['Approved/Declined Amount'].median()
print("Max loan total: ", max_loan_amt)
for i in cntry_loan_sum_clean.index:
    loan_total = cntry_loan_sum_clean.loc[i, "Approved/Declined Amount"]
    if loan_total < (max_loan_amt * 0.10):
        other_countries.append(i)
        cntry_loan_sum_clean.drop(i, inplace=True)
        other_cntry_loan_total += loan_total
print(f"Other countries loan total: {other_cntry_loan_total} {len(other_countries)}")
cntry_loan_sum_clean.at["Other countries", 'Approved/Declined Amount'] = other_cntry_loan_total
print(cntry_loan_sum_clean.head(6))

#Small business loan amount per country
sb_loan_df = data.loc[:,['Country', 'Small Business Authorized Amount']].groupby('Country').sum()
sb_loan_df= sb_loan_df.dropna()
sb_loan_df.drop("Multiple - Countries", inplace=True)

print("Small business loan")
print(sb_loan_df.head(6))
sb_max_loan = sb_loan_df['Small Business Authorized Amount'].max()
sb_other_cntry_loan_total = 0
other_countries_sb = []
for i in sb_loan_df.index:
    smb = sb_loan_df.loc[i, "Small Business Authorized Amount"]
    if smb < sb_max_loan * .10:
        sb_other_cntry_loan_total += smb
        other_countries_sb.append(i)
        sb_loan_df.drop(i, inplace=True)
sb_loan_df.at["Other countries", 'Small Business Authorized Amount'] = sb_other_cntry_loan_total
print(f"Other countries small business : {sb_other_cntry_loan_total}, {len(other_countries_sb)}")


#Get number of small business loans that each country received
nonzero_sb_loan = data[data['Small Business Authorized Amount'] > 0]
sb_cntry = pd.DataFrame(nonzero_sb_loan['Country'].value_counts())
sb_cntry= sb_cntry.dropna()
print(sb_cntry.head(8))
sb_cntry.drop("Multiple - Countries", inplace=True)
other_sb = []
other_countries_loan_received = 0

for i in sb_cntry.index:
    valcount = sb_cntry.loc[i, "count"]
    if valcount <= 190:
        other_sb.append(i)
        sb_cntry.drop(i, inplace=True)
        other_countries_loan_received += valcount
sb_cntry.at["Other countries", 'count'] = other_countries_loan_received
print(f"Other countries - number of small business loans: {other_countries_loan_received}, {len(other_sb)}")

fig1, ax1 = plt.subplots(2,2, figsize = (16,12))
ax1[0,0].pie(data =cntry_clean,
           labels = cntry_clean.index,
                   x = 'count',
                   startangle=0,
                   autopct='%1.2f')
ax1[0,0].set_title("Total loans received")
ax1[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax1[0,1].pie(data= cntry_loan_sum_clean,
           labels = cntry_loan_sum_clean.index,
           x = 'Approved/Declined Amount',
           startangle=0,
           autopct='%1.2f')
ax1[0,1].set_title("Loan amount per country")
ax1[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax1[1,0].pie(data = sb_loan_df,
          x = 'Small Business Authorized Amount',
           labels = sb_loan_df.index,
           startangle=0,
           autopct='%1.2f')
ax1[1,0].set_title("Small business loan amount per country")
ax1[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax1[1,1].pie(data=sb_cntry,
           labels = sb_cntry.index,
           x='count',
           startangle=0,
           autopct='%1.2f')
ax1[1,1].set_title("Number of small business loans per country")
ax1[1,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#Create line plot

#Get total loan amount approves for every fiscal year
loan_approved_fisc = data.loc[data["Primary Exporter State Name"] == "Maryland", ["Fiscal Year", 'Approved/Declined Amount']].groupby(['Fiscal Year']).sum().dropna()
#print(loan_approved_fisc.head())

#Mark ticks at every month
fig, ax = plt.subplots(figsize = (16,8))
#plot
ax.plot(loan_approved_fisc.index, loan_approved_fisc['Approved/Declined Amount'])
#Set ticks every year
ax.grid(True)
ax.set_xlabel('Fiscal Year')
ax.set_ylabel('Approved/Declined Amount')
plt.title("The plot showing approved loans from Maryland")
plt.show()

#Create surface plot
loan_interest_clean = data.loc[(data["Disbursed/Shipped Amount"] !=0) & (data['Fiscal Year'].astype(int) >=2020)
                               & (data['Approved/Declined Amount'] < 5000000),
            ["Approved/Declined Amount", "Disbursed/Shipped Amount","Loan Interest Rate"]]
#fill in NaN values
loan_interest_clean = loan_interest_clean.fillna(0)
Xval = loan_interest_clean["Approved/Declined Amount"]
Yval = loan_interest_clean['Disbursed/Shipped Amount']
X, Y = np.meshgrid(Xval, Yval)
def get_Ratio(apr, dis):
    return apr/dis #return ratio of loan approval to disbursed

zs = np.array(get_Ratio(np.ravel(X),np.ravel(Y)))
Z = zs.reshape(X.shape)
fig, ax = plt.subplots(figsize = (10,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X,Y,Z, cmap = 'coolwarm', antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Approved/Declined Amount")
ax.set_ylabel("Disbursed/Shipped Amount")
ax.set_zlabel("Approved/Disbursed ratio")
ax.set_title("Approved/Declined Amount and their ratios")
ax.set_zlim(np.min(np.ravel(Z)),np.max(np.ravel(Z)))
plt.show()


#Bar plot
loan_subtype_df = pd.DataFrame(
    {
        'Total' : [data['Woman Owned Authorized Amount'].fillna(0).sum(), data['Small Business Authorized Amount'].fillna(0).sum(),data['Minority Owned Authorized Amount'].fillna(0).sum()]
    },
    index = ['Woman Owned Authorized Amount', 'Small Business Authorized Amount', 'Minority Owned Authorized Amount']
)

#drop outliers
#print(loan_subtype_df.head())
fig2, ax2 = plt.subplots(figsize = (10,6))
ax2.bar(loan_subtype_df.index, loan_subtype_df['Total'], label = loan_subtype_df.index)
plt.title('Authorized amount for each loan subtype')
plt.show()

#count plot
#count number of times each state appears
state_counts = data["Primary Exporter State Name"].value_counts().sort_values(ascending = False)
#Get the top five
top_5 = state_counts.head(5)
cntplot = px.histogram(top_5, x = top_5.index, y = 'count')
cntplot.update_layout(title=dict(text="Number of transactions logged by the top 5 states"))
cntplot.show()
#distplot for each loan type

data.astype({'Fiscal Year':'int64'})
#Last 3 years
last_fifteen_years = data[(data['Fiscal Year'].astype(int) <= 2025) & (data['Fiscal Year'].astype(int) >= 2015)]
loan_type_fifteen_years = pd.DataFrame(
    {
        'Woman Owned' : last_fifteen_years.loc[last_fifteen_years['Woman Owned Authorized Amount'] <= 20000000, 'Woman Owned Authorized Amount'],
        'Small Business' :last_fifteen_years.loc[ last_fifteen_years['Small Business Authorized Amount'] <=20000000, 'Small Business Authorized Amount'],
        'Minority Owned' : last_fifteen_years.loc[last_fifteen_years['Minority Owned Authorized Amount'] <20000000, 'Minority Owned Authorized Amount']
    }
)
#Histogram
plt.hist(loan_type_fifteen_years,label = loan_type_fifteen_years.columns,histtype = 'bar', bins = 20)
plt.legend(title = 'loan money subcategory')
plt.title('Histogram of Loan Type Owned Amount over the past fifteen years')
plt.show()

#Scatter plot
approved_vs_disbursed = data.loc[:, ['Approved/Declined Amount', 'Disbursed/Shipped Amount']].fillna(0)
#print(approved_vs_disbursed.head())

scatter1 = sns.lmplot(data = approved_vs_disbursed, x ='Approved/Declined Amount', y='Disbursed/Shipped Amount')
plt.title('Approved/Declined Amount vs Disbursed')
plt.show()
#Note, strong positive correlation, about 3/4. Trend is generally disbursed amt is just under approved amt

#Multivariate Box plot
total_loan_type = pd.DataFrame({'Amount' : data['Approved/Declined Amount'], 'State' : data['Primary Exporter State Name']})
#Outlier elimination: Get rid of any data point with a loan amount over like 10000000.
for index, row in total_loan_type.iterrows():
    if row['Amount'] >= 10000000:
        total_loan_type.drop(index, inplace = True)

fig4, ax4 = plt.subplots(figsize = (10,8), tight_layout = True)
plt.title('Box plot')
plt.xlabel('Loan Amount')
box1 = sns.boxplot(data = total_loan_type, x = 'Amount', y='State')
plt.show()

#Multivariate boxen plot
fig5, ax5 = plt.subplots(figsize = (10,8), tight_layout = True)
plt.title("Boxen plot")
sns.boxenplot(data = total_loan_type, x = 'Amount', y='State')
plt.xlabel('Loan Amount')
plt.show()

#Group by industry
#Area plot
industry_descriptions = ["Aircraft Manufacturing", "Drilling Oil and Gas Wells", "Engineering Services"]
df1 = data.loc[(data["Product Description"] == "Aircraft Manufacturing"), ["Fiscal Year", "Approved/Declined Amount"]].groupby("Fiscal Year").sum()
df2 = data.loc[(data["Product Description"] == "Drilling Oil and Gas Wells"), ["Fiscal Year", "Approved/Declined Amount"]].groupby("Fiscal Year").sum()
df3 = data.loc[(data["Product Description"] == "Engineering Services"), ["Fiscal Year", "Approved/Declined Amount"]].groupby("Fiscal Year").sum()
#Equalize data
area_plot = go.Figure()
area_plot.add_trace(go.Line(x = df1.index, y = df1["Approved/Declined Amount"], fill = 'tonexty', name = industry_descriptions[0]))
area_plot.add_trace(go.Line(x = df2.index, y = df2["Approved/Declined Amount"], fill = 'tonexty',name = industry_descriptions[1]))
area_plot.add_trace(go.Line(x = df3.index, y = df3["Approved/Declined Amount"], fill = 'tozeroy', name = industry_descriptions[2]))
area_plot.update_layout(title=dict(text="Area plot"), xaxis_title = "Fiscal Year", yaxis_title = "Approved Amount")
area_plot.show()
#Making a Violin plot

sns.violinplot(x = data.loc[(data['Woman Owned Authorized Amount'] <= 400000), 'Woman Owned Authorized Amount'])
plt.title("Woman Owned Authorized Amount Violin Plot")
plt.show()

#Do a histogram plot with KDE + rug plot
#the data: Small Business Loan in MD from 2016 to 2020, remove anything over 10000000
dmv_loans_sb = data[((data['Primary Exporter State Name'] == 'Maryland') | (data['Primary Exporter State Name'] == 'Virginia') | (data['Primary Exporter State Name'] == 'Delaware')) & (data['Small Business Authorized Amount'] <= 50000000)]
print(dmv_loans_sb.head())
#print("Primary Exporter State Names:")
hist1 = sns.histplot(data = dmv_loans_sb, x = 'Small Business Authorized Amount', hue = 'Term', kde = True, bins = 40)
rug1 = sns.rugplot(data = dmv_loans_sb, x = 'Small Business Authorized Amount', height = -0.02, clip_on=False, hue = 'Term')
plt.title("Small Business Loans in Delaware, Maryland, and Virginia - hist + rug plot")
plt.show()


#Dist plot
#Get transaction data from the last two years for the West Coast
west_coast_states = ["California", "Oregon", "Washington", "Alaska", 'Hawaii']

west_coast = data[(data['Primary Exporter State Name'] == "California") | (data['Primary Exporter State Name'] == "Oregon") | (data['Primary Exporter State Name'] == "Washington") | (data['Primary Exporter State Name'] == "Alaska")].reset_index()

west_last_2_yr = west_coast[(west_coast['Fiscal Year'] == '2025') | (west_coast['Fiscal Year'] == '2024')]
print("West coast state data:")
print(west_last_2_yr.shape)
g=sns.displot(data=west_last_2_yr, x = 'Woman Owned Authorized Amount', hue = 'Primary Exporter State Name', multiple = 'dodge')
plt.title("Dist plot showing loa ns in the West Coast")
plt.show()
# Q-Q plot

from seaborn_qqplot import pplot

pplot(west_last_2_yr, x='Woman Owned Authorized Amount', y = "Minority Owned Authorized Amount", kind = 'qq')
plt.title("Q-Q plot")
plt.show()


#first get state data values
state_values = west_last_2_yr['Primary Exporter State Name'].values.tolist()
applicant_vals = west_last_2_yr['Primary Applicant'].values.tolist()
for idx, row, in west_last_2_yr.iterrows():
    #Get the state number to encode
    state_name = row['Primary Exporter State Name']
    state_enc = state_values.index(state_name)
    app_enc = applicant_vals.index(row['Primary Applicant'])
    #replace
    west_last_2_yr.at[idx, 'Primary Exporter State Name'] = state_enc
    west_last_2_yr.at[idx, 'Primary Applicant'] = app_enc

data_state_vals = data['Primary Exporter State Name'].values.tolist()
for idx, row, in data.iterrows():
    #Get the state number to encode
    state_name = row['Primary Exporter State Name']
    state_enc = data_state_vals.index(state_name)
    #replace
    data.at[idx, 'Primary Exporter State Name Encoded'] = state_enc
#Making heatmap showing correlation between numeric values
num_data_df = data[['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount',
                    'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount', 'Primary Exporter State Name Encoded',]]

fig2 = plt.figure(figsize = (20,16))
fig = sns.heatmap(num_data_df.corr(), annot=True)
fig.set_title("Heatmap")
plt.show()
#print("new data")
#print(west_last_2_yr.head())

#Hexbin
#get quantiles

west_coast.dropna(subset = ['Approved/Declined Amount', 'Disbursed/Shipped Amount'], inplace = True)
#Convert to int
west_coast.astype({'Approved/Declined Amount' : 'int64', 'Disbursed/Shipped Amount' : 'int64'})
west_coast = west_coast[(west_coast['Approved/Declined Amount'] < 100000)| (west_coast['Disbursed/Shipped Amount'] < 100000)].reset_index()
#print(west_coast.head())
data_approved_q1 = np.quantile(data['Approved/Declined Amount'], 0.25)
data_approved_q3 = np.quantile(data['Approved/Declined Amount'], 0.75)
data_disbursed_q1 = np.quantile(data['Disbursed/Shipped Amount'], 0.25)
data_disbursed_q3 = np.quantile(data['Disbursed/Shipped Amount'], 0.75)
data_outlier_elim = data[(data['Approved/Declined Amount'] <= data_approved_q3)
                         & (data['Disbursed/Shipped Amount'] <= data_disbursed_q3)]

hexbin= plt.hexbin(data_outlier_elim['Disbursed/Shipped Amount'],
           data_outlier_elim['Approved/Declined Amount'], gridsize = 20, cmap = "coolwarm")
cbar = plt.colorbar(hexbin, label = 'counts')
plt.title("Hexbin")
plt.xlabel("Disbursed/Shipped Amount")
plt.ylabel("Approved/Declined Amount")
plt.show()

#Dataset description: a hex grid
#Cluster map
numeric_data = ['Disbursed/Shipped Amount'
                ,'Small Business Authorized Amount',
                "Woman Owned Authorized Amount", 'Minority Owned Authorized Amount', 'Approved/Declined Amount', "Primary Exporter State Name"]

num_df = west_coast[['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount',
                    'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount']]
print(num_df.head())
print("Graphing cluster map")
sns.clustermap(data = num_df.loc[0:100], figsize = (16,10))
plt.title("Cluster Map")
plt.show()

#Swarm plot
print("Graphing swarm plot")
#Get the loan amount for 4 years
last_4_yrs = pd.DataFrame(dmv_loans_sb[(dmv_loans_sb['Fiscal Year'].astype(int) >= 2020) & (dmv_loans_sb['Approved/Declined Amount'].astype(int) <= 10000000)])
sns.swarmplot(data = last_4_yrs, x='Approved/Declined Amount',y = 'Fiscal Year')
plt.title("Swarm Plot")
plt.xlabel("Approved/Declined Amount")
plt.ylabel("Fiscal Year")
plt.show()

#Joint plot
west_recent_clean = west_coast[(west_coast['Woman Owned Authorized Amount'] < 1000000) & (west_coast['Minority Owned Authorized Amount'] < 1000000)]
sns.jointplot(data = west_recent_clean, x = 'Woman Owned Authorized Amount', y = 'Minority Owned Authorized Amount', hue = 'Term')
plt.xlabel('Woman Owned Authorized Amount')
plt.ylabel('Minority Owned Authorized Amount')
plt.title("Joint plot")

plt.show()

#Pair plot
print(west_recent_clean.columns)
pair =sns.pairplot(west_recent_clean[["Woman Owned Authorized Amount", "Primary Exporter State Name", "Minority Owned Authorized Amount", "Small Business Authorized Amount", "Approved/Declined Amount"]], hue = 'Primary Exporter State Name')
plt.title("Pair plot")
plt.show()

#PCA Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import normaltest
from scipy import stats


scaler = StandardScaler()
X = scaler.fit_transform(num_data_df.iloc[:, :])
pca =PCA(n_components=6, svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)
#_,d,_ = np.linalg.svd(X)
#print("D_original = ", d)

#_,d_pca,_ = np.linalg.svd(X_PCA)
#print("D_PCA = ", d_pca)
print(f"condition number for original dataset: {np.linalg.cond(X)}")
print(f"Condition number for PCA dataset: {np.linalg.cond(X_PCA)}")

#normality test
norm_results = stats.kstest(num_data_df[['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount',
                    'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount']],stats.norm.cdf)
shapiro_test = stats.shapiro(num_data_df)
print("Shapiro: ", shapiro_test)
print(f"Approved amt quantile 1: {data_approved_q1}, quantile 3: {data_approved_q3}, median: {np.quantile(data['Approved/Declined Amount'], 0.50)}")
print(f"Disbursed quantile 1: {data_disbursed_q1}, third quantile: {data_disbursed_q3}, median: {np.quantile(data['Disbursed/Shipped Amount'], 0.5)}")
print(f"Norm test results: {norm_results}")
plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1),
         np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
plt.grid()
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()