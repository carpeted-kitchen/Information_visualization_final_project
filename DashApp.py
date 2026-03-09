from dash import Dash, html, dcc, callback, Output, Input, no_update
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import normaltest

my_app = Dash("Term Project")
server = my_app.server

data = pd.read_csv("Data.Gov+-+FY25+Q1.csv",dtype = dict({"Primary Export Product NAICS/SIC code" : np.str_, "Multiyear Working Capital Extension": np.str_, "Fiscal Year": np.str_}))
#clean data a little
data.dropna(inplace=True, subset='Fiscal Year')
data = data.astype({'Fiscal Year': 'int32'})


# print(data.astype({'Fiscal Year': 'int32'}).dtypes)
# total number of loans per country
valcounts = data['Country'].value_counts().dropna()
new_pd = data
cntry = pd.DataFrame(data["Country"].value_counts())
# Track number of "Other Countries" instances
other_country_count = 0

cntry_clean = cntry.dropna()
cntry_clean.drop("Multiple - Countries", inplace=True)

other_country_list1 = []
for i in cntry_clean.index:
    count = cntry_clean.loc[i, "count"]
    # print(f"{i}: {count}")
    if count <= 310:
        other_country_count += count
        cntry_clean.drop(i, inplace=True)
        other_country_list1.append(i)
# Add "Other Countries" row
cntry_clean.at["Other Countries", 'count'] = other_country_count

# Loan amount per country
cntry_loan_sum = data.loc[:, ['Country', 'Approved/Declined Amount']].groupby('Country').sum()
cntry_loan_sum_clean = cntry_loan_sum.dropna(subset=['Approved/Declined Amount'])
cntry_loan_sum_clean.fillna(0)
cntry_loan_sum_clean.drop("Multiple - Countries", inplace=True)
# print(cntry_loan_sum_clean.head())
max_loan_amt = cntry_loan_sum_clean['Approved/Declined Amount'].max()
other_cntry_loan_total = 0
other_countries = []
loan_cntry_avg = cntry_loan_sum_clean['Approved/Declined Amount'].median()
for i in cntry_loan_sum_clean.index:
    loan_total = cntry_loan_sum_clean.loc[i, "Approved/Declined Amount"]
    if loan_total < (max_loan_amt * 0.10):
        other_countries.append(i)
        cntry_loan_sum_clean.drop(i, inplace=True)
        other_cntry_loan_total += loan_total

cntry_loan_sum_clean.at["Other countries", 'Approved/Declined Amount'] = other_cntry_loan_total

# Small business loan amount per country
sb_loan_df = data.loc[:, ['Country', 'Small Business Authorized Amount']].groupby('Country').sum()
sub_loan_df = sb_loan_df.dropna()
sb_loan_df.drop("Multiple - Countries", inplace=True)

# print("Small business loan")
# print(sb_loan_df.head(6))
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
# print(f"Other countries small business : {sb_other_cntry_loan_total}, {len(other_countries_sb)}")

# Get number of small business loans that each country received
nonzero_sb_loan = data[data['Small Business Authorized Amount'] > 0]
sb_cntry = pd.DataFrame(nonzero_sb_loan['Country'].value_counts())
sb_cntry = sb_cntry.dropna()
# print(sb_cntry.head(8))
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
PieChartDict = dict({
    "Loans Per Country": cntry_clean,
    "Total Loans Received": cntry_loan_sum_clean,
    "Small Business Loan Amt.": sb_loan_df,
    ' Small Business Loan Num.': sb_cntry
})

data_no_outliers = data[
    (data['Woman Owned Authorized Amount'] < 10000000) & (data['Small Business Authorized Amount'] < 10000000)
    & (data['Minority Owned Authorized Amount'] < 10000000)]
loan_subtype_df = pd.DataFrame(
    {
        'Total': [data_no_outliers['Woman Owned Authorized Amount'].fillna(0).sum(),
                  data_no_outliers['Small Business Authorized Amount'].fillna(0).sum(),
                  data_no_outliers['Minority Owned Authorized Amount'].fillna(0).sum()]
    },
    index=['Woman Owned Authorized Amount', 'Small Business Authorized Amount', 'Minority Owned Authorized Amount']
)

last_fifteen_years = data_no_outliers[data_no_outliers['Fiscal Year'] >= 2010]

# Get total loan amount approves for every fiscal year
loan_approved_fisc = data.loc[:, ["Fiscal Year", 'Approved/Declined Amount']].groupby(['Fiscal Year']).sum().dropna()
# print(loan_approved_fisc.head())
year_range = np.arange(2007, 2025, 1)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
# surface plot

loan_interest_clean = data.loc[(data['Fiscal Year'].astype(int) >= 2023)
                               & (data['Approved/Declined Amount'] < 5000000),
["Approved/Declined Amount", "Disbursed/Shipped Amount", "Loan Interest Rate"]]
# iterate thru rows and clean
# print(f"shape: {loan_interest_clean.shape}")
# print(loan_interest_clean.head())
# fill in NaN values
loan_interest_clean = loan_interest_clean.fillna(0)
Xval = np.array(loan_interest_clean["Approved/Declined Amount"])
Yval = np.array(loan_interest_clean['Disbursed/Shipped Amount'])


def get_Ratio(apr, dis):
    z = []
    for i in range(len(apr)):
        d = dis[i]
        # print(f"d={d}")
        a = apr[i]
        if d != 0:
            z.append(a / d)
        else:
            z.append(0)
    return z


zs = np.array(get_Ratio(Xval, Yval))  # Z is currently a 1-D list
# print(len(zs))
xi = np.linspace(Xval.min(), Xval.max(), 300)
yi = np.linspace(Yval.min(), Yval.max(), 300)
X, Y = np.meshgrid(xi, yi)

Z = griddata((Xval, Yval), zs, (X, Y), method='cubic')

print("Creating surface plot")
surf = go.Figure(go.Surface(x=xi, y=yi, z=Z))
surf.update_layout(title=dict(text='Ratio between Approved and Disbursed Amount for loans'),
                   xaxis=dict(title="Approved Amount"), yaxis=dict(title="Disbursed Amount"))

# surf.show()
# Bar chart
fig1 = px.bar(loan_subtype_df, x=loan_subtype_df.index, y='Total')

# count plot
# count number of times each state appears
state_counts = data["Primary Exporter State Name"].value_counts().sort_values(ascending=False)
# Get the top three
top_5 = state_counts.head(5)
cntplot = px.histogram(top_5, x=top_5.index, y='count')

# Histogram
data.astype({'Fiscal Year': 'int64'})
# Last 3 years
last_fifteen_years = data[(data['Fiscal Year'].astype(int) <= 2025) & (data['Fiscal Year'].astype(int) >= 2015)]
loan_type_fifteen_years = pd.DataFrame(
    {
        'Woman Owned': last_fifteen_years['Woman Owned Authorized Amount'],
        'Small Business': last_fifteen_years['Small Business Authorized Amount'],
        'Minority Owned': last_fifteen_years['Minority Owned Authorized Amount']
    }
)
# print(loan_type_fifteen_years.head())

# Scatter plot
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots

approved_vs_disbursed = data.loc[:, ['Approved/Declined Amount', 'Disbursed/Shipped Amount']].fillna(0)
# go.Scatter(approved_vs_disbursed, x = 'Approved/Declined Amount', y='Disbursed/Shipped Amount')
X = approved_vs_disbursed['Approved/Declined Amount']
Y = approved_vs_disbursed['Disbursed/Shipped Amount']
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Scatter(x=approved_vs_disbursed['Approved/Declined Amount'],
                          y=approved_vs_disbursed['Disbursed/Shipped Amount'], mode="markers",
                          marker=dict(color="blue", opacity=0.8), name="Approved vs. Disbursed"),
               secondary_y=False)
reg = LinearRegression()
# Fit to x and y
res = reg.fit(np.array(X).reshape(-1, 1), np.array(Y))
reg_pred = reg.predict(np.array(X).reshape(-1, 1))
fig3.add_trace(
    go.Scatter(x=X, y=reg_pred, name='Trendline', mode='lines', marker_color="lightblue"),
    secondary_y=True
)
# add trendline
fig3.update_layout(xaxis=dict(title="Approved/Declined Amount"),
                   yaxis=dict(title="Disbursed/Shipped Amount")
                   )
fig3.update_traces(hoverinfo="none", hovertemplate=None)
# fig3.show()
# Box Plot
total_loan_type = pd.DataFrame(
    {'Amount': data['Approved/Declined Amount'], 'State': data['Primary Exporter State Name']})
# Outlier elimination: Get rid of any data point with a loan amount over like 10000000.
for index, row in total_loan_type.iterrows():
    if row['Amount'] >= 10000000:
        total_loan_type.drop(index, inplace=True)

fig4 = px.box(total_loan_type, x="Amount", y="State")
fig4.update_layout(xaxis=dict(title="Loan Amount"), yaxis=dict(title="State"),
                   title=dict(text="Loan amount for each state"))
# Violin
fig5 = px.violin(x=data.loc[(data['Woman Owned Authorized Amount'] <= 400000), 'Woman Owned Authorized Amount'])
fig5.update_layout(xaxis=dict(title="Fiscal Year"), yaxis=dict(title="Woman Owned Authorized Amount"),
                   title=dict(text="Woman Owned Authorized Amount loans for 2007 and 2008"))
# fig5.show()
# Do a histogram plot with KDE
# the data: Small Business Loan in MD from 2016 to 2020, move anything over 10000000
dmv_loans_sb = data[((data['Primary Exporter State Name'] == 'Maryland')
                     | (data['Primary Exporter State Name'] == 'Virginia')
                     | (data['Primary Exporter State Name'] == 'Delaware')) & (
                                data['Small Business Authorized Amount'] <= 50000000)]

dmvdf = [dmv_loans_sb.loc[dmv_loans_sb["Term"] == "Short Term", "Small Business Authorized Amount"],
         dmv_loans_sb.loc[dmv_loans_sb["Term"] == "Medium Term", "Small Business Authorized Amount"],
         dmv_loans_sb.loc[dmv_loans_sb["Term"] == "Long Term", "Small Business Authorized Amount"]]
terms = ["Short", "Medium", "Long"]
print("About to do KDE")
fig6 = ff.create_distplot(dmvdf, terms, colors=['red', 'orange', 'green'])
fig6.update_layout(title="Histogram with KDE", xaxis=dict(title="Small business loan amount"),
                   yaxis=dict(title="count"))
# fig6.show()
# print(dmv_loans_sb.head())

# heatmap
num_data_df = data[['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount',
                    'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount']]

print("Built correlation matrix")
fig7 = px.imshow(num_data_df.loc[0:50, :].corr(), text_auto=True)
fig7.update_layout(title=dict(text='Heatmap'), xaxis_tickangle=-90)
# fig7.show()

# Hexbin
data_approved_q1 = np.quantile(data['Approved/Declined Amount'], 0.25)
data_approved_q3 = np.quantile(data['Approved/Declined Amount'], 0.75)
print(f"Approved amt quantile 1: {data_approved_q1}, quantile 3: {data_approved_q3}")
data_disbursed_q1 = np.quantile(data['Disbursed/Shipped Amount'], 0.25)
data_disbursed_q3 = np.quantile(data['Disbursed/Shipped Amount'], 0.75)

data_outlier_elim = data[(data['Approved/Declined Amount'] <= data_approved_q3)
                         & (data['Disbursed/Shipped Amount'] <= data_disbursed_q3)]
# Creating this hexbin was a pain, since the normal ff hexbin function doesn't work with this dataset
# Found a workaround by converting a matplotlib hexbin to something that can be read by
# Figure Factory

plt.axis('off')
HB = plt.hexbin(data_outlier_elim['Disbursed/Shipped Amount'],
                data_outlier_elim['Approved/Declined Amount'], gridsize=25, cmap=cm.coolwarm)

cbar = plt.colorbar(HB, label='counts')
plt.title("Hexbin")
plt.xlabel("Disbursed/Shipped Amount")
plt.ylabel("Approved/Declined Amount")


def get_hexbin_attributes(hexbin):
    paths = hexbin.get_paths()
    points_codes = list(paths[0].iter_segments())  # path[0].iter_segments() is a generator
    prototypical_hexagon = [item[0] for item in points_codes]
    return prototypical_hexagon, hexbin.get_offsets(), hexbin.get_facecolors(), hexbin.get_array()


def pl_cell_color(mpl_facecolors):
    return [f'rgb({int(R * 255)}, {int(G * 255)}, {int(B * 255)})' for (R, G, B, A) in mpl_facecolors]


def make_hexagon(prototypical_hex, offset, fillcolor, linecolor=None):
    new_hex_vertices = [vertex + offset for vertex in prototypical_hex]
    vertices = np.asarray(new_hex_vertices[:-1])
    # hexagon center
    center = np.mean(vertices, axis=0)
    if linecolor is None:
        linecolor = fillcolor
    # define the SVG-type path:
    path = 'M '
    for vert in new_hex_vertices:
        path += f'{vert[0]}, {vert[1]} L'
    return dict(type='path',
                line=dict(color=linecolor,
                          width=0.5),
                path=path[:-2],
                fillcolor=fillcolor,
                ), center


hexagon_vertices, offsets, mpl_facecolors, counts = get_hexbin_attributes(HB)
l1 = len(offsets)
# I had to hard-code this so I didn't have to instal MAtplotLib 3.5.3 and Python 3.10
cell_color = ['rgb(246, 189, 164)', 'rgb(179, 3, 38)', 'rgb(232, 213, 202)', 'rgb(221, 220, 219)', 'rgb(246, 183, 156)',
              'rgb(131, 166, 251)', 'rgb(86, 115, 224)', 'rgb(65, 86, 201)', 'rgb(90, 120, 227)', 'rgb(71, 95, 208)',
              'rgb(111, 145, 242)', 'rgb(63, 83, 198)', 'rgb(72, 96, 209)', 'rgb(133, 168, 251)', 'rgb(59, 77, 193)',
              'rgb(66, 88, 202)', 'rgb(115, 149, 244)', 'rgb(97, 130, 234)', 'rgb(70, 93, 207)', 'rgb(99, 131, 234)',
              'rgb(63, 83, 198)',
              'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(59, 77, 193)', 'rgb(65, 86, 201)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(67, 90, 204)',
              'rgb(58, 76, 192)', 'rgb(63, 83, 198)', 'rgb(86, 115, 224)', 'rgb(92, 123, 229)', 'rgb(66, 88, 202)',
              'rgb(99, 131, 234)', 'rgb(63, 83, 198)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(64, 84, 199)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(65, 86, 201)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)',
              'rgb(70, 93, 207)', 'rgb(80, 107, 218)', 'rgb(65, 86, 201)', 'rgb(87, 117, 225)', 'rgb(62, 81, 196)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(59, 77, 193)',
              'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(64, 84, 199)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(66, 88, 202)', 'rgb(69, 91, 205)', 'rgb(78, 105, 216)', 'rgb(78, 105, 216)',
              'rgb(62, 81, 196)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)',
              'rgb(64, 84, 199)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(64, 84, 199)',
              'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(63, 83, 198)', 'rgb(65, 86, 201)', 'rgb(63, 83, 198)',
              'rgb(99, 131, 234)', 'rgb(63, 83, 198)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(64, 84, 199)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(62, 81, 196)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(65, 86, 201)', 'rgb(62, 81, 196)',
              'rgb(76, 102, 214)', 'rgb(75, 100, 212)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(59, 77, 193)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(63, 83, 198)', 'rgb(62, 81, 196)',
              'rgb(70, 93, 207)', 'rgb(62, 81, 196)',
              'rgb(111, 145, 242)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(65, 86, 201)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(67, 90, 204)', 'rgb(59, 77, 193)',
              'rgb(66, 88, 202)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(64, 84, 199)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(60, 79, 195)', 'rgb(70, 93, 207)', 'rgb(60, 79, 195)',
              'rgb(60, 79, 195)', 'rgb(73, 98, 211)', 'rgb(62, 81, 196)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(69, 91, 205)', 'rgb(59, 77, 193)',
              'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(65, 86, 201)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(69, 91, 205)', 'rgb(59, 77, 193)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(87, 117, 225)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(66, 88, 202)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(67, 90, 204)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(64, 84, 199)', 'rgb(60, 79, 195)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(72, 96, 209)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(64, 84, 199)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)', 'rgb(59, 77, 193)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)',
              'rgb(90, 120, 227)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)', 'rgb(59, 77, 193)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(139, 174, 253)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(60, 79, 195)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(59, 77, 193)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(70, 93, 207)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(62, 81, 196)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(59, 77, 193)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(62, 81, 196)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(62, 81, 196)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(245, 160, 129)', 'rgb(171, 199, 252)', 'rgb(244, 157, 126)',
              'rgb(128, 164, 250)', 'rgb(99, 131, 234)', 'rgb(71, 95, 208)', 'rgb(215, 84, 68)', 'rgb(59, 77, 193)',
              'rgb(63, 83, 198)', 'rgb(60, 79, 195)', 'rgb(70, 93, 207)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)',
              'rgb(59, 77, 193)', 'rgb(86, 115, 224)', 'rgb(109, 144, 241)', 'rgb(104, 137, 238)', 'rgb(85, 113, 222)',
              'rgb(65, 86, 201)', 'rgb(69, 91, 205)', 'rgb(104, 137, 238)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(69, 91, 205)', 'rgb(80, 107, 218)', 'rgb(95, 126, 231)', 'rgb(76, 102, 214)', 'rgb(62, 81, 196)',
              'rgb(69, 91, 205)', 'rgb(90, 120, 227)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(63, 83, 198)',
              'rgb(71, 95, 208)', 'rgb(92, 123, 229)', 'rgb(73, 98, 211)', 'rgb(60, 79, 195)', 'rgb(63, 83, 198)',
              'rgb(87, 117, 225)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(66, 88, 202)', 'rgb(77, 103, 215)',
              'rgb(92, 123, 229)', 'rgb(62, 81, 196)', 'rgb(64, 84, 199)', 'rgb(85, 113, 222)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(66, 88, 202)', 'rgb(75, 100, 212)', 'rgb(70, 93, 207)',
              'rgb(70, 93, 207)', 'rgb(65, 86, 201)', 'rgb(78, 105, 216)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(69, 91, 205)', 'rgb(67, 90, 204)', 'rgb(63, 83, 198)',
              'rgb(70, 93, 207)', 'rgb(81, 108, 219)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(67, 90, 204)', 'rgb(65, 86, 201)', 'rgb(60, 79, 195)', 'rgb(65, 86, 201)',
              'rgb(80, 107, 218)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(65, 86, 201)', 'rgb(64, 84, 199)', 'rgb(60, 79, 195)', 'rgb(62, 81, 196)', 'rgb(86, 115, 224)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(64, 84, 199)',
              'rgb(63, 83, 198)', 'rgb(59, 77, 193)', 'rgb(62, 81, 196)', 'rgb(75, 100, 212)', 'rgb(65, 86, 201)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(63, 83, 198)',
              'rgb(59, 77, 193)', 'rgb(63, 83, 198)', 'rgb(71, 95, 208)', 'rgb(58, 76, 192)', 'rgb(70, 93, 207)',
              'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(63, 83, 198)', 'rgb(59, 77, 193)',
              'rgb(64, 84, 199)', 'rgb(70, 93, 207)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(70, 93, 207)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)',
              'rgb(70, 93, 207)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(62, 81, 196)', 'rgb(60, 79, 195)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)',
              'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(70, 93, 207)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(80, 107, 218)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(62, 81, 196)',
              'rgb(59, 77, 193)', 'rgb(60, 79, 195)', 'rgb(70, 93, 207)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(76, 102, 214)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(70, 93, 207)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(64, 84, 199)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(59, 77, 193)', 'rgb(59, 77, 193)', 'rgb(60, 79, 195)',
              'rgb(71, 95, 208)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(65, 86, 201)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(69, 91, 205)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(67, 90, 204)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(65, 86, 201)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(66, 88, 202)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)',
              'rgb(60, 79, 195)', 'rgb(66, 88, 202)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(63, 83, 198)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(64, 84, 199)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(59, 77, 193)', 'rgb(65, 86, 201)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)', 'rgb(58, 76, 192)', 'rgb(60, 79, 195)', 'rgb(58, 76, 192)', 'rgb(58, 76, 192)',
              'rgb(58, 76, 192)']

l2 = len(cell_color)

shapes = []
print(HB.get_facecolors())
centers = []
for k in range(len(offsets)):
    shape, center = make_hexagon(hexagon_vertices, offsets[k], cell_color[k])
    shapes.append(shape)
    centers.append(center)


def mpl_to_plotly(cmap, N):
    h = 1.0 / (N - 1)
    pl_colorscale = []
    for k in range(N):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([round(k * h, 2), f'rgb({C[0]}, {C[1]}, {C[2]})'])
    return pl_colorscale


pl_coolwarm = mpl_to_plotly(cm.coolwarm, 11)
X, Y = zip(*centers)

# define  text to be  displayed on hovering the mouse over the cells
text = [f'x: {round(X[k], 2)}<br>y: {round(Y[k], 2)}<br>counts: {int(counts[k])}' for k in range(len(X))]
trace = go.Scatter(
    x=list(X),
    y=list(Y),
    mode='markers',
    marker=dict(size=0.5,
                color=counts,
                colorscale=pl_coolwarm,
                showscale=True,
                colorbar=dict(
                    thickness=20,
                    ticklen=4
                )),
    text=text,
    hoverinfo='text'
)
axis = dict(showgrid=False,
            showline=False,
            zeroline=False,
            ticklen=4
            )

layout = go.Layout(title='Hexbin showing Approved vs. Disbursed Amounts',
                   width=530, height=550,
                   xaxis=axis,
                   yaxis=axis,
                   hovermode='closest',
                   shapes=shapes,
                   plot_bgcolor='black')

hexbin = go.Figure(data=[trace], layout=layout)

# Dist plot
west_coast_states = ["California", "Oregon", "Washington", "Alaska"]
west_coast = data[
    (data['Primary Exporter State Name'] == "California") | (data['Primary Exporter State Name'] == "Oregon") | (
                data['Primary Exporter State Name'] == "Washington") | (
                data['Primary Exporter State Name'] == "Alaska")].reset_index()

west_last_2_yr = west_coast[(west_coast['Fiscal Year'] == 2025) | (west_coast['Fiscal Year'] == 2024)]
# print("west last 2 yr")
# print(west_last_2_yr[0:10])
recent = data[(data['Fiscal Year'] <= 2025)
              & (data['Fiscal Year'] >= 2023)]
cali = recent.loc[recent['Primary Exporter State Name'] == 'California', 'Woman Owned Authorized Amount']
recent_or = recent.loc[recent['Primary Exporter State Name'] == 'Oregon', 'Woman Owned Authorized Amount']
recent_wa = recent.loc[recent['Primary Exporter State Name'] == 'Washington', 'Woman Owned Authorized Amount']

west_coast_recent_wo = [
    recent.loc[recent['Primary Exporter State Name'] == 'California', 'Woman Owned Authorized Amount'],
    recent.loc[recent['Primary Exporter State Name'] == 'Oregon', 'Woman Owned Authorized Amount'],
    recent.loc[recent['Primary Exporter State Name'] == 'Washington', 'Woman Owned Authorized Amount']
]

fig8 = px.histogram(west_last_2_yr, x='Woman Owned Authorized Amount', color='Primary Exporter State Name')
fig8.update_layout(title=dict(text="Distribution plot"), xaxis=dict(title="Woman-Owned Amount"),
                   xaxis_range=[0, 3000000])

# fig8.show()
# Q-Q plot
# Woman-Owned vs. Minority Owned


# qq plot
wc_wo = west_last_2_yr["Woman Owned Authorized Amount"]
# print(wc_wo[0:10])
qqplot_data_wo = stats.probplot(wc_wo)
wc_mo = west_last_2_yr["Minority Owned Authorized Amount"]
qqplot_data_mo = stats.probplot(wc_mo)

# compute quantiles
quantiles = np.arange(0, 1, 0.05)
# quantile_mo =

X_lognorm = np.random.lognormal(mean=0.0, sigma=1.7, size=500)

qq = stats.probplot(X_lognorm, dist='lognorm', sparams=(1))
x = np.array([qq[0][0][0], qq[0][0][-1]])

fig = go.Figure()
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
fig.add_scatter(x=x, y=qq[1][1] + qq[1][0] * x, mode='lines')
fig.layout.update(showlegend=False)
# fig.show()

us_state_list = data["Primary Exporter State Name"].values

# clustergram
# import dash_bio as dashbio

num_df = west_coast.loc[0:50,
         ['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount',
          'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount']]
data_array = num_df.to_numpy()
data_array = data_array.transpose()

labels = num_df.columns
# Initialize figure by creating upper dendrogram
# Initialize figure by creating upper dendrogram
cluster = ff.create_dendrogram(data_array, orientation='bottom', labels=labels)
for i in range(len(cluster['data'])):
    cluster['data'][i]['yaxis'] = 'y2'

# Create Side Dendrogram
dendro_side = ff.create_dendrogram(data_array, orientation='right')
for i in range(len(dendro_side['data'])):
    dendro_side['data'][i]['xaxis'] = 'x2'

# Add Side Dendrogram Data to Figure
for dat in dendro_side['data']:
    cluster.add_trace(dat)
# Create Heatmap
dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
dendro_leaves = list(map(int, dendro_leaves))
data_dist = pdist(data_array)
heat_data = squareform(data_dist)
heat_data = heat_data[dendro_leaves, :]
heat_data = heat_data[:, dendro_leaves]

heatmap = [go.Heatmap(x=dendro_leaves, y=dendro_leaves, z=heat_data,
                      colorscale='Blues'
                      )
           ]
heatmap[0]['x'] = cluster['layout']['xaxis']['tickvals']
heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

# Add Heatmap Data to Figure
for dat in heatmap:
    cluster.add_trace(dat)

# Edit Layout
cluster.update_layout({'width': 800, 'height': 800,
                       'showlegend': False, 'hovermode': 'closest',
                       })
# Edit xaxis
cluster.update_layout(xaxis={'domain': [.15, 1],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'ticks': ""})
# Edit xaxis2
cluster.update_layout(xaxis2={'domain': [0, .15],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

# Edit yaxis
cluster.update_layout(yaxis={'domain': [0, .85],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': False,
                             'ticks': ""
                             })
# Edit yaxis2
cluster.update_layout(yaxis2={'domain': [.825, .975],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

# Plot!
# cluster.show()
# swarm plot
last_4_yrs = pd.DataFrame(dmv_loans_sb[(dmv_loans_sb['Fiscal Year'].astype(int) >= 2020) & (
            dmv_loans_sb['Approved/Declined Amount'].astype(int) <= 10000000)])
swarm = px.strip(last_4_yrs, x='Approved/Declined Amount', y='Fiscal Year')
swarm.update_layout(title=dict(text="Swarm plot: Approved Amt. vs. Fiscal Year"), xaxis=dict(title="Approved Amount"),
                    yaxis=dict(title="Fiscal Year"))

# joint plot
west_recent_clean = west_coast[(west_coast['Woman Owned Authorized Amount'] < 1000000) & (
            west_coast['Minority Owned Authorized Amount'] < 1000000)]
jplot = px.scatter(west_recent_clean, x='Woman Owned Authorized Amount', y='Minority Owned Authorized Amount',
                   color="Term", marginal_x="rug", marginal_y="histogram", title="Joint plot")
jplot.update_layout(title=dict(text="Joint plot"))

# scatter matrix
labels_of_interest = ["Woman Owned Authorized Amount", "Primary Exporter State Name",
                      "Minority Owned Authorized Amount", "Small Business Authorized Amount",
                      "Approved/Declined Amount"]
pairplot = px.scatter_matrix(west_recent_clean, dimensions=labels_of_interest
                             , color="Primary Exporter State Name")

data_state_vals = data['Primary Exporter State Name'].values.tolist()
for idx, row, in data.iterrows():
    # Get the state number to encode
    state_name = row['Primary Exporter State Name']
    state_enc = data_state_vals.index(state_name)
    # replace
    data.at[idx, 'Primary Exporter State Name Encoded'] = state_enc
num_data_df2 = data[['Approved/Declined Amount', 'Disbursed/Shipped Amount', 'Small Business Authorized Amount',
                     'Woman Owned Authorized Amount', 'Minority Owned Authorized Amount',
                     'Primary Exporter State Name Encoded']]
# PCA time:
scaler = StandardScaler()
X = scaler.fit_transform(num_data_df2.iloc[:, :])
pca = PCA(n_components=6, svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)
condx = np.linalg.cond(X)
condxpca = np.linalg.cond(X_PCA)
pca_fig = px.line(x=np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1),
                  y=np.cumsum(pca.explained_variance_ratio_))
pca_fig.update_layout(title="PCA for numerical dataset", xaxis_title="Number of features",
                      yaxis_title="Explained variance ratio")

num_df_cols = list(num_df.columns.values)
num_df_rows = list(num_df.index)
my_app.layout = html.Div([
    html.Hr(),
    dcc.Tabs(id="app-tabs",
             children=[
                 dcc.Tab(label="tab 1", value='tab1'),
                 dcc.Tab(label="tab 2", value='tab2'),
                 dcc.Tab(label="tab 3", value='tab3'),
                 dcc.Tab(label="tab 4", value='tab4'),
                 dcc.Tab(label="tab 5", value='tab5'),
                 dcc.Tab(label="tab 6", value='tab6'),
             ]
             ),

    html.Div(id='layout')
])

tab1_layout = html.Div([
    html.H1("First Tab"),
    html.H5('line graph and pie chart'),
    dcc.RadioItems(
        options=['Loans Per Country', 'Total Loans Received', 'Small Business Loan Amt.', 'Small Business Loan Num.'],
        value='Loans Per Country', id='controls-and-radio-item'),
    dcc.Graph(figure={}, id='pie_charts'),
    dcc.Graph(figure={}, id='hist'),
    html.P("Number of bins"),
    dcc.Slider(500, 1000, 10, value=750, id='hist-slider'),

    dcc.Dropdown(options=[
        {'label': 'Maryland', 'value': 'Maryland'},
        {'label': 'Virginia', 'value': 'Virginia'},
        {'label': 'Oregon', 'value': 'Oregon'},
    ], value='Maryland', id='state-name'),

    dcc.Graph(figure={}, id='line-graph'),
    dcc.Graph(figure={}, id='area-plot'),
    dcc.RangeSlider(2007, 2025,
                    marks={year: str(year) for year in range(2007, 2026)},
                    value=[2007, 2025], id='range-slider'),

    html.Br(),
    dcc.Checklist(options=top_5.index, id='countplot-checklist', value=top_5.index),
    dcc.Graph(figure={}, id='count-plot'),
])


@my_app.callback(
    Output('area-plot', 'figure'),
    Input('range-slider', 'value'),
)
def update_area_plot(value):
    to = int(value[0])
    frm = int(value[1])
    print(value)
    # Group by industry
    # Area plot
    industry_descriptions = ["Aircraft Manufacturing", "Drilling Oil and Gas Wells", "Engineering Services"]
    df1 = data.loc[
        data["Product Description"] == "Aircraft Manufacturing", ["Fiscal Year", "Approved/Declined Amount"]].groupby(
        "Fiscal Year").sum()
    df2 = data.loc[data["Product Description"] == "Drilling Oil and Gas Wells", ["Fiscal Year",
                                                                                 "Approved/Declined Amount"]].groupby(
        "Fiscal Year").sum()
    df3 = data.loc[
        data["Product Description"] == "Engineering Services", ["Fiscal Year", "Approved/Declined Amount"]].groupby(
        "Fiscal Year").sum()

    # industry_df = data.loc[(data["Product Description"] == "Aircraft Manufacturing") & (data["Product Description"] == "Drilling Oil and Gas Wells")
    # & ( data["Product Description"] == "Engineering Services"), ["Fiscal Year", "Approved/Declined Amount", "Product Description"]
    # ]
    # Equalize data
    d1 = df1[(df1.index >= to) & (df1.index <= frm)]

    # print(d1.head())
    area_plot = go.Figure()
    area_plot.add_trace(
        go.Scatter(x=df1.index, y=df1["Approved/Declined Amount"], fill='tonexty', name=industry_descriptions[0]))
    area_plot.add_trace(
        go.Scatter(x=df2.index, y=df2["Approved/Declined Amount"], fill='tonexty', name=industry_descriptions[1]))
    area_plot.add_trace(
        go.Scatter(x=df3.index, y=df3["Approved/Declined Amount"], fill='tozeroy', name=industry_descriptions[2]))
    area_plot.update_layout(title=dict(text="Area plot"), xaxis_title="Fiscal Year", yaxis_title="Approved Amount")
    area_plot.update_xaxes(range=[to, frm])
    # Come back to this
    return area_plot


@my_app.callback(
    Output("hist", 'figure'),
    Input("hist-slider", 'value')
)
def update_histogram(value):
    print(f"bins: {value}")
    fig2 = px.histogram(loan_type_fifteen_years, title="Histogram of Loan subtypes",
                        nbins=value)  # maybe add a slider here
    fig2.update_layout(xaxis=dict(title="Loan amount"))
    return fig2


# Add controls for radio button
@my_app.callback(
    Output(component_id='pie_charts', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_pie_chart(chart_chosen):
    target_df = PieChartDict[chart_chosen]
    target_col = target_df.columns[0]
    # print(target_col)
    fig = px.pie(target_df, values=target_col, names=target_df.index, title=chart_chosen)
    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig


# pick US state for line plot
@my_app.callback(
    Output(component_id='line-graph', component_property='figure'),
    Input(component_id='state-name', component_property='value'),

)
def update_line(state_name):
    # print(f"Type of data: {type(data)}")
    tgt_df = data.loc[
        data["Primary Exporter State Name"] == state_name, ["Fiscal Year", 'Approved/Declined Amount']].groupby(
        ['Fiscal Year']).sum()
    tgt_df.dropna()
    # create the line plot
    newlineplot = px.line(tgt_df, x=tgt_df.index, y='Approved/Declined Amount')
    return newlineplot


@my_app.callback(
    Output('count-plot', component_property='figure'),
    Input('countplot-checklist', component_property='value'),
)
def update_countplot(options):
    target_data = top_5.loc[options]
    # print("Target data count plot")
    # print(target_data.head())
    newcnt = px.histogram(x=target_data.index, y=target_data)
    newcnt.update_layout(title="Count plot")
    return newcnt


tab2_layout = html.Div([
    html.H1("Second Tab"),
    html.H4("Bar Chart showing different loan subtypes"),
    dcc.Checklist(options=["Woman Owned Authorized Amount", "Small Business Authorized Amount",
                           "Minority Owned Authorized Amount"],
                  id='bar-chart-checklist', value=["Woman Owned Authorized Amount", "Small Business Authorized Amount",
                                                   "Minority Owned Authorized Amount"]),
    dcc.Graph(figure={}, id='bar-chart'),
    html.Hr(),
    html.Div(className="container",
             children=[
                 dcc.Graph(figure=fig3, id='scatter-plot', clear_on_unhover=True),  # Note to self: Add Tooltips

                 dcc.Tooltip(id='scatter-tooltip', loading_text="Loading..."),
             ]),
    dcc.Graph(figure=swarm, id='swarm-plot'),
])


# bar chart callback
@my_app.callback(
    Output(component_id='bar-chart', component_property='figure'),
    Input(component_id='bar-chart-checklist', component_property='value')
)
def update_bar_chart(options_chosen):
    # Get the chosen columns
    data_chosen = loan_subtype_df.loc[options_chosen]
    print(options_chosen)
    newfig = px.bar(data_chosen, x=options_chosen, y="Total")
    return newfig


@my_app.callback(
    Output("scatter-tooltip", "show"),
    Output("scatter-tooltip", "bbox"),
    Output("scatter-tooltip", "children"),
    Input("scatter-plot", "hoverData")
)
def scatter_display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    # print(f"point: {hoverData["points"]}")
    # display x and y coordinate
    bbox = pt["bbox"]
    # print(f"bbox: {bbox}")
    children = [
        html.P(f"x: {pt['x']}, y: {pt['y']}")
    ]
    return True, bbox, children


tab3_layout = html.Div([
    dcc.Graph(figure=fig7, id='heatmap'),
    dcc.Dropdown(options=['Minority Owned Authorized Amount', 'Woman Owned Authorized Amount',
                          'Small Business Authorized Amount'], value="Woman Owned Authorized Amount",
                 id='violin-dropdown'),
    dcc.Graph(figure={}, id='violin-plot'),
    html.H4("PCA Analysis"),
    dcc.Graph(figure=pca_fig, id='pca-plot'),
])


@my_app.callback(
    Output('violin-plot', 'figure'),
    Input('violin-dropdown', 'value')

)
def update_violin(value):
    # print("Violin plot value: ", value)
    fig5 = px.violin(x=data.loc[(data[value] <= 500000), value])
    fig5.update_layout(xaxis=dict(title="Fiscal Year"), yaxis=dict(title=value),
                       title=dict(text="Violin plot for loan subtypes"))
    return fig5


tab4_layout = html.Div([
    html.H2("QQ plot"),
    dcc.Graph(figure=fig, id='qq-plot'),
    dcc.Tooltip(id='qq-tooltip', loading_text="Loading..."),

    dcc.Graph(figure=surf, id='surface plot'),
    dcc.Dropdown(
        id='clustergram-input', options=[{'label': row, 'value': row} for row in list(num_df.index)
                                         ], value=num_df_rows[:10], multi=True
    ),
    html.Div(id='clustergram'),
    dcc.Graph(figure=cluster, id='clustergram'),

    dcc.Graph(figure=pairplot, id='pair-plot'),
]
)


@my_app.callback(
    Output("qq-tooltip", "show"),
    Output("qq-tooltip", "bbox"),
    Output("qq-tooltip", "children"),
    Input("qq-plot", "hoverData")
)
def show_qq_tooltip(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    children = [
        html.P(f"X : {pt['x']}, Y: {pt['y']}")
    ]
    return True, bbox, children


tab5_layout = html.Div([
    html.H2("Tab 5"),
    dcc.Graph(figure=hexbin, id='hexbin-plot'),
    dcc.Tooltip(id='hexbin-tooltip'),
    html.H4("Box"),
    dcc.Graph(figure=fig4, id='box-plot'),
])


@my_app.callback(
    Output("hexbin-tooltip", "show"),
    Output("hexbin-tooltip", "bbox"),
    Output("hexbin-tooltip", "children"),
    Input("hexbin-plot", "hoverData")
)
def show_hex_tooltip(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    children = [
        html.P(f"Dis : {pt['x']}, Apr: {pt['y']}")
    ]
    return True, bbox, children


tab6_layout = html.Div([
    dcc.Graph(figure=jplot, id='joint plot'),
    dcc.Graph(figure=fig6, id='hist-plot-kde'),
])


@my_app.callback(
    Output(component_id='layout', component_property='children'),
    [Input(component_id='app-tabs', component_property='value')]
)
def update_layout(tab):
    if tab == 'tab1':
        return tab1_layout
    elif tab == 'tab2':
        return tab2_layout
    elif tab == 'tab3':
        return tab3_layout
    elif tab == 'tab4':
        return tab4_layout
    elif tab == 'tab5':
        return tab5_layout
    elif tab == 'tab6':
        return tab6_layout
#run app
if __name__ == '__main__':
    my_app.run(debug=True)