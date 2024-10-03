import numpy as np
import pandas as pd

df = pd.read_csv('liberty_data_2.csv')
df2 = df.drop(df.columns[[0,1,2,3,174,175,176,177,178,179,180]], axis=1)
df2.fillna(0,inplace=True)
df2
df2['conversions'] = np.where(df2['conversions']=='N',0,1)

def desc_func(d,target_col,drop_col):
    des_df = pd.DataFrame(d.describe(include='all',percentiles=[0.25,0.50,0.70,0.75,0.90]).T)
    des_df['var'] = des_df['std']**2
    des_df.reset_index(inplace=True)
    des_df.rename(columns = {'index':'feature'},inplace=True)
    des_df
    l_ones = []
    l_zeros = []
    for i in d.columns:
        l_ones.append(pd.crosstab(d[i],d[target_col])[1][1])
        l_zeros.append(pd.crosstab(d[i],d[target_col])[0][1])

    ones = [i/d[target_col].value_counts()[1] for i in l_ones]
    zeros = [i/d[target_col].value_counts()[0] for i in l_zeros]

    des_df['count_of_ones'] = l_ones
    des_df['count_of_zeros'] = l_zeros
    des_df['conversion_perc'] = ones
    des_df['non-conversion_perc'] = zeros
    des_df2 = des_df.sort_values(by='conversion_perc',ascending=False)
    sig_flag = []
    for i in des_df2['conversion_perc']:
        if i>0.05:
            sig_flag.append('Y')
        else:
            sig_flag.append('N')
    des_df2['Significant'] = sig_flag
    des_df2['Significant'].value_counts()
    des_df2.to_csv('desc_func.csv')
    return des_df2

desc_func(df2,'conversions')
df