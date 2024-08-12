import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import warnings
import re
import sys

site = sys.argv[1]

# First, get all the files in the desired directory
mypath='./sim_outputs/underdominant/single'
gz_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Next, load in the betas
filename = 'betas.txt'
beta_vals = np.loadtxt(filename, delimiter='\t', dtype=float)
beta_vals = beta_vals.tolist()

# Define the function that builds the dfs
def buildDF(site_1, gz_files, site_2=None):
    if site_2:
        pattern = re.compile(f'sim_outs_single_{site_1}_{site_2}_[0-9]+\.txt\.gz')
    else:
        pattern = re.compile(f'underdom_{site_1}_[0-9]+\.txt\.gz')
    specified_files = []
    for file in gz_files:    
        if bool(pattern.match(file)):
            specified_files.append(file)

    # Build the dataframe
    df_list = [pd.read_csv(mypath + f'/{file}', sep='\t', compression='gzip') for file in specified_files]
    sim_df = pd.concat(df_list)

    return sim_df

# Now go through and build the dfs
sim_df = buildDF(site, gz_files)
sim_df.to_csv(f'/n/scratch/users/s/sjg319/dataframes/underdominant/sim_{site}_{site}.csv', sep='\t')
print(f'built dataframe for {site}')
