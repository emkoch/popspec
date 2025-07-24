import pandas as pd
import numpy as np
import requests
import argparse
import pickle
from tqdm import tqdm
import os
import sys
sys.path.append("/home/sjg319/monosemanticity/bio_models_sae/utils")

from toga_utils import *
from msa_helpers import *

# ----- Parse SLURM array task ID -----
parser = argparse.ArgumentParser()
parser.add_argument("--chunk_id", type=int, default=None)
parser.add_argument("--num_chunks", type=int, default=20)
args = parser.parse_args()

if args.chunk_id is None:
    args.chunk_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

chunk_id = args.chunk_id
num_chunks = args.num_chunks

data_dir = '/n/data2/hms/dbmi/sunyaev/lab/sgandhi/effect_size/variant datasets/Beltran'
ddG_df = pd.read_csv(f'{data_dir}/SupplementaryTable4.txt', sep='\t')
VEP_df = pd.read_csv(f'{data_dir}/SupplementaryTable5.txt', sep='\t')

with open("uniprot_pos.pkl", "rb") as f:
    uniprot_pos = pickle.load(f)

chunked = np.array_split(uniprot_pos, num_chunks)
my_chunk = chunked[chunk_id]

print('Loaded files')
############################################### Required functions #################################################

def uniprotToGene(uniprot_id):
    # UniProt REST API URL (UniProtKB API returns JSON)
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # The gene name is usually under 'genes'
        if 'genes' in data:
            gene_names = []
            for gene in data['genes']:
                if 'geneName' in gene and 'value' in gene['geneName']:
                    return gene['geneName']['value']
        else:
            print(f"No gene names found for {uniprot_id}")
            return None
    else:
        print(f"Error: {response.status_code} - {response.reason}")
        return None

def returnENSP(gene):
    url = f'https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}?expand=1'
    headers = {'Content-type': 'application/json'}
    
    response = requests.get(url, headers=headers)
    
    if response.ok:
        data = response.json()
    else:
        print(f"Error: {response.status_code} - {response.reason}")
        return None

    try:
        ensp = data['Transcript'][0]['Translation']['id']
        return ensp
    except:
        return None

def enspToCoordinates(ensp, pos):
    url = f'https://rest.ensembl.org/map/translation/{ensp}/{pos}..{pos+1}?'
    headers = {'Content-type': 'application/json'}
    
    response = requests.get(url, headers=headers)
    
    if response.ok:
        data = response.json()
        chrom = data['mappings'][0]['seq_region_name']
        start = data['mappings'][0]['start']
        end = data['mappings'][0]['end']
    
        return chrom, start, end
    else:
        print(f"Error: {response.status_code} - {response.reason}")
        return None
    

def mapCDS(gene):
    gene_id = get_ensembl_gene_id(gene)
    cds = get_longest_cds(gene_id)
    ens_id = cds[0]
 
    server = "https://rest.ensembl.org"
    ext = f"/map/cds/{ens_id}/1..{len(cds[1])}?"
     
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
     
    if not r.ok:
      return None
     
    decoded = r.json()

    # Sort the contigs based on start
    contigs = pd.DataFrame(0, index=[], columns=['start', 'end', 'strand'])
    for i, contig in enumerate(decoded['mappings']):
        strand = 0
        if contig['strand'] != 0:
            contigs.loc[len(contigs), :] = [contig['start'], contig['end'], contig['strand']]
            strand = contig['strand']
    contigs = contigs.sort_values(by='start')
    
    # Next, we go through and give each nucleotide its correct position
    mapped_positions = []
    for i in range(len(contigs)):
        mapped_positions.extend(np.arange(contigs.loc[i, 'start'], contigs.loc[i, 'end'] + 1))
    
    mapped_positions = [int(x) for x in mapped_positions]
    return mapped_positions, strand


def get_ensembl_gene_id(gene_name):
    # Define the Ensembl REST API URL for gene search
    base_url = "https://rest.ensembl.org"
    endpoint = f"/lookup/symbol/human/{gene_name}?content-type=application/json"
    
    # Send the request to the Ensembl API
    response = requests.get(base_url + endpoint)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response to get the Ensembl Gene ID (ENSG)
        data = response.json()
        
        # Return the ENSG gene ID if found
        ensembl_gene_id = data.get('id', None)
        if ensembl_gene_id:
            return ensembl_gene_id
        else:
            return "Gene not found"
    else:
        return f"Error: {response.status_code}"

############################################# Main loop block ##########################################################
genomic_coords = pd.DataFrame('', index=np.arange(len(my_chunk)), columns=['chr', 'start', 'end', 'gene', 'ref', 'seq', 'strand'])

gene_data = {}
chr_data = {}
uniprot_to_gene = {}
gene_to_ens = {}
skipped = []

for i in tqdm(np.arange(len(my_chunk))):
    uniprot_id, pos = my_chunk[i]
    pos = int(pos)
    try:
        subset = ddG_df[(ddG_df['uniprot_ID'] == uniprot_id) & (ddG_df['pos'] == pos)]
        ref_aa = subset['wt_aa'].iloc[0]

        # Map uniprot -> gene
        if uniprot_id not in uniprot_to_gene:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            response = requests.get(url)
            data = response.json()
            gene = data["genes"][0]["geneName"]["value"]
            uniprot_to_gene[uniprot_id] = gene
        else:
            gene = uniprot_to_gene[uniprot_id]

        # Map gene -> transcript
        if gene not in gene_to_ens:
            url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}?expand=1"
            headers = {"Content-Type": "application/json"}
            response = requests.get(url, headers=headers)
            data = response.json()
            ens_id = data['Transcript'][0]['id']
            gene_to_ens[gene] = ens_id
        else:
            ens_id = gene_to_ens[gene]

        # Get CDS mapping
        if gene not in gene_data:
            mapped_pos, strand = mapCDS(gene)
            mapped_pos = np.sort(mapped_pos)
            gene_data[gene] = (mapped_pos, strand)
        else:
            mapped_pos, strand = gene_data[gene]
            mapped_pos = np.sort(mapped_pos)

        if strand == -1:
            mapped_pos = mapped_pos[::-1]
            dna_pos = (pos-1)*3
            end = mapped_pos[dna_pos]
            start = mapped_pos[dna_pos+2]
        else:
            dna_pos = (pos-1)*3
            start = mapped_pos[dna_pos]
            end = mapped_pos[dna_pos+2]

        # Get chromosome
        if gene in chr_data:
            chrom = chr_data[gene]
        else:
            url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}?content-type=application/json"
            response = requests.get(url)
            data = response.json()
            chrom = data['seq_region_name']
            chr_data[gene] = chrom

        # Get DNA sequence
        url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:{strand}"
        headers = {"Content-Type": "text/plain"}
        response = requests.get(url, headers=headers)
        if response.ok:
            seq = response.text.strip()
        else:
            seq = ''

        genomic_coords.iloc[i, :] = [chrom, start, end, gene, ref_aa, seq, strand]

    except Exception as e:
        skipped.append(i)


################################################################################
# Save output
genomic_coords.to_csv(f"/n/data2/hms/dbmi/sunyaev/lab/sgandhi/effect_size/variant datasets/Beltran/processed_pairs/chunk_{chunk_id}.csv", index=False)
