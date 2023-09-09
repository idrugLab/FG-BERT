import rdkit
from importlib.resources import contents
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdescriptors
import csv
import pandas as pd


def lipinski_wt_limit(m):
    return Descriptors.MolWt(m) <= 500

def lipinski_logp_limit(m):
    return Descriptors.MolLogP(m) <= 5

# def lipinski_hba_limit(m):
#     return rdescriptors.CalcNumLipinskiHBA(m) <= 10

def lipinski_hbd_limit(m):
    return rdescriptors.CalcNumLipinskiHBD(m) <= 5

# def NumRotatableBonds(m):
#     return (rdkit.Chem.rdMolDescriptors.CalcNumRotatableBonds(m)) <= 10

def lipinski_violations(m):
    return 3 - sum((lipinski_wt_limit(m),
                    lipinski_logp_limit(m),
                    lipinski_hbd_limit(m)))

def my_select(smis:list)->list:
    res=[]
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            if lipinski_violations(mol)==0:
                res.append(smi)
                with open("chembl_select_3.txt","a+") as f:
                    f.write(smi)
    return res

lines=open('chembl.txt','r').readlines()
res = my_select(lines)
