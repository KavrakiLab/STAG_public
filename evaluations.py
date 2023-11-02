import sys

from models.STAG.cross_validation import cross_validation as cv_STAG
from models.CNN_3D.cross_validation import cross_validation as cv_CNN_3D
from models.Contacts.cross_validation import cross_validation as cv_Contacts

tables = {'GLCTLVAML':'datasets/GLCTLVAML/GLCTLVAML.csv','IVTDFSVIK':'datasets/IVTDFSVIK/IVTDFSVIK.csv','NLVPMVATV':'datasets/NLVPMVATV/NLVPMVATV.csv','ELAGIGILTV':'datasets/ELAGIGILTV/ELAGIGILTV.csv','RAKFKQLL':'datasets/RAKFKQLL/RAKFKQLL.csv','GILGFVFTL':'datasets/GILGFVFTL/GILGFVFTL.csv','AVFDRKSDAK':'datasets/AVFDRKSDAK/AVFDRKSDAK.csv','KLGGALQAK':'datasets/KLGGALQAK/KLGGALQAK.csv','pan_peptide':'datasets/pan_peptide/pan_peptide.csv'}
dirs = {'GLCTLVAML':'datasets/GLCTLVAML/structures','IVTDFSVIK':'datasets/IVTDFSVIK/structures','NLVPMVATV':'datasets/NLVPMVATV/structures','ELAGIGILTV':'datasets/ELAGIGILTV/structures','RAKFKQLL':'datasets/RAKFKQLL/structures','GILGFVFTL':'datasets/GILGFVFTL/structures','AVFDRKSDAK':'datasets/AVFDRKSDAK/structures','KLGGALQAK':'datasets/KLGGALQAK/structures','pan_peptide':'datasets/pan_peptide/structures'}

if __name__ == '__main__':
    method = str(sys.argv[1])
    dataset = str(sys.argv[2])

    if method == "STAG":
        cv_STAG(tables[dataset],dirs[dataset])
    elif method == "CNN_3D":
         cv_CNN_3D(tables[dataset],dirs[dataset])
    elif method == "Contacts":
        cv_Contacts(tables[dataset],dirs[dataset])
    else:
        raise Exception("Supported methods are 'STAG' 'CNN_3D' and 'Contacts'.")

    if dataset not in tables.keys():
        print('Supported Datasets: ',list(tables.keys()))
        print(dataset+' not supported')
        raise Exception("Please choose a valid dataset")
