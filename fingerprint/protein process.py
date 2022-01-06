from utils import load_json
uniprot2triplets = load_json('D:/deeplearning/MSA-transformer examples/data/albertdata/gpcr_uniprot2triplets.json')
i=1
for uni in uniprot2triplets.keys():
    if i>100:
        break
    else:
        triplets = uniprot2triplets[uni].strip().split(' ')
        triplets.pop(0)
        triplets.pop(-1)
        uniprot2triplets[uni] = ' '.join(triplets)
        i+=1
        print(uni)

       #m = list(set(uniprot2triplets[uni]))
