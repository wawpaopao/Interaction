from __future__ import print_function
import sys
import copy
import logging
import itertools
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
import numpy as np
from fingerprint.graph import load_from_mol_tuple
from utils import get_lstm_embedding
import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, ikey2smiles=None,
                       ikey2mol=None,
                       berttokenizer=None,
                       uniprot2triplets=None,
                       prediction_mode=None,
                       protein_embedding_type=None,
                       datatype='train',
                       max_steps=1000,
                       batch=48,
                       shuffle=False):

        self.ikey2smiles=ikey2smiles
        self.ikey2mol=ikey2mol
        self.berttokenizer=berttokenizer
        self.uniprot2triplets=uniprot2triplets
        self.prediction_mode=prediction_mode
        self.prottype=protein_embedding_type
        self.datatype=datatype
        self.max_steps=max_steps
        self.batch=batch
        self.shuffle=shuffle
        logging.info("{} Evaluator for {} data initialized. Max {} steps for batch-size {}. \
        Shuffle {}".format(self.prediction_mode,datatype,max_steps,batch,shuffle))

    def eval(self,model,pairs,labels,epoch):
        def run_model(chem_repr, prot_repr, label, evaluate=True):
            chem_repr = load_from_mol_tuple(chem_repr)
            if isinstance(chem_repr, Variable) and torch.cuda.is_available():
                chem_repr = chem_repr.cuda()
            if isinstance(prot_repr, Variable) and torch.cuda.is_available():
                prot_repr = prot_repr.cuda()
            batch_input = {'protein': prot_repr, 'ligand': chem_repr}
            with torch.no_grad():
                logits = model(batch_input)
            batch_labels = torch.tensor(label)
            batch_labels = batch_labels.cpu().detach().numpy()
            logits = logits.cpu().detach().numpy()
            logging.debug("Evaluator-run_model: batch_labels {}, logits {}".format(batch_labels.shape, logits.shape))
            if evaluate:
                f1, auc, aupr = evaluate_binary_predictions(batch_labels, logits)
            else:
                return batch_labels, logits


        def get_repr_from_pairs_3(pairs):
            chem_repr = [(self.ikey2smiles[pair[0]], self.ikey2mol[pair[0]]) for pair in pairs]
            prot_repr = torch.stack(
                [torch.tensor(self.berttokenizer.encode(self.uniprot2triplets[pair[1]])) for pair in pairs])
            return (chem_repr, prot_repr)

        model.eval()

        datatype=self.datatype


        since=time.time()
        collected_logits = []
        collected_labels = []
        
        if len(pairs)<= self.batch:
            #sample size smaller than a batch
            chem_repr, prot_repr = get_repr_from_pairs(pairs)
            metrics = run_model(chem_repr,prot_repr,labels,evaluate=True)
            eval_time=time.time() - since
            logging.info("{:.2f} seconds for {} evaluation. Epoch {}".format(eval_time,datatype,epoch))
            print("{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(epoch,
                datatype,metrics[0],metrics[1],metrics[2]))
            return metrics

        elif len(pairs)<= self.max_steps*self.batch:
            #evaluate all pairs if small enough
            maxsteps=int(np.ceil(float(len(pairs))/float(self.batch)))
            for step in range(maxsteps):
                pairs_part=pairs[self.batch*step:self.batch*(step+1)]
                if len(pairs_part)==self.batch:
                    labels_part=labels[self.batch*step:self.batch*(step+1)]
                    chem_repr, prot_repr = get_repr_from_pairs_3(pairs_part)
                    # chem_repr,prot_repr = get_repr_from_pairs(pairs_part)
                    batch_labels,logits=run_model(chem_repr,prot_repr,labels_part,evaluate=False)
                    logging.debug("in-loop: batch_labels {}, logits {}".format(batch_labels.shape,logits.shape))
                    collected_logits.append(logits)
                    collected_labels.append(batch_labels)
            
        else:
            idxs = np.arange(len(pairs))
            np.random.shuffle(idxs)
            for step in range(self.max_steps):
                pairs_part=pairs[self.batch*step:self.batch*(step+1)]
                labels_part=labels[self.batch*step:self.batch*(step+1)]
                chem_repr, prot_repr = get_repr_from_pairs(pairs_part)
                batch_labels,logits=run_model(chem_repr,prot_repr,labels_part,evaluate=False)
                logging.debug("in-loop: batch_labels {}, logits {}".format(batch_labels.shape,logits.shape))
                collected_logits.append(logits)
                collected_labels.append(batch_labels)
                
        collected_labels = np.concatenate(collected_labels,axis=0)
        collected_logits = np.concatenate(collected_logits,axis=0)
        if self.prediction_mode.lower() in ['binary']:
            metrics = evaluate_binary_predictions(collected_labels,collected_logits)
        #else:
            #metrics = evaluate_continuous_predictions(collected_labels,collected_logits)

        #plot_roc_curve(collected_labels,collected_logits)

        eval_time=time.time() - since
        logging.info("{:.2f} seconds for {} evaluation. Epoch {}".format(eval_time,datatype,epoch))
        print("{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}".format(epoch,datatype,metrics[0],metrics[1],metrics[2]))
        return metrics

def plot_roc_curve(label, predprobs):

    probs = np.array(predprobs)
    probs = 1 / (1 + np.exp(-probs))
    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 0], pos_label=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('roc_curve')
    plt.plot(fpr, tpr, color='b', linewidth=0.8)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()


def evaluate_binary_predictions(label,predprobs):
    logging.debug("label {}, predprobs {}".format(label.shape,predprobs.shape))
    probs=np.array(predprobs)

    probs = 1/(1+np.exp(-probs))
    predclass = probs[:,0]>=0.5

    f1 = metrics.f1_score(label, predclass, average='weighted')

    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 0], pos_label=1)
    auc=metrics.auc(fpr, tpr)

    accu = metrics.accuracy_score(label, predclass)
    recall_score = metrics.recall_score(label,predclass)

    prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:, 0], pos_label=1)
    aupr=metrics.auc(reca,prec)
    return f1,auc,aupr,accu,recall_score



