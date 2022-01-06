# Interaction
1. Install Prerequisites
python 3.7
Pytorch
rdkit
Transformers (Huggingface. version 2.3.0)
2. Clone this repository
3. Download Data
All data could be download here and put it under this repository, i.e. in the same directory as the finetuning_train.py.

There will be four subdirectories in the data folder.

image

activity: gives you the train/dev/test set split based on protein similarity at threshold of bitscore 0.035
albertdata: gives you pretrained ALBERT model. The ALBERT is pretraind on distilled triplets of whole Pfam
Integrated: gives collected chemicals from several database
protein: gives you mapping from uniprot ID to triplets form
4. Generate clusters:
1. Cluster your protein dataset with `cdhit.sh`. Input is fasta file with all protein sequences in your dataset.
2. Apply multi-sequence alignment to the clusters with Clustal Omega. (`clustalo.sh`)
3. Build hmm profiles for the clusters with hmmbuild. (`hmmer_build.sh`)
4. Redo multi-sequence alignment with the hmm profiles and HMP clusters with HMMER. (`hmmer_align.sh`)
5. Construct corpus (singlets and triplets, represent sequence and all sequences) with `construct_hmp_singlets_and_triplets.py`. This step could take long if use only one CPU. Multiprocessing can significantly reduce computing time.
6. Generate TFRecord with the corpus with `create_tfrecords.sh`.
5. Run Finetuning
To run ALBERT model (default: ALBERRT frozen transformer):

python finetuning_train.py --protein_embedding_type="albert"
To try other freezing options, change "frozen_list" to choose modules to be frozen.

To run LSTM model:

python finetuning_train.py --protein_embedding_type="lstm"
