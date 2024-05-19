# QComp
A robust, interpretable, non-iterative data completion framework for sparse datasets in drug discovery.

## Dependencies:
pytorch>=2.0.1

deepchem>=2.7.1

numpy

scipy

pandas

sklearn

tensorboard

matplotlib

## Run
(1) Download and decompress the dataset from https://zenodo.org/doi/10.5281/zenodo.11215174

(2) Execute ''python main.py''

## Public ADMET datasets and QSAR results
The ADMET data compiled from various public sources and the corresponding Chemprop multitask model predictions are located under `public_data_results`. The dataset is randomly split to 80 % training and 20 % test sets using 5-folds. Details on the files under `public_data_results`:
- `all_data`: `public_admet_data_all.csv` contains all data for 25 different ADMET assays along with SMILES strings and molecular weights of compounds. `data_count_name_unit_info.csv` contains detailed information and unit of each ADMET assay. The dataset is very sparse. `data_overlap_count_between_prop.csv` shows the number of compound overlaps between each pair of assays. `spearman_corr_heatmap.pdf` shows the Spearman correlation heatmap generated for the assay pair that has at least 10 overlapping compounds.
- `random_split_data_results`: contains training and test sets for each fold. In each fold, `chemprop_multitask_pred` folder contains the predictions from Chemprop multitask model (e.g. `public_admet_data_random_fold_0_test_set_model_pred.csv`) and the ensemble variance of the model predictions (e.g. `public_admet_data_random_fold_0_test_set_model_ensemble_variance.csv`).
- `result_figs`: `pred_comparison_RF_Chemprop_single_multitask.pdf` shows the comparison among Random Forest (RF), Chemprop single-task, and Chemprop multi-task models. The RF model uses the Morgan Fingerprints and MOE2D descriptors. The errors are evaluated over 5-fold cross validation on the random split, and the error bars represent the standard deviations among the 5-folds.

The original source of the individual data point will be added soon for the public dataset.
