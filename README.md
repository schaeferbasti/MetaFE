# MetaFE: Guide Feature Engineering with Meta-Learning

MetaFE is a meta-learning framework for automated feature engineering. Instead of exhaustively
searching or applying fixed heuristics, MetaFE trains a model on dataset and feature
**metafeatures** to predict which feature-engineering transformations are likely to improve a
downstream model's performance. The predicted best transformation is then applied recursively,
allowing the feature set to be improved step by step without retraining the downstream model on
every candidate transformation.

MetaFE is evaluated against [OpenFE](https://github.com/IIIS-Li-Group/OpenFE), an established
automated feature engineering method, on 44 datasets sourced from [OpenML](https://www.openml.org/).

## Repository Structure

```
MetaFE/
├── knowledge_base/
│   ├── Core_Matrix_Complete.parquet/         # Model- and Improvement-only Knowledge Base
│   ├── Pandas_Matrix_Complete/               # Complete Knowledge Base (Core_Matrix_Complete.parquet with meta-features)
├── results/
│   ├── MetaFE_146818.parquet                 # Exemplary result of MetaFE
├── utils/
│   ├── ...                                   # All utils files
├── Add_Pandas_Metafeatures.py                # Meta-feature extractor
├── run_metafe.py                             # Entry point for MetaFE, executing this file runs MetaFE on the OpenML task ids listed within the given ressource limits
├── Surrogate_Model.py                        # Code of the MetaFE model that predicts the improvements of transformations
└── README.md
```
                              |

## Installation

```bash
git clone https://github.com/schaeferbasti/MetaFE.git
cd MetaFE
conda create -n metafe python=3.10
conda activate metafe
pip install -r requirements.txt
```


## Usage

Run MetaFE:

```bash
python run_metafe.py                          # Adapt datasets and ressource limits if required
```
