# Real-Time-Machine-Learning-for-ICU-Patient-Outcomes-Predictions

## Motivation

Recent research on predicting Intensive Care Unit (ICU) patient outcomes has largely emphasized physiological time series data, often overlooking sparse inputs such as diagnoses and medications. When these additional features are used at all, they are typically appended at a late modeling stage—an approach that may fail to capture uncommon disease patterns effectively. In contrast, we propose leveraging diagnoses as relational cues by linking clinically similar patients in a graph. Specifically, we introduce LSTM-GNN, a hybrid method that combines Long Short-Term Memory networks (LSTMs) to model temporal dependencies and Graph Neural Networks (GNNs) to incorporate neighborhood information among patients. Our experiments on the eICU database show that LSTM-GNN outperforms an LSTM-only baseline for length of stay prediction. More broadly, our findings suggest that incorporating neighbor information through GNNs offers considerable promise for enhancing supervised learning performance in Electronic Health Record analyses.

## Pre-Processing Instructions

### eICU Pre-Processing

1. To run the sql files you must have the eICU database set up: https://physionet.org/content/eicu-crd/2.0/.

2. Follow the instructions: https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ to ensure the correct connection configuration.

3. Replace the eICU_path in `paths.json` to a convenient location in your computer, and do the same for `eICU_preprocessing/create_all_tables.sql` using find and replace for
   `'/Users/emmarocheteau/PycharmProjects/eICU-GNN-LSTM/eICU_data/'`. Leave the extra '/' at the end.

4. In your terminal, navigate to the project directory, then type the following commands:

   ```
   psql 'dbname=eicu user=eicu options=--search_path=eicu'
   ```

   Inside the psql console:

   ```
   \i eICU_preprocessing/create_all_tables.sql
   ```

   This step might take a couple of hours.

   To quit the psql console:

   ```
   \q
   ```

5. Then run the pre-processing scripts in your terminal. This will need to run overnight:

   ```
   python -m eICU_preprocessing.run_all_preprocessing
   ```

### Graph Construction

To make the graphs, you can use the following scripts:

This is to make most of the graphs that we use. You can alter the arguments given to this script.

```
python -m graph_construction.create_graph --freq_adjust --penalise_non_shared --k 3 --mode k_closest
```

Write the diagnosis strings into `eICU_data` folder:

```
python -m graph_construction.get_diagnosis_strings
```

Get the bert embeddings:

```
python -m graph_construction.bert
```

Create the graph from the bert embeddings:

```
python -m graph_construction.create_bert_graph --k 3 --mode k_closest
```



## Training the ML Models

Before proceeding to training the ML models, do the following.

1. Define data_dir, graph_dir, log_path and ray_dir in `paths.json` to convenient locations.

2. Run the following to unpack the processed eICU data into mmap files for easy loading during training. The mmap files will be saved in `data_dir`.

   ```
   python3 -m src.dataloader.convert
   ```

The following commands train and evaluate the models introduced in our paper.

N.B.

* The models are structured using pytorch-lightning. Graph neural networks and neighbourhood sampling are implemented using pytorch-geometric.

* Our models assume a default graph which is made with k=3 under a k-closest scheme. If you wish to use other graphs, refer to `read_graph_edge_list` in `src/dataloader/pyg_reader.py` to add a reference handle to `version2filename` for your graph.

* The default task is **In-House-Mortality Prediction (ihm)**, add `--task los` to the command to perform the **Length-of-Stay Prediction (los)** task instead.

* These commands use the best set of hyperparameters; To use other hyperparameters, remove `--read_best` from the command and refer to `src/args.py`.

### a. LSTM-GNN

The following runs the training and evaluation for LSTM-GNN models. `--gnn_name` can be set as `gat`, `sage`, or `mpnn`. When `mpnn` is used, add `--ns_sizes 10` to the command.

```
python -m train_ns_lstmgnn --bilstm --ts_mask --add_flat --class_weights --gnn_name gat --add_diag --read_best
```

The following runs a hyperparameter search.

```
python -m src.hyperparameters.lstmgnn_search --bilstm --ts_mask --add_flat --class_weights  --gnn_name gat --add_diag
```

### b. Dynamic LSTM-GNN

The following runs the training & evaluation for dynamic LSTM-GNN models. `--gnn_name` can be set as `gcn`, `gat`, or `mpnn`.

```
python -m train_dynamic --bilstm --random_g --ts_mask --add_flat --class_weights --gnn_name mpnn --read_best
```

The following runs a hyperparameter search.

```
python -m src.hyperparameters.dynamic_lstmgnn_search --bilstm --random_g --ts_mask --add_flat --class_weights --gnn_name mpnn
```

### c. GNN

The following runs the GNN models (with neighbourhood sampling). `--gnn_name` can be set as `gat`, `sage`, or `mpnn`. When `mpnn` is used, add `--ns_sizes 10` to the command.

```
python3 -m train_ns_gnn --ts_mask --add_flat --class_weights --gnn_name gat --add_diag --read_best
```

The following runs a hyperparameter search.

```
python -m src.hyperparameters.ns_gnn_search --ts_mask --add_flat --class_weights --gnn_name gat --add_diag
```

### d. LSTM (Baselines)

The following runs the baseline bi-LSTMs. To remove diagnoses from the input vector, remove `--add_diag` from the command.

```
python -m train_ns_lstm --bilstm --ts_mask --add_flat --class_weights --num_workers 0 --add_diag --read_best
```

The following runs a hyperparameter search.

```
python -m src.hyperparameters.lstm_search --bilstm --ts_mask --add_flat --class_weights --num_workers 0 --add_diag
```

