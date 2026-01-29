# GC-TSG

This repository provides the implementation of **GC-TSG**, a graph-based framework for traffic sign grouping and clustering.

---

## Environment Requirements

- **Python** >= 3.10  
- **Recommended**: latest versions of **PyTorch** and **PyTorch Geometric (PyG)**

### Dependencies

- torch  
- torch-geometric  
- numpy  
- scipy  
- scikit-learn  
- tqdm  
- path  
- rtree  

---

## Dataset

### AAL

- Source: GoMapClustering  
  https://github.com/fromm1990/GomapClustering
- Download the AAL dataset from the link above and place it in the corresponding data directory expected by the code.

### CPH

- The CPH dataset is provided in this repository.
- Location: `./data_cph/`

---

## Running GC-TSG

### Entry Script

Run the main pipeline with:

```bash
python main_scc.py
