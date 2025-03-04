# Comparison and Analysis of Value Linking in Text-to-SQL Systems

**Experiment, Analysis & Benchmark**

---

## Overview

This repository contains the code and resources associated with the paper **"Comparison and Analysis of Value Linking in Text-to-SQL Systems"**. Here, you will find everything needed to reproduce the experiments, analysis, and benchmarks presented in the paper.

---

## Environment Setup

### 1. Create a Conda Environment

Create and activate a new Conda environment with Python 3.10:

    conda create -n value_linking python=3.10 -y
    conda activate value_linking

### 2. Install JDK and Maven Dependencies

Install the required JDK and Maven dependencies using Conda:

    conda install -c conda-forge openjdk=21 maven -y

### 3. Install Python Dependencies

Install the necessary Python packages by running:

    pip install -r requirements.txt

---

## Dataset Preparation

Before running the experiments, prepare the dataset for value linking by executing:

    python create_value_linking_dataset.py

---

## Running the Experiments

This repository includes several scripts to run the experiments described in the paper. Use the following commands to execute each experiment:

- **Baseline Experiments**  
      
      python run_baselines.py
  
- **Indexes Experiments**  
      
      python run_indexes_experiments.py
  
- **Value Reference Detection Experiments**  
      
      python run_value_reference_experiments.py
  
- **Filtering Experiments**  
      
      python run_filtering_experiments.py

---

## Additional Information

- Each script is annotated with comments explaining its function and how it contributes to the overall analysis.
- For detailed explanations of the experiments, please refer to the paper and the inline comments within the code.

---
