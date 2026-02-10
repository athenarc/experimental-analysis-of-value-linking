# Comparison and Analysis of Value Linking in Text-to-SQL Systems
**[Experiments & Analysis]**

This repository contains the code for the paper **_"Comparison and Analysis of Value Linking in Text-to-SQL systems [Experiments & Analysis]"_**.  
Here, you will find everything needed to **reproduce the experiments, analysis, and benchmarks** presented in the paper.

---

## 🧩 Reproducing the Experiments

### 1. Clone the repository and set up the environment

```bash
wget https://anonymous.4open.science/api/repo/experimental-analysis-of-value-linking-36B3/zip -O experimental-analysis-of-value-linking.zip
unzip experimental-analysis-of-value-linking.zip -d experimental-analysis-of-value-linking
cd experimental-analysis-of-value-linking/
```

### 2. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate value_linking
pip install -e .
```

### 3. Download required datasets and precomputed indexes

```bash
hf download ValueLinking/value_linking_assets --repo-type dataset --local-dir ./
```

⚠️ **Note:** The HuggingFace account used to upload this data was created solely for the purpose of this paper and is completely anonymized. This command downloads the databases and all precomputed indexes required for running the experiments. The download may take some time. If you prefer, you can instead download only the database and compute the indexes manually using the provided scripts.

---

## 📊 Benchmark Creation

To create our benchmark:

**1. Data Exploration**

We used `utils/data_explorer.py` to perform data exploration that generated LLM-based discrepancies.
   - Raw discrepancies: `assets/data_exploration`
   - After manual curation: `assets/data_exploration_human`

**2. Benchmark Generation**

Using `scripts/run_alter_execute_verify.py`, we created the final benchmark:
   - Category-specific benchmarks: `assets/all_benchmarks_human`
   - Final deduplicated benchmark: `assets/all_benchmarks_human/all_dump_good.json`

---

## ⚙️ Running the Experiments

### 1. Indexing

To evaluate the Value Linking component, first run the indexing:

```bash
python scripts/run_indexing.py
```

⏱ This may take some time, but you can skip it if you have downloaded the full assets from HuggingFace.

### 2. Retrieval (RQ1)

- Fill in your `wandb_entity` inside `scripts/run_retrieval.py` to log results to Weights & Biases:

```bash
python scripts/run_retrieval.py
```

- For retrieval at k:

```bash
python scripts/run_retrieval_k.py
```

### 3. Text-to-SQL Experiments (RQ2)

**Standard Systems:**
Preprocessing and running Text-to-SQL experiments for systems like OpenSearch-SQL, OmniSQL, and CHESS follow the original paper implementations. Replace the dev file with our dev files (`assets/all_benchmarks_human/all_dump_good.json`) as needed.

**LLM Baselines:**
To reproduce the results for frontier LLMs (Claude Sonnet 4.5, GPT-5.2) which lack explicit value linking modules:

*   **Claude Sonnet 4.5 (via AWS Bedrock):**
    Update `scripts/run_bedrock.py` with your `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `REGION_NAME`.
    ```bash
    python scripts/run_bedrock.py --model_id "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    ```

*   **GPT-5.2 (via OpenRouter):**
    Update `scripts/run_open_router.py` with your `OPENROUTER_API_KEY`.
    ```bash
    python scripts/run_open_router.py
    ```

### 4. Performance with Ideal Value Links and Noise (RQ3)

We modified the original systems to exclude value linking, and included their code in this repository. Simply run the original system scripts, replacing the dev file each time with:

```bash
assets/augmented_dump_p*.json
```

where `*` corresponds to different precision values. All necessary preprocessing files are included in `assets`.

### 5. Performance Impact of Detection and Mapping Methods (RQ4)

To evaluate the trade-offs between different combinations of detection (e.g., NER, LLM, n-grams) and mapping methods (e.g., Dense Index, MinHashLSH):

```bash
python scripts/run_value_reference_detection.py
```

### 6. Generalization to Unseen Schemas (OpenAIRE)

To validate findings on the independent OpenAIRE Research Graph hold-out set:

1.  **Indexing:**
    Update `scripts/run_openaire_indexing_retrievers.py` with your PostgreSQL connection details (`DB_CONFIG`).
    ```bash
    python scripts/run_openaire_indexing_retrievers.py
    ```

2.  **Retrieval Evaluation:**
    Update `scripts/run_openaire_retrieval_retrievers.py` with your PostgreSQL connection details.
    ```bash
    python scripts/run_openaire_retrieval_retrievers.py
    ```

### 7. Scalability Analysis (RQ5)

To evaluate indexing time, storage size, and query latency across increasing database sizes (10k to 50M rows) using the OpenAIRE dataset:

1.  Ensure you have the OpenAIRE PostgreSQL database configured.
2.  Update `scripts/run_openaire_indexing_scalability.py` with your `DB_CONFIG`.
3.  Run the scalability benchmark:

```bash
python scripts/run_openaire_indexing_scalability.py
```
