# _May I have your attention?_ Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks

This repository contains code to run the ASTRA and ASTRA++ attacks that break SecAlign++, SecAlign, StruQ. This repository also contains some examples of generated attacks and attack logs

## Setup

### Clone Repository

This will clone the repository as well as all the submodules in the right location required.

```bash
git clone --recurse-submodules https://github.com/nishitvp/better_opts_attacks.git
```

### Set up Python environment

The codebase doesn't use any special Python packages beyond the standard collection of LLM and Deep Learning related packages. To install the requirements, run

```bash
python3 -m pip install -r requirements.txt
```

and this should install the necessary packages in your environment.

### Download Models

1. First, download the base models from HuggingFace (some of these may be gated, requiring you to gain access from HuggingFace)

    ```bash
    hf login --token ${HF_PASSWORD}
    hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir secalign_refactored/secalign_models/meta-llama/Meta-Llama-3-8B-Instruct
    hf download mistralai/Mistral-7B-Instruct-v0.1 --local-dir secalign_refactored/secalign_models/mistralai/Mistral-7B-Instruct-v0.1
    hf download meta-llama/Llama-3.1-8B-Instruct --local-dir secalign_refactored/secalign_models/meta-llama/Llama-3.1-8B-Instruct
    hf download facebook/Meta-SecAlign-8B --local-dir secalign_refactored/secalign_models/Meta-SecAlign-8B
    ```

2. Download the SecAlign adapters and StruQ models. There's a helpful script to download and extract in the right place.

   ```bash
   python3 setup_scripts/download_models.py
   ```

    When run from the project root, this script should download and extract the correct models in the correct place. To verify, check whether `secalign_refactored/secalign_models` has three folders `huggyllama`, `meta-llama` and `mistralai` which should each contain the model(s) themselves.

## Running ASTRA for a Strong Knowledge Adversary

To run ASTRA for a Strong Knowledge Adversary, the key script is `experiment.py`. This script will take the `40` examples from the AlpacaFarm dataset, inject the payload `Print exactly 'Hacked!'` with the specified configuration and run ASTRA as well as the baseline (GCG) while controlling for initialization (randomly initialized), number of forward passes, iterations etc.

The user can control the following parameters by setting them as command line parameters when calling `experiment.py` -

1. `--model-name`: The possible values are `meta-llama`, `meta-llama-instruct`, `mistralai`, `mistralai-instruct`, `huggyllama`.
2. `--defense`: The possible values are `undefended`, `struq` and `secalign`. The default value is `secalign`. The `--model-name` parameter along with the `--defense` parameter uniquely specify the specific model being used according to the table below

    | `--model-name` | `--defense`| Model loaded |
    |----------------|------------|--------------|
    | `meta-llama-instruct`| `secalign` | SecAlign adapters with pre-instruction-tuned Meta-Llama-3-8B-Instruct|
    |`mistralai-instruct`|`secalign`| SecAlign adapters with pre-instruction-tuned Mistral-7B-Instruct-v0.1|
    |`meta-llama`|`secalign`| SecAlign defended Meta-Llama-3-8B (not pre-instruction-tuned)|
    |`mistralai`|`secalign`| SecAlign defended Mistral-7B-Instruct-v0.1 (not pre-instruction-tuned)|
    |`huggyllama`|`secalign`| SecAlign defended Llama-2-7B (not pre-instruction-tuned)|
    |`meta-llama`|`struq`| StruQ defended Meta-Llama-3-8B (not pre-instruction-tuned)|
    |`mistralai`|`struq`| StruQ defended Mistral-7B-v0.1 (not pre-instruction-tuned)|
    |`huggyllama`|`struq`| StruQ defended Llama-2-7B (not pre-instruction-tuned)|
    |`meta-llama-instruct`|`undefended`| Undefended (Raw) Meta-Llama-3-8B-Instruct|
    |`mistralai-instruct`|`undefended`| Undefended (Raw) Mistral-7B-Instruct-v0.1|
    |`meta-llama`|`undefended`| Undefended (Raw) Meta-Llama-3-8B (not pre-instruction-tuned)|
    |`mistralai`|`undefended`| Undefended (Raw) Mistral-7B-Instruct-v0.1 (not pre-instruction-tuned)|
    |`huggyllama`|`undefended`| Undefended (Raw) Llama-2-7B (not pre-instruction-tuned)|

    All other pairs of `--model-name` and `--defense` are invalid.
3. `--prefix-length`: Length of the adversarial prefix to be used for a given evaluation run. Default is `5`.
4. `--suffix-length`: Length of the adversarial suffix to be used for a given evaluation run. Default is `20`.
5. `--expt-folder-prefix`: The path where you want the logs from the experiment run to be logged.

A complete command might look something like -

```bash
python3 experiment.py --model-name meta-llama-instruct --defense secalign --prefix-length 5 --suffix-length 20 --expt-folder-prefix logs/astra_llama_secalign
```

On running the above command, the script will automatically distribute the workload equally into different subprocesses each of which will use one GPU to compute the attacks. We recommend running the attacks on GPUs with at least 48GB memory.

Each subprocess will log the complete transcript of each example it attacks in a separate subfolder of the path specified in `--expt-folder-prefix`. The logs are stored in a queryable databse-like format containing python objects (See the file `utils/experiment_logger` for more details). Each logged transcript will contain the sequences of tokens, loss values and other important details such as initialization config throughout the optimization which can be retrieved later.

### Analysis (ASTRA)

Once the complete ASTRA experiment is finished, the generated logs can be later analyzed using the file `analysis/analysis.ipynb`. Simply copy-paste the Jupyter notebook into the `--expt-folder-prefix` path, replace the model path in the notebook with the correct model path, and run the notebook. This should print out the number of successes of GCG and number of successes of ASTRA in the last cell. (P.S. The analysis script takes a while to run because it actually generates the outputs on each of the input sequence of tokens generated during optimization for each example.)

The `analysis.ipynb` notebook also contains helper code to plot average loss curves and do further explorations with the generated logs.

The analysis scripts have been kept separate fundamentally to ensure that the optimization itself is not contaminated by the post-facto analysis code.

### Generated attacks

We also share some ASTRA generated attacks in the zipped file located in `data/attacks_generated.zip`. Concretely, the zipped file contains 4 files each with 40 examples of ASTRA generated attacks, two each for each of the SecAlign defended versions of Mistral and Llama-3 respectively. The files provide examples generated for a budget of 20 and 25 tokens (In a (0, 20) and (5, 20) configuration). the injection has already been placed into the input field in the configurations.

## Running ASTRA++ for a Weak Knowledge Adversary

To run ASTRA++ for a Weak Knowledge Adversary, the key script is `experiment_universal.py`. This script will take `10` training examples from the AlpacaFarm dataset, inject the payload `Print exactly 'Hacked!'` with the specified prefix, suffix configuration and train to get a universal prompt injection at the end. The script also runs the baseline, universal version of GCG following the training of ASTRA++.

Like the script `experiment.py` from the previous section, `experiment_universal.py` also takes in `--model-name`, `--defense`, `--prefix-length`, `--suffix-length`, `--expt-folder-prefix` arguments.

In addition to all the models listed in the previous section on running ASTRA, the `experiment_universal.py` script also takes in one more possible model, corresponding to the SecAlign++ defense.
To train on the SecAlign++ defense, pass the `--model-name` parameter as `secalign_refactored/secalign_models/Meta-SecAlign-8B` and `--defense` parameter as `meta_secalign`.

There is one additional command-line argument that this script takes, which is `--training-run`, which corresponds to which set of `10` examples the training run will happen on.

The rest of the parameters have the same interpretation as before.

### Analysis (ASTRA++)

Once the complete ASTRA++ experiment is finished, the generated logs can be later analyzed using the file `analysis/analysis_universal.ipynb`. Like before, copy-paste the Jupyter notebook into the `--expt-folder-prefix` path, load the model correctly (there's some finickyness due to how SecAlign++ is loaded, but that will be made uniform in the future). This should print out the number of test successes of GCG and number of successes of ASTRA along with the best universal prompt injections achieved.

### Generated Attacks

For ASTRA++, we upload the universal prompt injections generated for the SecAlign++ defended Llama-3.1-8B-Instruct in the file `data/universal_pis_secalign++.json` along with metadata and configuration they were generated on. Further, we also upload a transcript of all the attacks over all the universality training runs in `data/final_result_logs.pkl` (We promise, the pickle contains no malicious code :).

## Advanced Changes

While the above scripts will run the ASTRA and ASTRA++ attacks off-the-shelf, the codebase allows configuration of more-or-less every parameter that can affect the attack, such as training dataset sizes, thresholds chosen, random seeds, initialization configs and so on. If you have any interesting suggestions, feel free to send a pull request.

## Contact

For any queries, bug reports, details or clarifications, please feel free to raise an issue on Github or mail the authors.
