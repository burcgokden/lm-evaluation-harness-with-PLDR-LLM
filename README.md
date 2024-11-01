# Language Model Evaluation Harness Suite with PLDR-LLM support

This repository is a fork of the LM Evaluation Harness Suite with PLDR-LLM model support pinned at version 0.4.3. This version was used to evaluate PLDR-LLM models on benchmark datasets for the research paper: [PLDR-LLM: Large Language Model From Power Law Decoder Representations](https://arxiv.org/abs/2410.16703).

# How to evaluate PLDR-LLMs on LM  Evaluation Harness Suite

- Clone this repository.
- Main branch has the PLDR-LLM support with LM Eval Harness Suite version pinned at 0.4.3.
- Install lm_eval module as described at [LM Evaluation Harness repository](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#install):
    ```sh
    cd lm-eval-harness-with-PLDR-LLM
    pip install -e .
    ```
- Install tinyBenchmarks package. Details can be found [here](lm_eval/tasks/tinyBenchmarks/README.md) and at the [tinyBenchmarks repository](https://github.com/felipemaiapolo/tinyBenchmarks). This package is used by LM Evaluation Harness to evaluate tinyBenchmarks scores.
    ```sh
    pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
    ```
- Add path of src/ and pldr_model_v500/ (or pldr_model_v900/) to sys.path from [PLDR-LLM github repository](https://github.com/burcgokden/LLM-from-Power-Law-Decoder-Representations). keras model file needs visibility to the module to deserialize model class.
    ```python
    import os
    import sys

    src_path=os.path.abspath("./LLM-from-Power-Law-Decoder-Representations/src")
    pldr_v500_path=os.path.abspath("./LLM-from-Power-Law-Decoder-Representations/src/pldr_model_v500")

    sys.path.insert(0, src_path)
    sys.path.insert(0, pldr_v500_path)
    ```

- Pretrained PLDR-LLM models and tokenizers used in the research paper can be found at [https://huggingface.co/fromthesky](https://huggingface.co/fromthesky) .
- Unzip the tokenizer model files. The extracted files are a tensorflow saved model folder, vocabulary (.txt, .vocab) and sentencepiece model (.model) files. We need to use the tensorflow saved model folder to load the tokenizer.
- Use the .keras model file and the tokenizer tensorflow saved model folder to load and evaluate the model. *Note:* The model and the support module for LM evaluation harness are not optimized for fast inference, evaluation of benchmarks take a long time.

    ```python
    import lm_eval

    #load pldrllm model and tokenizer
    saved_model=tf.keras.models.load_model("path/to/.keras/file")
    tok_model=tf.saved_model.load("path/to/tokenizer/tf/saved/model/folder")
    tokenizer=tok_model.en

    #initialize a pldrllm class object in lm_eval
    pldr_model=lm_eval.models.pldrllm.pldrllm(model=saved_model, tokenizer=tokenizer, 
                                            batch_size=8, max_length=1024, 
                                            max_gen_toks=256,
                                            temperature=1.0, top_p=1.0, top_k=0)

    #evaluate benchmarks
    task_manager = lm_eval.tasks.TaskManager()
    eval_results=lm_eval.simple_evaluate(model=pldr_model, tasks=["tinyBenchmarks_PLDR", "pldrllm_zeroshot"], 
                                        task_manager=task_manager)

    #print results
    print("SHOWING RESULTS:")
    print(eval_results["results"])
    ```


