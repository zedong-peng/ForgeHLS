# How to start:

After installing the llama-factory, run the following command to start the training:

```bash
llamafactory-cli train ./*.yaml
# attention: cd to the yaml directory and run llamafactory-cli train *.yaml
# do not run llamafactory-cli train ./yaml/*.yaml
# this will cause the relative file not found error
```

the `--template` is the model name, check https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models