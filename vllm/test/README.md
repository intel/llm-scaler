# llm-scaler-vllm auto test (experimental)
llm-scaler-vllm auto test is an internally used batch test script generation tool, currently in the early stages of development.

## Quick start

```bash
python run_scripts/gen_run_script.py --config test_config/auto_test.yaml > my_test.sh

sh my_test.sh
```

## Architecture
```
test_config/
├── example.yaml # an example configuration
run_scripts/
├── gen_run_script.py # generate a runnable script
├── post_process_data.py # sort data and perform data visualization.
├── process_data.py # collect the raw data
└── script_config.py # parse configuration
analysis/ # default final result directory
auto_test_log/ # default test log directory
```