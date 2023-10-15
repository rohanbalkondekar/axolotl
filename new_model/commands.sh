# To Begin Finetuning accelerate launch -m axolotl.cli.train ./path/to/config.yml
accelerate launch -m axolotl.cli.train mistral-orca-config.yml

# For inferencing the Finetuned Model
accelerate launch -m axolotl.cli.inference mistral-orca-config.yml --sample_packing False

# For viewing the Prepared Prompts on which the model is being finetuned on
accelerate launch -m axolotl.cli.train mistral-orca-config.yml --debug_text_only --debug

# Use Flag --prepare_ds_only to also see the corresponding tokens
accelerate launch -m axolotl.cli.train mistral-orca-config.yml --prepare_ds_only --debug