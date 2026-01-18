# Run the validation routine script
py .\validate.py --use_numeric;

py .\validate.py --validation_dataset=output\val_original_ratio.jsonl --sample_size=8000 --use_numeric;

py .\validate.py --validation_dataset=additional_datasets\blackbox-1\val_original_ratio.jsonl --sample_size=8000 --use_numeric;

