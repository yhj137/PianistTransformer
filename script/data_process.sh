export PYTHONPATH=.

python src/data_process/01_generate_pretrain_data.py

python src/data_process/02_generate_pretrain_ex_data.py

python src/data_process/03_generate_pretrain_map_data.py \
    --input_dir data/processed/pretrain/raw/aria \
    --output_dir data/processed/pretrain/cut/aria \
    --enhanced

python src/data_process/03_generate_pretrain_map_data.py \
    --input_dir data/processed/pretrain/raw/scores \
    --output_dir data/processed/pretrain/cut/scores

python src/data_process/04_convert_to_arrow.py \
    --input_dir data/processed/pretrain/cut/aria \
    --output_dir data/processed/pretrain/arrow/aria

python src/data_process/04_convert_to_arrow.py \
    --input_dir data/processed/pretrain/cut/scores \
    --output_dir data/processed/pretrain/arrow/scores

python src/data_process/05_merge_arrow.py \
    --input_dirs data/processed/pretrain/arrow/aria data/processed/pretrain/arrow/scores \
    --output_dir data/processed/pretrain/arrow/all

python src/data_process/06_generate_sft_data.py
