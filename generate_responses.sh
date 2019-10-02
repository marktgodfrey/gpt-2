for i in {1..6}
do
	python src/generate_conditional_samples.py --top_k 40 --temperature 1.0 --length 1024 --nsamples 5 --model_name 774M --run_name 1M_finetune_774M_run1 --checkpoint_dir /workspace/checkpoint --prompt_path ./prompts3/0$i.txt --out_path ./prompts3/out/1M_finetune_774M_run1-0$i.txt
done
