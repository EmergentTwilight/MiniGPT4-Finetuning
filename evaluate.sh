before_trained_path="/mnt/data4/home/ziyuan/MiniGPT4-Finetuning/prerained_minigpt4_7b.pth"
trained_path=(
    "/mnt/data4/home/ziyuan/MiniGPT4-Finetuning/minigpt4/output/minigpt4_flickr_finetune/20250608110/checkpoint_0.pth"
    "/mnt/data4/home/ziyuan/MiniGPT4-Finetuning/minigpt4/output/minigpt4_flickr_finetune/20250608110/checkpoint_1.pth"
    "/mnt/data4/home/ziyuan/MiniGPT4-Finetuning/minigpt4/output/minigpt4_flickr_finetune/20250608110/checkpoint_2.pth"
    "/mnt/data4/home/ziyuan/MiniGPT4-Finetuning/minigpt4/output/minigpt4_flickr_finetune/20250608110/checkpoint_3.pth"
    "/mnt/data4/home/ziyuan/MiniGPT4-Finetuning/minigpt4/output/minigpt4_flickr_finetune/20250608110/checkpoint_4.pth"
)

export PYTHONPATH=$PYTHONPATH:/mnt/data4/home/ziyuan/MiniGPT4-Finetuning

echo "Evaluating pretrained model..."
proxy torchrun --master-port 3456 --nproc_per_node 1 eval_scripts/eval_flickr30k.py \
    --cfg-path /mnt/data4/home/ziyuan/MiniGPT4-Finetuning/eval_configs/minigpt4_flickr_eval.yaml \
    --ckpt "${before_trained_path}" \
    --save_path /mnt/data4/home/ziyuan/eval_output/pretrained


echo "Evaluating fine-tuned model..."
for path in "${trained_path[@]}"; do
    echo "Evaluating ${path}"
    proxy torchrun --master-port 3456 --nproc_per_node 1 eval_scripts/eval_flickr30k.py \
        --cfg-path /mnt/data4/home/ziyuan/MiniGPT4-Finetuning/eval_configs/minigpt4_flickr_eval.yaml \
        --ckpt "$path" \
        --save_path /mnt/data4/home/ziyuan/eval_output/finetuned_${path}
done
