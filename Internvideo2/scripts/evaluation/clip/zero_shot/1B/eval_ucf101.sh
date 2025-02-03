JOB_NAME='zs_ucf101.'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
zero_shot=True
evaluate=True
export PYTHONPATH=${PYTHONPATH}:.

python tasks_clip/retrieval.py \
    /root/autodl-tmp/InternVideo/InternVideo2/multi_modality/scripts/evaluation/clip/zero_shot/1B/config_ucf101.py \
    pretrained_path /root/autodl-tmp/InternVideo2-Stage1-1B-224p-f8-k710/1B_ft_k710_f8.pth
    output_dir ${OUTPUT_DIR}
