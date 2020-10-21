gpui="5"
starti=65
endi=77

offseti=13
_offseti=12

########

for i in $(seq $starti $offseti $endi)
do
CUDA_VISIBLE_DEVICES=$gpui python bert_influence.py --output_dir="new_aws_dirctr_influence_outputs_bert_full_e3" --data_dir="resources/processed_dataset/"\
    --bert_model="bert-base-uncased" --do_lower_case --trained_model_dir="model/aws_dirctr_tagger_output_bert_full_e3/" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2020\
    --damping=3e-3 --scale=1e8 --lissa_repeat=1 --lissa_depth_pct=0.25\
    --start_test_idx=$i --end_test_idx=$((i+_offseti))\
    --logging_steps=200 --full_bert --alt_mode="dirctr"
done

for i in $(seq $starti $offseti $endi)
do
CUDA_VISIBLE_DEVICES=$gpui python bert_embin.py --output_dir="new_aws_dirctr_embin_outputs_bert_full_e3" --data_dir="resources/processed_dataset/"\
    --bert_model="bert-base-uncased" --do_lower_case --trained_model_dir="model/aws_dirctr_tagger_output_bert_full_e3/" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2020\
    --start_test_idx=$i --end_test_idx=$((i+_offseti))\
    --logging_steps=200 --full_bert --alt_mode="dirctr"
done

for i in $(seq $starti $offseti $endi)
do
CUDA_VISIBLE_DEVICES=$gpui python bert_trackin.py --output_dir="new_aws_dirctr_trackin_outputs_bert_full_e3" --data_dir="resources/processed_dataset/"\
    --bert_model="bert-base-uncased" --do_lower_case --trained_model_dir="model/aws_dirctr_tagger_output_bert_full_e3/" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2020\
    --start_test_idx=$i --end_test_idx=$((i+_offseti))\
    --logging_steps=200 --full_bert --num_trained_epoch=3 --alt_mode="dirctr"
done

