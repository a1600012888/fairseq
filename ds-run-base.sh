for task in RTE CoLA MRPC SST-2 # CoLA RTE MRPC SST-2 # QNLI QQP STS-B MNLI
do
    for cp_dir in  /home/tyzhang/models # mix_electra-electra-ns-byG+2-100C-0119 # mix_electra-electra-50
    do
	for step in baseline-roberta_base-03768-p512-b256-20w # 1_20000 2_50000 # 4_100000 7_200000 13_400000 20_600000 26_800000 32_1000000
	do
	    CUDA_VISIBLE_DEVICES=0 bash submit/glue/finetune/${cp_dir}/${step}/${task}.00.sh &
	    CUDA_VISIBLE_DEVICES=1 bash submit/glue/finetune/${cp_dir}/${step}/${task}.01.sh &
	    CUDA_VISIBLE_DEVICES=2 bash submit/glue/finetune/${cp_dir}/${step}/${task}.02.sh &
  	    CUDA_VISIBLE_DEVICES=3 bash submit/glue/finetune/${cp_dir}/${step}/${task}.03.sh & wait
	done
    done
done
