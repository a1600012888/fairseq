for cp_dir in ~/exps/fairseq/SE-F-SE-F-16-roberta_base-03768-p512-b256/ # mix_electra-electra-ns-byG+2-100C-0119
do
    for cp in 7_200000
    do
	python3 glue_hyperparam_search.py -p2 $cp_dir -c $cp -arch roberta_learner -p1 .
    done
done
