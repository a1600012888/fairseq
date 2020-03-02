for cp_dir in ~/models/ # mix_electra-electra-ns-byG+2-100C-0119
do
    for cp in baseline-roberta_base-03768-p512-b256-20w
    do
	python3 glue_hyperparam_search.py -p2 $cp_dir -c $cp -arch roberta_base -p1 .
    done
done
