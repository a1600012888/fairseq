MP=~/exps/fairseq/SE-F-SE-AVG-16-roberta_base-03768-p512-b256/checkpoint_7_200000.pt
ARCH='roberta_base-se'

python3 scripts/ft_glue.py --arch ${ARCH} --task SST-2 -p ${MP}  -d 1 > sst1.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task SST-2 -p ${MP}  -d 2 > sst2.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task SST-2 -p ${MP}  -d 3 > sst3.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task SST-2 -p ${MP}  -d 0 | tee sst0.txt

sleep 10

python3 scripts/ft_glue.py --arch ${ARCH} --task RTE -p ${MP} -d 1 >  rte1.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task RTE -p ${MP} -d 2 >  rte2.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task RTE -p ${MP}  -d 3 >  rte3.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task RTE -p ${MP}  -d 0 | tee rte0.txt

sleep 10

python3 scripts/ft_glue.py --arch ${ARCH} --task MRPC -p ${MP} -d 1  > mrpc1.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task MRPC -p ${MP}  -d 2  > mrpc2.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task MRPC -p ${MP}  -d 3  > mrpc3.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task MRPC -p ${MP}  -d 0 | tee mrpc0.txt

sleep 10

python3 scripts/ft_glue.py --arch ${ARCH} --task QNLI -p ${MP}  -d 1 > qnli1.txt &
python3 scripts/ft_glue.py --arch ${ARCH} --task QNLI -p ${MP}  -d 0 | tee qnli0.txt
