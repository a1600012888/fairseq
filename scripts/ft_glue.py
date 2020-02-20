import argparse



task_name = ['MNLI', 'QNLI', 'QQP', 'RTE', 'SST-2', 'MRPC', 'CoLA', 'STS-B']

dataset_dir_name = ['MNLI-bin', 'QNLI-bin', 'QQP-bin',
                    'RTE-bin', 'SST-2-bin', 'MRPC-bin',
                    'CoLA-bin', 'STS-B-bin']

num_classes = [3,2,2,2,2,2,2,1]
lrs = [1e-5, 1e-5, 1e-5, 2e-5, 1e-5, 1e-5, 1e-5, 2e-5]
max_sentences = [32,32,32,16,32,16,16,16]
total_num_updates = [123873, 33112, 113272, 2036, 20935, 2296, 5336, 3598]
warm_updates = [7432, 1986, 28318, 122, 1256, 137, 320, 214]





if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--arch', type=str, default='robert_base',
                        choices=['robert_base', 'robert_leaner',
                                 'robert_base_deepsup', 'robert_base_norm_fl'])

    parser.add_argument('--task', type=str, default='SST-2',
                        choices=task_name)

    parser.add_argument('-d', type=int, default=0)

    parser.add_argument('-p', type=str)

    args = parser.parse_args()

    index = task_name.index(args.task)






    cmds =["CUDA_VISIBLE_DEVICES={}".format(args.d),
           " python3 train.py /home/tianyuan/data/glue-32768-fast/{}/ ".format(dataset_dir_name[index]),
    "--restore-file {} ".format(args.p),
    "--max-positions 512 --max-sentences {} ".format(max_sentences[index]),
    "--max-tokens 4400 --task sentence_prediction --reset-optimizer ",
    "--reset-dataloader --reset-meters --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 ",
    "--arch {} ".format(args.arch),
    "--criterion sentence_prediction ",
    "--num-classes {} ".format(num_classes[index]),
    "--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 --lr-scheduler polynomial_decay ",
    "--lr {} --total-num-update {} ".format(lrs[index], total_num_updates[index]),
    "--warmup-updates {} ".format(warm_updates[index]),
    "--max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;"
           ]

    cmd = ' '

    for c in cmds:
        cmd += c

    print(cmd)