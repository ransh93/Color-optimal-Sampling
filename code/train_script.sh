#!/bin/bash

# train full
python3 train.py --name rgb_full_128 --niter 20 --save_epoch_freq 10 --niter_decay 0 --lr 0.00001 --phase train --output_nc 3 --input_nc 4 --loadSize 128 --fineSize 128


# train specific
python3 train_specific.py --name rgb_specific_70_20 --red_ratio 70 --green_ratio 20 --niter 20 --save_epoch_freq 10 --niter_decay 0 --lr 0.00001 --phase train --output_nc 3 --input_nc 4 --loadSize 128 --fineSize 128
python3 train_specific.py --name rgb_specific_50_30 --red_ratio 50 --green_ratio 30 --niter 20 --save_epoch_freq 10 --niter_decay 0 --lr 0.00001 --phase train --output_nc 3 --input_nc 4 --loadSize 128 --fineSize 128
python3 train_specific.py --name rgb_specific_10_60 --red_ratio 10 --green_ratio 60 --niter 20 --save_epoch_freq 10 --niter_decay 0 --lr 0.00001 --phase train --output_nc 3 --input_nc 4 --loadSize 128 --fineSize 128
