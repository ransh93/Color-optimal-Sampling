# Color-optimal-Sampling
Finding an optimal Sampling for Digital Color Camera using deep learning methods for colorization

We used a DNN which primarly aimed to improve image colorization result in order to train that network to find an optimal color ratio and Sampling for Digital Color Camera
In order to run our code, you can follow this guied and ask some quistions using our personal emails (presented at the end)

<b>Assumptions:</b>

1. Under ...\root\dataset\ilsvrc2012\ there should be three sub-directories: 'train', test', 'val'.
2. Images should be contained in a subfolder under each of the above dirs. For example: ...\test\1\*.JPEG.


<b>Files:</b>

1. train.py - used to train the 'generic' network.
2. train_specific.py - used to train the 'specific' network.
3. test.py - used to run a test that reports PSNR & MSE statistics and sample images.
4. gen_graphs.py - used to create two heatmap graphs: PSNR, MSE.


<b>Training:</b>
1. Important Flags:
	1.	name			- the name of network
	2.	niter			- the number of epochs
	3.	save_epoch_freq	- controls the frequency in which the network is being saved during training
	4.	niter_decay		- number of epochs to decay learning rate
	5.	lr				- the learning rate
	6.	phase			- train\test
	7.	output_nc		- output dimension
	8.	input_nc		- input dimension
	9.	loadSize		- scale images to this size
	10.	fineSize		- crop images to this size
	11. batch_size		- the batch size
	12.	red_ratio		- % of red values in mask (specific network only)
	13.	green_ratio		- % of green values in mask (specific network only)
	
2. Command Examples:
	1. python3 train.py --name rgb_full_128 --niter 20 --save_epoch_freq 10 --niter_decay 0 --lr 0.00001 --phase train --output_nc 3 --input_nc 4 --loadSize 128 --fineSize 128
	2. python3 train_specific.py --name rgb_specific_70_20 --red_ratio 70 --green_ratio 20 --niter 20 --save_epoch_freq 10 --niter_decay 0 --lr 0.00001 --phase train --output_nc 3 --input_nc 4 --loadSize 128 --fineSize 128
	
	
<b>Testing:</b>
1. Important Flags:
	1. how_many	- how many iterations to test on
	2. load_mask	- load specific mask of this network (specific network only)
2. Command Example:
	1. python3 test.py --name rgb_specific_10_60 --loadSize 128 --fineSize 128 --input_nc 4 --output_nc 3 --how_many 100 --load_mask True

<b>Graphs generator:</b>
1. Command Example:
	1. python3 gen_graphs.py
	

<b>
Many thanks for this article and code base for giving us good start for our research, it has been a pleasure working with your code:
<br>
https://richzhang.github.io/ideepcolor/ <br>
https://github.com/richzhang/colorization-pytorch
</b>
