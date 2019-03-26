CKPT=mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt
PB=Freeze_save
IAA=False
ILR=0.0005
CLSNUM=20
BATCH=32
DATASET=voc
MAXEP=10
MODEL=pureconv
MODELCMP=~/Documents/kendryte-model-compiler
LRDECAYFACTOR=1.0
LRDECAYSTEP=10
CLASSIFIER=False
OBJWEIGHT=5.0
NOOBJWEIGHT=0.5
IMG=data/003.jpg
OBJTHRESH=0.75
IOUTHRESH=0.5

all:
	@echo please use \"make train\" or other ...

train:
	python3 train.py \
			--train_set ${DATASET} \
			--class_num ${CLSNUM} \
			--train_classifier ${CLASSIFIER} \
			--pre_ckpt ${CKPT} \
			--model_def ${MODEL} \
			--augmenter ${IAA} \
			--anchor_file data/${DATASET}_anchors.list \
			--image_size 240 320 \
			--output_size 7 10 \
			--batch_size ${BATCH} \
			--rand_seed 3 \
			--max_nrof_epochs ${MAXEP} \
			--init_learning_rate ${ILR} \
			--learning_rate_decay_epochs ${LRDECAYSTEP} \
			--learning_rate_decay_factor ${LRDECAYFACTOR} \
			--obj_weight ${OBJWEIGHT} \
			--noobj_weight ${NOOBJWEIGHT} \
			--obj_thresh ${OBJTHRESH} \
			--iou_thresh ${IOUTHRESH} \
			--log_dir log

freeze:
	python3 freeze_graph.py \
			--class_num ${CLSNUM} \
			--anchor_file data/${DATASET}_anchors.list \
			${MODEL} \
			240 320 \
			${CKPT} \
			${PB}.pb \
			Yolo/Final/conv2d/BiasAdd  
			
inference:
	python3	inference.py \
			--pb_path ${PB}.pb \
			--class_num ${CLSNUM} \
			--anchor_file data/${DATASET}_anchors.list \
			--image_size 240 320 \
			--image_path ${IMG}
			
tflite:
	toco --graph_def_file=${PB}.pb \
			--output_file=${PB}.tflite \
			--output_format=TFLITE \
			--input_shape=1,240,320,3 \
			--input_array=Input_image \
			--output_array=Yolo/Final/conv2d/BiasAdd \
			--inference_type=FLOAT && \
			cp -f ${PB}.tflite ~/Documents/nncase/tflites/

nncase_convert:
	cd ~/Documents/nncase/ && \
	/home/zqh/Documents/nncase/ncc-linux-x86_64/ncc \
			-i tflite -o k210code \
			--dataset dataset/flowers \
			--postprocess 0to1 \
			tflites/${PB}.tflite build/model.c

kmodel_convert:
	cp -f ${PB}.pb ${MODELCMP}/pb_files/ && \
	cd ${MODELCMP} && \
	python3 __main__.py --dataset_input_name Input_image:0 \
			--dataset_loader "dataset_loader/img_0_1.py" \
			--image_h 240 --image_w 320 \
			--dataset_pic_path dataset/example_img \
			--model_loader "model_loader/pb" \
			--pb_path "pb_files/${PB}.pb" \
			--tensor_output_name Yolo/Final/conv2d/BiasAdd \
			--eight_bit_mode True


anchor_list:
	python3	make_anchor_list.py \
			--train_set ${DATASET} \
			--max_iters 10 \
			--is_random False \
			--is_plot True \
			--in_hw 240 320 \
			--out_hw 7 10 \
			--out_anchor_file data/${DATASET}_anchors.list