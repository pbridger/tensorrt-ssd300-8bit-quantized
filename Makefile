
CONTAINER_NAME := tensorrt-ssd300:latest
# update to match your COCO 2017 location, or remove that volume mapping if you don't need to run --mode=coco
DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --net=bridge --ulimit core=0 --ipc=host -v $(shell pwd):/app -v /data/coco2017:/data/coco2017
PROFILE_CMD := profile -t cuda,cublas,cudnn,nvtx,osrt --force-overwrite=true --duration=30 --delay=6


### External - to be used from outside the container ###

build-container: Dockerfile
	docker build -f $< -t ${CONTAINER_NAME} .

run-container: build-container
	${DOCKER_CMD} ${CONTAINER_NAME}


logs/%.svg: logs/%.rec
	cat $< | svg-term --no-cursor > $@


logs/ssd300.fp32.b16.k256.pytorch.rec:
	rm -f logs/ssd300.fp32.b16.k256.pytorch.bench $@
	asciinema rec $@.tmp -c 'make --no-print-directory logs/ssd300.fp32.b16.k256.pytorch.bench sleep'
	python optrec.py $@.tmp $@

logs/ssd300.fp16.b16.k256.pytorch.rec:
	rm -f logs/ssd300.fp16.b16.k256.pytorch.bench $@
	asciinema rec $@.tmp -c 'make --no-print-directory logs/ssd300.fp16.b16.k256.pytorch.bench sleep'
	python optrec.py $@.tmp $@

logs/ssd300.fp32.b16.k256.trt.rec:
	rm -f logs/ssd300.fp32.b16.k256.trt.bench $@
	asciinema rec $@.tmp -c 'make --no-print-directory logs/ssd300.fp32.b16.k256.trt.bench sleep'
	python optrec.py $@.tmp $@

logs/ssd300.fp16.b16.k256.trt.rec:
	rm -f logs/ssd300.fp16.b16.k256.trt.bench $@
	asciinema rec $@.tmp -c 'make --no-print-directory logs/ssd300.fp16.b16.k256.trt.bench sleep'
	python optrec.py $@.tmp $@

logs/ssd300.int8.b16.k256.trt.rec:
	rm -f logs/ssd300.int8.b16.k256.trt.bench $@
	asciinema rec $@.tmp -c 'make --no-print-directory logs/ssd300.int8.b16.k256.trt.bench sleep'
	python optrec.py $@.tmp $@

sleep:
	@sleep 4
	@echo '-'


### Internal - to be used from within the container (after run-container) ###

### Build models

models/ssd300.fp32.b16.k256.plan:
	python ssd300_trt.py --mode=export --precision=fp32 --batch-dim=16 --topk=256 --runtime=trt

models/ssd300.fp16.b16.k256.plan:
	python ssd300_trt.py --mode=export --precision=fp16 --batch-dim=16 --topk=256 --runtime=trt

models/ssd300.int8.b16.k256.plan:
	python ssd300_trt.py --mode=export --precision=int8 --batch-dim=16 --topk=256 --runtime=trt

models: models/ssd300.fp32.b16.k256.plan models/ssd300.fp16.b16.k256.plan models/ssd300.int8.b16.k256.plan

### COCO evaluation

logs/ssd300.fp32.b16.k256.pytorch.coco: ssd300_trt.py
	python $< --mode=coco --precision=fp32 --batch-dim=16 --topk=256 --runtime=pytorch --output-path=$@

logs/ssd300.fp16.b16.k256.pytorch.coco: ssd300_trt.py
	python $< --mode=coco --precision=fp16 --batch-dim=16 --topk=256 --runtime=pytorch --output-path=$@

logs/ssd300.fp32.b16.k256.trt.coco: ssd300_trt.py models/ssd300.fp32.b16.k256.plan
	python $< --mode=coco --precision=fp32 --batch-dim=16 --topk=256 --runtime=trt --output-path=$@

logs/ssd300.fp16.b16.k256.trt.coco: ssd300_trt.py models/ssd300.fp16.b16.k256.plan
	python $< --mode=coco --precision=fp16 --batch-dim=16 --topk=256 --runtime=trt --output-path=$@

logs/ssd300.int8.b16.k256.trt.coco: ssd300_trt.py models/ssd300.int8.b16.k256.plan
	python $< --mode=coco --precision=int8 --batch-dim=16 --topk=256 --runtime=trt --output-path=$@

coco: logs/ssd300.fp32.b16.k256.pytorch.coco logs/ssd300.fp16.b16.k256.pytorch.coco logs/ssd300.fp32.b16.k256.trt.coco logs/ssd300.fp16.b16.k256.trt.coco logs/ssd300.int8.b16.k256.trt.coco

### Throughput benchmarking

logs/ssd300.fp32.b16.k256.pytorch.bench: ssd300_trt.py
	python $< --mode=bench --precision=fp32 --batch-dim=16 --topk=256 --runtime=pytorch --output-path=$@

logs/ssd300.fp16.b16.k256.pytorch.bench: ssd300_trt.py
	python $< --mode=bench --precision=fp16 --batch-dim=16 --topk=256 --runtime=pytorch --output-path=$@

logs/ssd300.fp32.b16.k256.trt.bench: ssd300_trt.py models/ssd300.fp32.b16.k256.plan
	python $< --mode=bench --precision=fp32 --batch-dim=16 --topk=256 --runtime=trt --output-path=$@

logs/ssd300.fp16.b16.k256.trt.bench: ssd300_trt.py models/ssd300.fp16.b16.k256.plan
	python $< --mode=bench --precision=fp16 --batch-dim=16 --topk=256 --runtime=trt --output-path=$@

logs/ssd300.int8.b16.k256.trt.bench: ssd300_trt.py models/ssd300.int8.b16.k256.plan
	python $< --mode=bench --precision=int8 --batch-dim=16 --topk=256 --runtime=trt --output-path=$@

bench: logs/ssd300.fp32.b16.k256.pytorch.bench logs/ssd300.fp16.b16.k256.pytorch.bench logs/ssd300.fp32.b16.k256.trt.bench logs/ssd300.fp16.b16.k256.trt.bench logs/ssd300.int8.b16.k256.trt.bench

### Nsight systems report generation

logs/ssd300.fp32.b16.k256.pytorch.qdrep: ssd300_trt.py
	nsys ${PROFILE_CMD} -o $@ python $< --mode=bench --precision=fp32 --batch-dim=16 --topk=256 --runtime=pytorch

logs/ssd300.fp16.b16.k256.pytorch.qdrep: ssd300_trt.py
	nsys ${PROFILE_CMD} -o $@ python $< --mode=bench --precision=fp16 --batch-dim=16 --topk=256 --runtime=pytorch

logs/ssd300.fp32.b16.k256.trt.qdrep: ssd300_trt.py models/ssd300.fp32.b16.k256.plan
	nsys ${PROFILE_CMD} -o $@ python $< --mode=bench --precision=fp32 --batch-dim=16 --topk=256 --runtime=trt

logs/ssd300.fp16.b16.k256.trt.qdrep: ssd300_trt.py models/ssd300.fp16.b16.k256.plan
	nsys ${PROFILE_CMD} -o $@ python $< --mode=bench --precision=fp16 --batch-dim=16 --topk=256 --runtime=trt

logs/ssd300.int8.b16.k256.trt.qdrep: ssd300_trt.py models/ssd300.int8.b16.k256.plan
	nsys ${PROFILE_CMD} -o $@ python $< --mode=bench --precision=int8 --batch-dim=16 --topk=256 --runtime=trt

qdrep: logs/ssd300.fp32.b16.k256.pytorch.qdrep logs/ssd300.fp16.b16.k256.pytorch.qdrep logs/ssd300.fp32.b16.k256.trt.qdrep logs/ssd300.fp16.b16.k256.trt.qdrep logs/ssd300.int8.b16.k256.trt.qdrep

### Logs for article

logs/subscript_assignment: subscript_assignment.py
	-python $< >$@ 2>&1 

logs/masking: masking.py
	-python $< >$@ 2>&1 

logs: logs/masking logs/subscript_assignment


