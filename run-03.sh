#!/bin/bash

mlpiper --logging-level debug run \
	--input-model models/xgboost_model_21features_noise_1 \
	-f pipelines/03_mlpiper_simple_pipeline.json \
	-r ./ \
	-d /tmp/03_example \
	--force
