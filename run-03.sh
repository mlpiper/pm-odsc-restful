#!/bin/bash

mlpiper --logging-level debug run \
	--input-model models/xgboost_model_21features_noise_1 \
	-f pipelines/02_mlpiper_simple_pipeline.json \
	-r ./ \
	-d /tmp/02_example \
	--force
