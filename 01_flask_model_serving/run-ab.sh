#!/bin/bash

ab -p predict_payload_1.json -T application/json -n 1000 -v 2 http://127.0.0.1:8888/predict/

