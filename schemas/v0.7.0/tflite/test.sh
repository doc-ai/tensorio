#!/bin/bash

for f in tests/*.json
do
  ajv -s schema.json -d tests/0.json
done