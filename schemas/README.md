## TensorIO Schemas

The schemas directory contains TensorIO model.json schemas, versioned in parallel to releases of TensorIO that include changes to the model.json file. Each backend maintains its own schema, as not all fields or values are supported by every backend.

There are currently three kinds of json files:

**model.json**

The model.json file describes a model's inputs and outputs and is packaged with a tiobundle when distributing a model. There are different schemas for different backends, currently tflite and tensorflow.

**tiotask**

A task describes a federated training task. It points to the model associated with that task and includes hyperparameters and placeholder values to inject into a model during on-device training.

**tioresult**

A result is uploaded by a client when it completes a training task and is packaged with the update to the model's weights. It includes the hyperparmeters that were specified by task along with device info, profiling information, and the training output, which is usually the local loss value.

### Verification

Install [ajv-cli](https://www.npmjs.com/package/ajv-cli) and verify a model.json file with the following command:

```bash
ajv -s schema.json -d model.json
```

### Versions

Updates to the schema are listed here.

#### Version 0.7.0

- Add model.modes field, an array of string values with support for the following values:
	- predict
	- train
	- eval
- Add train field to model.json with support for a train.ops array, which is an array of string values

#### Version 0.6.1

- Initial release
- Adds model.backend field whose value may be "tflite" or "tensorflow"
- Adds input.dtype and output.dtype field for array inputs and outputs with support for the following values:
	- uint8 (tflite and tensorflow)
	- float32 (tflite and tensorflow)
	- int32 (tensorflow)
	- int64 (tensorflow)
