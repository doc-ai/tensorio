## TensorIO Schemas

The schemas directory contains TensorIO model.json schemas, versioned in parallel to releases of TensorIO that include changes to the model.json file. Each backend maintains its own schema, as not all fields or values are supported by every backend.

### Verification

Install [ajv-cli](https://www.npmjs.com/package/ajv-cli) and verify a model.json file with the following command:

```bash
ajv -s schema.json -d model.json
```

### Versions

Updates to the schema are listed here.

#### Version 0.6.1

- Initial release
- Adds model.backend field whose value may be "tflite" or "tensorflow"
- Adds input.dtype and output.dtype field for array inputs and outputs with support for the following values:
	- uint8 (tflite and tensorflow)
	- float32 (tflite and tensorflow)
	- int32 (tensorflow)
	- int64 (tensorflow)