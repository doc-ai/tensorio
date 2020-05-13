<a name="quantization"></a>
## Quantization and Dequantization

#### Contents

* [ Introduction ](#introduction)
* [ A Basic Example ](#quantization-basic-example)
* [ The Quantize Field ](#quantize-field)
* [ The Dequantize Field ](#dequantize-field)
* [ Selecting the Scale and Bias Terms ](#selecting-scale-bias)
* [ A Complete Example ](#quantization-complete-example)
* [ Quantized Models without Quantization ](#quantization-without-quantization)

<a name="introduction"></a>
#### Introduction

Quantization is a technique for reducing model sizes by representing weights with fewer bytes. Operations are then performed on these shorter byte representations. Quantized models trade accuracy for size. A full account of quantization is beyond the scope of this README, but more information may be found at https://www.tensorflow.org/performance/quantization.

In TF Lite, models represent weights with and perform operations on four byte floating point representations of data (`float_t`). These models receive floating point inputs and produce floating point outputs. Floating point models can represent numeric values in the range -3.4E+38 to +3.4E+38. Pretty sweet.

A quantized TF Lite model works with single byte representations `(uint8_t)`. It expects single byte inputs and it produces single byte outputs. A single unsigned byte can represent numbers in the range of 0 to 255. Still pretty cool.

When you use a quantized model but start with floating point data, you must first transform that four byte representation into one byte. This is called *quantization*. The model's single byte output must also be transformed back into a floating point representation, an inverse process called *dequantization*. Tensor/IO can do both for you.

Let's see what a basic quantization and dequantization look like.

<a name="quantization-basic-example"></a>
#### A basic example

First, when working with a quantized TF Lite model, change the *model.quantized* field in the *model.json* file to `true`:

```json
"model": {
  "file": "model.tflite",
  "quantized": true
},
```

For this example, let's say the input data coming from application space will always be in a floating point range from 0 to 1. Our quantized model requires those values to be in the range from 0 to 255. Quantization in TF Lite uniformly distributes a floating point range over a single byte range, so all we need to do here is apply a scaling factor of 255:

```
quantized_value = unquantized_value * 255
```

We can perform a sanity check with a few values:

```
Unquantized Value -> Quantized Value

0	->	0
0.5	->	127
1	->	255
```

Similarly, for this example let's say the output values produced by inference are a softmax probability distribution. The quantized model necessarily produces outputs in a range from 0 to 255, and we want to convert those back to a valid probability distribution. This will again be a uniform redistribution of values, and all we need to do is apply a scaling factor of 1.0/255.0:

```
unquantized_value = quantized_value * 1.0/255.0
```

Note that the transformations are inverses of one anther, and a sanity check produces the values we expect.

<a name="quantize-field"></a>
#### The Quantize Field

Instruct Tensor/IO to perform quantization by adding a *quantize* field to an input layer's description:

```json
"inputs": [
  {
    "name": "vector-input",
    "type": "array",
    "shape": [4],
    "quantize": {
      "scale": 255,
      "bias": 0
    }
  },
``` 

The *quantize* field is a dictionary value that may appear on *array* inputs only (*image* inputs use pixel normalization, more below). It contains either one or two fields: either both *scale* and *bias*, or *standard*.

*scale*

The *scale* field is a numeric value that specifies the scaling factor to apply to unquantized, incoming data.

*bias*

The *bias* field is a numeric value that specifies the bias to apply to unquantized, incoming data.

Together, Tensor/IO applies the following equation to any data sent to this layer:

```
quantized_value = (unquantized_value + bias) * scale
``` 

*standard*

The *standard* field is a string value corresponding to one of a number of commonly used quantization functions. Its presence overrides the *scale* and *bias* fields.

Tensor/IO currently has support for two standard quantizations. The ranges tell Tensor/IO *what range of values you are quantizing from*:

```json
"quantize": {
  "standard": "[0,1]"
}

"quantize": {
  "standard": "[-1,1]"
}
```

<a name="dequantize-field"></a>
#### The Dequantize Field

Dequantization is the inverse of quantization and is specified for an output layer with the *dequantize* field. The same *scale* and *bias* or *standard* fields are used.

For dequantization, scale and bias are applied in inverse order, where the bias value will be the negative equivalent of a quantization bias, and the scale will be the inverse of a quantization scale.

```
unquantized_value = quantized_value * scale + bias
```

For example, to dequantize from a range of 0 to 255 back to a range of 0 to 1, use a bias of 0 and a scale of 1.0/255.0:

```json
"outputs": [
  {
    "name": "vector-output",
    "type": "array",
    "shape": [4],
    "dequantize": {
      "scale": 0.004,
      "bias": 0
    }
  }
]
```

A standard set of dequantization functions is supported and describes *the range of values you want to dequantize back to*:

```json
"dequantize": {
  "standard": "[0,1]"
}

"dequantize": {
  "standard": "[-1,1]"
}
```

The *[0,1]* standard dequantization is particularly useful for softmax proability outputs with quantized models, when you must convert from a quantized range of [0,255] back to a valid probability distribution in the range of [0,1].

**Using Quantization and Dequantization**

Once these fields have been specified in a *model.json* file, no additional change is required in the Objective-C code. Simply send floating point values in and get floating point values back:

```objc
NSArray *vectorInput = @[ @(0.1f), @(0.2f), @(0.3f), @(0.4f) ]; // range in [0,1]

NSDictionary *features = @{
  @"vector-input": vectorInput
};

NSDictionary *inference = (NSDictionary *)[model runOn:features];

NSArray *vectorOutput = inference[@"vector-output"];

// vectorOutput[0] == 0.xx...
// vectorOutput[1] == 0.xx...
// vectorOutput[2] == 0.xx...
// vectorOutput[3] == 0.xx...

```

<a name="selecting-scale-bias"></a>
#### Selecting the Scale and Bias Terms

Selecting the scale and bias terms for either quantization or dequantization is a matter of solving a system of linear equations. 

**Quantization Scale and Bias**

For quantization, for example, you must know the range of values that are being quantized and the range of values you are quantizing to. The latter is always [0,255], while the former is up to you.

Then, given that the equation for quantizing a value is 

```
quantized_value = (unquantized_value + bias) * scale
```

You can form two equations:

```
(min + bias) * scale = 0
(max + bias) * scale = 255
```

And solve for scale and bias. Because the first equation is always set equal to zero, it is trivial to solve for bias. Use that result to solve for scale in the second equation:

```
bias  = -min
scale = 255 / (max - min)
```

For example, if you are quantizing from a range of values in [-1,1], then the scale and bias terms are:

```
bias  = -(-1) 
      = 1
      
scale = 255 / (1-(-1)) 
      = 255/2
      = 127.5
```

Which are exactly the values Tensor/IO uses when you specify a standard quantize string *"[-1,1]"*.

**Dequantization Scale and Bias**

For dequantization we do the same, using the equation:

```
unquantized_value = quantized_value * scale + bias
```

Form two equations:

```
min = 0 * scale   + bias
max = 255 * scale + bias
```

And solve for scale and bias:

```
bias  = min
scale = (max - bias) / 255
```

For example, if you are dequantizing from a range of values in [-1,1], then the scale and bias terms are:

```
bias  = -1

scale = (1-(-1)) / 255
      = 2/255
      = 0.0078
```

Which once again are the values Tensor/IO uses when you specify the standard dequantize string *"[-1,1]"*.

In both cases, you will need to know what the maximum and minimum values are that you are quantizing from and dequantizing to, and these must match the values you have used for your model.

<a name="quantization-complete-example"></a>
#### A Complete Example

Let's look at a complete example. This model is quantized and has two input layers and two output layers, with standard but different quantizations and dequantizations.

The model bundle will again have two files in it:

```
myquantizedmodel.tiobundle
  - model.json
  - model.tflite
```

Noting the value of the *model.quantized* field and the presence of *quantize* and *dequantize* fields in the input and output descriptions, the *model.json* file might look like: 

```json
{
  "name": "Example Quantized Model",
  "details": "This model takes two vector valued inputs and produces two vector valued outputs",
  "id": "my-awesome-quantized-model",
  "version": "1",
  "author": "doc.ai",
  "license": "Apache 2",
  "model": {
    "file": "model.tflite",
    "backend": "tflite",
    "quantized": true,
    "modes": ["predict"]
  },
  "inputs": [
    {
      "name": "foo-features",
      "type": "array",
      "shape": [4],
      "quantize": {
        "standard": "[0,1]"
      }
    },
    {
      "name": "bar-features",
      "type": "array",
      "shape": [8],
      "quantize": {
        "standard": "[-1,1]"
      }
    }
  ],
  "outputs": [
    {
      "name": "baz-outputs",
      "type": "array",
      "shape": [3],
      "dequantize": {
        "standard": "[0,1]"
      }
    },
    {
      "name": "qux-outputs",
      "type": "array",
      "shape": [6],
      "dequantize": {
        "standard": "[-1,1]"
      }
    }
  ]
}
```

Perform inference with this model as before:

```objc
NSArray *fooFeatures = @[ @(0.1f), @(0.2f), @(0.3f), @(0.4f) ]; // range in [0,1] 
NSArray *barFeatures = @[ @(-0.1f), @(0.2f), @(0.3f), (@0.4f), @(-0.5f), @(0.6f), @(-0.7f), @(0.8f) ]; // range in [-1,1] 

NSDictionary *features = @{
  @"foo-features": fooFeatures,
  @"bar-features": barFeatures
};

NSDictionary *inference = (NSDictionary *)[model runOn:features];

NSArray *bazOutputs = inference[@"baz-outputs"]; // length 3 in range [0,1]
NSArray *quxOutputs = inference[@"qux-outputs"]; // length 6 in range [-1,1]
```


<a name="quantization-without-quantization"></a>
#### Quantized Models without Quantization

The *quantize* field is optional for *array* input layers, even when the model is quantized. When you use a quantized model without including a *quantize* field, it is up to you to ensure that the data you send to Tensor/IO for inference is already quantized and that you treat output data as quantized. 

This may be the case when your input and output data is only ever in the range of [0,255], for example pixel data, or when you are quantizing the floating point inputs yourself before sending them to the model.

For example:

```objc
NSArray<NSNumber*> *unquantizedInput = @[ @(0.1f), @(0.2f), @(0.3f), @(0.4f) ]; // range in [0,1] 
NSArray<NSNumber*> *quantizedInput = [unquantizedInput map:^NSNumber * _Nonnull(NSNumber *  _Nonnull obj) {
  return @(obj.floatValue * 255); // convert from [0,1] to [0,255]
}];

NSDictionary *features = @{
  @"quantized-input": quantizedInput
};

NSDictionary *inference = (NSDictionary *)[model runOn:features];

NSArray *quantizedOutput = inference[@"quantized-output"]; // in range [0,255]
NSArray *dequantizedOutput = [quantizedOutput map:^NSNumber * _Nonnull(NSNumber *  _Nonnull obj) {
  return @(obj.unsignedCharValue * 1.0/255.); // convert from [0,255] to [0,1]]
}];
```


<< [Model JSON](ModelJSON.md) || [TOC](TOC.md) || [Image Data](ImageData.md) >>