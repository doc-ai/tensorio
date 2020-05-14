<a name="model-json"></a>
## The Model JSON File

#### Contents

* [ Basic Structure ](#introduction)
* [ Basic Structure ](#basic-structure)
* [ The Model Field ](#model-field)
* [ The Inputs Field ](#inputs-field)
* [ The Outputs Field ](#outputs-field)
* [ The Options Field ](#options-field)
* [ A Complete Example ](#complete-example)

<a name="basic-structure"></a>
#### Introduction

One of Tensor/IO's goals is to reduce the amount of new code required to integrate models into an application.

The primary work of using a model on iOS involves copying bytes of the right length to the right place. TF Lite, for example, is a C++ library, and the input and output tensors are exposed as C style buffers. In order to use a model we must copy byte representations of our input data into these buffers, ask the library to perform inference on those bytes, and then extract the byte representations back out of them.

Model interfaces can vary widely. Some models may have a single input and single output layer, others multiple inputs with a single output, or vice versa. The layers may be of varying shapes, with some layers taking single values, others an array of values, and yet others taking matrices or volumes of higher dimensions. Some models may work on four byte, floating point representations of data, while others use single byte, unsigned integer representations. The latter are called *quantized* models, more on them below.

Consequently, every time we want to try a different model, or even the same model with a slightly different interface, we must modify the code that moves bytes into and out of  buffers.

Tensor/IO abstracts the work of copying bytes into and out of tensors and replaces that imperative code with a declarative language you already know: JSON.

The *model.json* file in a Tensor/IO bundle contains metadata about the underlying model as well as a description of the model's input and output layers. Tensor/IO parses those descriptions and then, when you perform inference with the model, internally handles all the byte copying operations, taking into account layer shapes, data sizes, data transformations, and even output labeling. All you have to do is provide data to the model and ask for the data out of it.

The *model.json* file is the primary point of interaction with the Tensor/IO library. Any code you write to prepare data for a model and read data from a model will depend on a description of the model's input and output layers that you provide in this file.

Let's have a closer look.

<a name="basic-structure"></a>
#### Basic Structure

The *model.json* file has the following basic structure:

```json
{
  "name": "ModelName",
  "details": "Description of your model",
  "id": "unique-identifier",
  "version": "1",
  "author": "doc.ai",
  "license": "MIT",
  "model": {
    "file": "model.tflite",
    "quantized": false,
    "type": "image.classification.imagenet",
    "backend": "tflite",
    "modes": ["predict"]
  },
  "inputs": [
    {
      ...
    }
  ],
  "outputs": [
    {
      ...
    }
  ]
}

```

In addition to the model's metadata, such as name, identifier, version, etc, all of which are required, the JSON file also includes three additional, required entries:

1. The *model* field is a dictionary that contains information about the model itself
2. The *inputs* field is an array of dictionaries that describe the model's input layers
3. The *outputs* field is an array of dictionaries that describe the model's output layers

<a name="model-field"></a>
#### The Model Field

The model field is a dictionary that itself contains two to five entries:

```json
"model": {
  "file": "model.tflite",
  "backend": "tflite",
  "quantized": false,
  "modes": ["predict"],
  "type": "image.classification.imagenet",
  "class": "MyOptionalCustomClassName"
}
```

*file*

The *file* field is a string value that contains the name of your model file. For TF Lite models it is the file with the *.tflite* extension that resides at the top level of your model bundle folder. For TensorFlow models it is the directory produced by [Estimator.export_saved_model](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_saved_model) or by [tf.saved_model.simple_save](https://www.tensorflow.org/guide/saved_model#simple_save) and which contains the saved_model.pb file and a variables directory. This folder must reside at the top level of your model bundle.

This field is required.

*backend*

Tensor/IO supports multiple machine learning libraries, or backends. The *backend* field is a string that identifies which backend to use for this model. TensorFlow and TF Lite are currently supported, and this field should indicate which one to use, either of:

- tflite
- tensorflow

This field is required. 

*quantized*

The *quantized* field is a boolean value that is `true` when your model is quantized and `false` when it is not. Quantized models perform inference on single byte, unsigned integer representations of your data (`uint8_t`). Quantized models involve additional considerations which are discussed below.

This field is required.

*modes*

The *modes* field is an array of strings that describes the modes supported by this model, for example, if the model supports prediction, training, or evaluation. The strings may be the following values:

- *predict*
- *train*
- *eval*

TF Lite models only support prediction while TensorFlow models support both training and prediction.

This field is optional but will be required in a future version.

*type*

The *type* field is a string value that describes the class of models your model belongs to. Currently the field supports arbitrary strings with no formal hierarchy.

This field is optional.

*class*

The *class* field is a string value that contains the Objective-C class name of the custom class you would like to use with your model. It must conform to the `TIOModel` protocol and ship with your application. A custom class is not required, and Tensor/IO will use `TIOTFLiteModel` by default and assume you are using a TensorFlow Lite backend. If you are using the full TensorFlow build you must currently set the custom class name to `TIOTensorFlowModel`.

This field is optional.

<a name="inputs-field"></a>
#### The Inputs Field

The *inputs* field is an array of dictionaries that describe the input layers of your model. There must be a dictionary entry for each input layer in your model. Tensor/IO uses the information in this field to match inputs to model layers and correctly copy bytes into tensor buffers.

A basic entry in this array will have the following fields:

```json
{
  "name": "layer-name",
  "type": "array",
  "dtype": "float32",
  "shape": [224]
}
```

*name*

The *name* field is a string value that names this input tensor. It should match the name of a tensor in the underlying model and functions as a reference in application space in case you would like to pass an `NSDictionary` as input to a model's `runOn:` method.
This field is required.

*type*

The *type* field specifies the kind of data this tensor expects. Only two types are currently supported:

- *array*
- *image*

Use the *array* type for shapes of any dimension, including single values, vectors, matrices, and higher dimensional tensors. Use the *image* type for image inputs.

This field is required.

*dtype*

The *dtype* field indicates what type of data this input accepts and will correspond, for example, to a primitive C type or a TensorFlow dtype. The following data types are supported:

- *uint8*
- *float32*
- *int32*
- *int64*

Note that complete support for this field is in development and that not all backends support all datatypes. TFLite supports only uint8 and float32 data types, and this field is ignored. Quantized models automatically use uint8 types and unquantized models float32 types. The full TensorFlow backend, on the other hand, supports all four types, but if a type is not specified it defaults to float32.

This field is currently optional. The *float32* is assumed in most cases.

*shape*

The *shape* field is an array of integer values that describe the size of the input layer, ignoring whether the layer expects four byte or single byte values. Common shapes might include:

```json
// a single-valued input
"shape": [1]

// a vector with 16 values
"shape": [1,16]

// a matrix with 32 rows and 100 columns
"shape": [32,100]

// a three dimensional image volume with a width of 224px, 
// a height of 224px, and 3 channels (RGB)
"shape": [224,224,3]
```

If you are using TensorFlow models with tensors whose first dimension takes a variable batch size, use a `-1` for the first dimension of the shape:

```json
"shape": [-1,224,224,3]	
```

The shape should accurately reflect the shape of the underlying tensor, even though in many cases what matters is the total byte count. For example, a row vector with sixteen elements would have a shape of `[1,16]` while a column vector one of `[16,1]`.

This field is required.

**Unrolling Data**

Although we describe the inputs to a layer in terms of shapes with multiple dimensions, and from a mathematical perspective work with vectors, matrices, and tensors, at a machine level, neither Tensor/IO nor TensorFlow Lite has a concept of a shape.

From a tensor's perspective all shapes are represented as an unrolled vector of numeric values and packed into a contiguous region of memory, i.e. a buffer. Similary, from an Objective-C perspective, all values passed as input to a Tensor/IO model must already be unrolled into an array of data, either an array of bytes when using `NSData` or an array of `NSNumber` when using `NSArray`.

When you order your data into an array of bytes or an array of numbers in preparation for running a model on it, unroll the bytes using row major ordering. That is, traverse higher order dimensions before lower ones.

For example, a two dimensional matrix with the following values should be unrolled across columns first and then rows. That is, start with the first row, traverse every column, move to the second row, traverse every column, and so on:

```objc
[ [ 1 2 3 ]
  [ 4 5 6 ] ]
  
NSArray *matrix = @[ @(1), @(2), @(3), @(4), @(5), @(6) ]; 
```

Apply the same approach for volumes of a higher dimension, as mind-boggling as it starts to get.

**Additional Fields**

There are additional fields for handling data transformations such as quantization and pixel normalization. These will be discussed in their respective sections below.

**Both Order and Name Matter**

Input to a `TIOModel` may be organized by either index or name, so that both the order of the dictionaries in the *inputs* array and their names are significant. TF Lite tensors are accessed by index, but internally Tensor/IO associates a name with each index in case you prefer to send `NSDictionary` inputs to your models. TensorFlow models use the name exclusively, which is why names must match the names of underlying tensors.

**Example**

Here's what the *inputs* field looks like for a model with two input layers, the first a vector with 8 values and the second a 10x10 matrix:

```json
"inputs": [
  {
    "name": "vector-input",
    "type": "array",
    "shape": [8]
  },
  {
    "name": "matrix-input",
    "type": "array",
    "shape": [10,10]
  }
],
```

With this description we can pass either an array of arrays or a dictionary of arrays to the model's `runOn:` method. To pass an array, make sure the order of your inputs matches the order of their entries in the JSON file:

```objc
NSArray *vectorInput = @[ ... ]; // with 8 values
NSArray *matrixInput = @[ ... ]; // with 100 values in row major order

NSArray *arrayInputs = @[
  vectorInput,
  matrixInput
];

[model runOn:arrayInputs];
```

To pass a dictionary, simply associate the correct name with each value:

```objc
NSArray *vectorInput = @[ ... ]; // with 8 values
NSArray *matrixInput = @[ ... ]; // with 100 values in row major order

NSDictionary *dictionaryInputs = @{
  @"vector-input": vectorInput,
  @"matrix-input": matrixInput
};

[model runOn:dictionaryInputs];
```

<a name="outputs-field"></a>
#### The Outputs Field

The *outputs* field is an array of dictionaries that describe the output layers of your model. The *outputs* field is structured the same way as the *inputs* field, and the dictionaries contain the same basic entries as those in the *inputs* field:

```json
"outputs": [
  {
    "name": "vector-output",
    "type": "array",
    "dtype": "float32",
    "shape": [8]
  }
]
```

**The Labels Field**

An *array* type output optionally supports the presence of a *labels* field for classification outputs:

```json
"outputs": [
  {
    "name": "classification-output",
    "type": "array",
    "shape": [1000],
    "labels": "labels.txt"
  }
]
```

The value of this field is a string which corresponds to the name of a text file in the bundle's *assets* directory.  The *.tiobundle* directory structure for this model might look like:

```
mymodel.tiobundle
  - model.json
  - model.tflite
  - assets
    - labels.txt
```

Each line of the *labels.txt* text file contains the name of the classification for that line number index in the layer's output. When a *labels* field is present, Tensor/IO internally maps labels to their numeric outputs and returns an `NSDictionary` representation of that mapping, rather than a simple `NSArray` of values. Let's see what that looks like.

**Model Outputs**

Normally, a model returns a dictionary of array values from its `runOn:` method, and those values will usually be arrays. Each layer produces its own entry in that dictionary, corresponding to the name of the layer in its JSON description. 

For example, a self-driving car model might classify three kinds of things in an image (well, hopefully more than that!). The *outputs* field for this model might look like:

```json
"outputs": [
  {
    "name": "classification-output",
    "type": "array",
    "shape": [3],
  }
]
```

After performing inference the underlying TensorFlow model will produce an output with three values corresponding to the softmax probability that this item appears in the image. Tensor/IO extracts those bytes and packs them into an `NSArray` of `NSNumber`:

```objc
NSDictionary *inference = (NSDictionary *)[model runOn:input];
NSArray<NSNumber*> *classifications = inference[@"classification-output"];

// classifications[0] == 0.25
// classifications[1] == 0.75
// classifications[2] == 0.25
```

However, when a *labels* entry is present for a layer, the entry for that layer will itself be a dictionary mapping names to values.

Our self-driving car model might for example add a *labels* field to the above description:

```json
"outputs": [
  {
    "name": "classification-output",
    "type": "array",
    "shape": [3],
    "labels": "labels.txt"
  }
]
```

With a *labels.txt* file in the bundle's *assets* directory that looks like:

```txt
pedestrian
car
motorcycle
```

The underlying tensorflow model still produces an output with three values corresponding to the softmax probability that this item appears in the image. Tensor/IO, however, now maps labels to those probabilities and returns a dictionary of those mappings:

```objc
NSDictionary *inference = (NSDictionary *)[model runOn:input];
NSDictionary<NSString*, NSNumber*> *classifications = inference[@"classification-output"];

// classifications[@"pedestrian"] == 0.25
// classifications[@"car"] == 0.75
// classifications[@"motorcycle"] == 0.25
```

**Single Valued Outputs**

In some cases your model might output a single value in one of its output layers. Consider the housing price model we discussed earlier. When that is the case, instead of wrapping that single value in an array and returning an array for that layer, Tensor/IO will simply output a single value for it.

Consider a model with two output layers. The first layer outputs a vector of four values while the second outputs a single value:

```json
"outputs": [
  {
    "name": "vector-output",
    "type": "array",
    "shape": [4]
  },
  {
    "name": "scalar-output",
    "type": "array",
    "shape": [1]
  }
]
```

After performing inference, access the first layer as an array of numbers and the second layer as a single number:

```objc
NSDictionary *inference = (NSDictionary *)[model runOn:input];
NSArray<NSNumber*> *vectorOutput = inference[@"vector-output"];
NSNumber *scalarOutput = inference[@"scalar-output"];
```

*Scalar outputs are supported as a convenience. Model outputs may change in a later version of this library and so this convenience may be removed or modified.*

<a name="options-field"></a>
#### The Options Field

You may optionally included an *options* field in the JSON description. It contains properties that are not required by Tensor/IO to perform inference but which are used in application specific ways. Tensor/IO will ignore these properties but you may inspect them from application space to change your product's behavior when a particular model is running.

Two options are currently supported: *device\_position* and *output\_format*:

**Device Position**

The *device\_position* option target computer vision models specifically and tells a consumer of the model which device camera it should begin with for this model. For example, some models target facial features and would prefer that the model run on the front facing camera initially, while others target features of the world and would prefer to run on the back facing camera. Valid entries for this field include:

- *front*
- *back*

**Output Format**

The value of the *output\_format* field is an arbitrary, application specific string providing a hint to consumers of the model about how they should format the model's output. Provide any arbitrary string. 

For example, Net Runner knows how to interpret *"image.classification.nodecay"*. When it sees this output format identifier, it will inspect a model's output, expecting a single "classification" output of an array of values, and format the probability values to two decimal places without applying any exponential decay to them.

<a name="complete-example"></a>
#### A Complete Example

Let's see a complete example of a model with two input layers and two output layers. The model takes two vectors, the first with 4 values and the second with 8 values, and outputs two vectors, the first with 3 values and the second with 6.

Our *tiobundle* folder will have the following contents:

```
mymodel.tiobundle
  - model.json
  - model.tflite
```

The *model.json* file might look something like:

```json
{
  "name": "Example Model",
  "details": "This model takes two vector valued inputs and produces two vector valued outputs",
  "id": "my-awesome-model",
  "version": "1",
  "author": "doc.ai",
  "license": "Apache 2",
  "model": {
    "file": "model.tflite",
    "quantized": false
  },
  "inputs": [
    {
      "name": "foo-features",
      "type": "array",
      "shape": [4]
    },
    {
      "name": "bar-features",
      "type": "array",
      "shape": [8]
    }
  ],
  "outputs": [
    {
      "name": "baz-outputs",
      "type": "array",
      "shape": [3]
    },
    {
      "name": "qux-outputs",
      "type": "array",
      "shape": [6]
    }
  ],
  "options": {
    "device_position": "back",
    "output_format": "image.classification.nodecay"
  }
}
```

And we can perform inference with this model as follows:

```objc
NSArray *fooFeatures = @[ @(1), @(2), @(3), @(4) ]; 
NSArray *barFeatures = @[ @(1), @(2), @(3), (@4), @(5), @(6), @(7), @(8) ]; 

NSDictionary *features = @{
  @"foo-features": fooFeatures,
  @"bar-features": barFeatures
};

NSDictionary *inference = (NSDictionary *)[model runOn:features];

NSArray *bazOutputs = inference[@"baz-outputs"]; // length 3
NSArray *quxOutputs = inference[@"qux-outputs"]; // length 6
```


<< [Model Bundles](ModelBundles.md) || [TOC](TOC.md) || [Quantization](Quantization.md) >>