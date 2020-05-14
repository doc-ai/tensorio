<a name="basic-usage"></a>
## Basic Usage

A Tensor/IO model takes a set of inputs, performs inference, and returns a set of outputs.

Consider a model that predicts the price of a house given a feacture vector that includes square footage, number of bedrooms, number of bathrooms, proximity to a school, and so forth.

With Tensor/IO you construct an `NSArray` of numeric values for these features, pass the array to your model, and extract the price from the results.

```objc
TIOTFLiteModel *model = ...
NSArray *input = @[ @(1890), @(3), @(2), @(1.6) ];
NSDictionary *output = (NSDictionary *)[model runOn:input];
NSNumber *price = output[@"price"];
```

**TIOData**

Tensor/IO models take inputs and produce outputs of type `TIOData`. This is a generic protocol that simply marks native data types as available to Tensor/IO models. A backend that supports a specific underlying machine learning library extends this protocol and implements methods that copy data into and out of tensors.

Tensor/IO backends such as those for TensorFlow and TFLite will always include implementations of this protocol for the following classes:

- NSNumber
- NSData
- NSArray
- NSDictionary
- CVPixelBufferRef with TIOPixelBuffer

In the above example, we're passing a single `NSArray` to the model. The model extracts numeric byte values from the array, copying them into the underlying TF Lite model. It asks the underlying model to perform inference, and then copies the resulting bytes back into an `NSNumber`. That `NSNumber` is added to a dictionary under the `@"price"` key, and it is this dictionary which the model returns.

**Model Outputs**

Why is the resulting price not returned directly, and how do we know that the value is keyed to `@"price"` in the returned dictionary?

Because models may have multiple inputs and outputs, Tensor/IO tries to make no assumptions about how many input and output layers a model actually has. This gives it some flexiblity in what kinds of inputs it can take, for example a single numeric value, arrays of numeric arrays, or a dictionary, and it intelligently matches those inputs to the underlying tensor buffers, but a model consequently always returns a dictionary of outputs. 

(*Note: this may change in a future implementation, and single outputs may be returned directly*)

To understand why the output value is keyed to a specific entry, we must understand how Tensor/IO is able to match Objective-C inputs and outputs to the underlying model's input and output layers, and for that we require an understanding of model bundles and the JSON file which describes the underlying model.


<< [License](License.md) || [TOC](TOC.md) || [Model Bundles](ModelBundles.md) >>