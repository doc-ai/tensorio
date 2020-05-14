<a name="images"></a>
## Working with Image Data

#### Contents

* [ Introduction ](#introduction)
* [ A Basic Example ](#images-basic-example)
* [ Pixel Buffers ](#pixel-buffer)
* [ The Format Field ](#pixel-format)
* [ Pixel Normalization ](#pixel-normalization)
* [ Pixel Denormalization ](#pixel-normalization)
* [ A Complete Example ](#pixel-buffer-complete-example)

<a name="introduction"></a>
#### Introduction

Tensor/IO has built-in support for  image data and can perform inference on image data as well as return image data as an output. A key concept when working with image data is the *pixel buffer*, which is a pixel by pixel representation of an image in memory. 

Tensor/IO works with pixel buffers and includes a wrapper for the native `CVPixelBufferRef`. It also provides utility functions for converting instances of `UIImage` to and from pixel buffers.

<a name="images-basic-example"></a>
#### A Basic Example

As always, inform Tensor/IO that an input layer expects pixel buffer data by modifying that layer's description in *model.json*. Set its *type* to *image*. You must specify the *shape* as an array of *[height, width, channels]* and the *format* of the image as either *RGB* or *BGR*. More on image formats below. 

For now let's assume the tensor takes image volumes of size 224x224x3 with RGB byte ordering:

```json
"inputs": [
  {
    "name": "image-input",
    "type": "image",
    "shape": [224,224,3],
    "format": "RGB"
  }
]
```

We can then pass image data to this model by wrapping an image's pixel buffer in a `TIOPixelBuffer`, which knows how to copy pixel data to the tensor given the format:

```objc
UIImage *image = [UIImage imageNamed:@"example-image"];
CVPixelBufferRef pixelBuffer = image.pixelBuffer;
TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:pixelBuffer orientation:kCGImagePropertyOrientationUp];

NSDictionary *inference = (NSDictionary *)[model runOn:buffer];
```

<a name="pixel-buffer"></a>
#### Pixel Buffers

A pixel buffer is a pixel by pixel representation of image data laid out in a contiguous block of memory. On iOS some APIs provide raw pixel buffers by default, such the AVFoundation APIs, while in other cases we must construct pixel buffers ourselves.

A pixel buffer always has a size, which includes the width and height, as well as a format, such as ARGB or BGRA, which lets the buffer know how many *channels* of data there are for each pixel and in what order those bytes appear. In the case of ARGB and BGRA, there are four channels of data arranged in alpha-red-green-blue or blue-green-red-alpha order respectively.

The ARGB and BGRA pixel buffers on iOS represent each pixel using four bytes of memory, with a single byte allocated to each channel. Each color in the pixel is represented by a range of values from 0 to 255, and the alpha channel also, allowing a pixel to represent over 16 million colors with 256 alpha values.

<a name="pixel-format"></a>
#### The Format Field

Tensors operate on pixel buffers with specific byte orderings. Imagine the memory for a pixel buffer in ARGB format. The top left pixel at (0,0) will appear first, then the pixel to its right at (1,0), and to its right at (2,0) and so on, for each column and each row in the image, with the bytes appearing in alpha-red-green-blue order:

```
[ARGB][ARGB][ARGB][ARGB][ARGB]...
```

Now imagine what that same image looks like to the tensor in BGRA format:

```
[BGRA][BGRA][BGRA][BGRA][BGRA]...
```

The byte ordering, which is to say, the format of the pixel buffer, definitely matters! 

You must let Tensor/IO know what byte ordering an input layer expects via the *format* field. Consequently you must know what byte ordering your model expects.

Tensor/IO supports two byte orderings, *RGB* and *BGR*. Models ignore the alpha channel and don't expect it to be present, so Tensor/IO internally skips it when copying ARGB or BGRA pixel buffer bytes into tensors.

```json
{
  "format": "RGB"
}

{
  "format": "BGR"
}
```

<a name="pixel-normalization"></a>
#### Pixel Normalization

Notice that pixels are represented using a single byte of data for each color channel, a `uint8_t`. Recall what we know about quantized models. By default, TF Lite works with four byte floating point representations of data, `float_t`, but when the model is quantized it uses single byte `uint8_t` representations of data. 

Hm. It looks like pixel buffer data is already "quantized"! 

In fact, when working with quantized models, you may pass pixel buffer data directly to input layers and read it directly from output layers without needing to transform the data (other than skipping the alpha channel). Quantized models already work on values in a range from 0 to 255, and pixel buffer data is exactly in this range.

Models that are not quantized, however, expect pixel buffer data in a floating point representation, and they will typically want it in a *normalized* range of values, usually from 0 to 1 or from -1 to 1. The process of converting pixel values from a single byte representation to a floating point representation is called *normalization*, and Tensor/IO includes built-in support for it.

**The Normalize Field**

As always, you will need to update the description of an input layer to indicate what kind of normalization you want. Include the *normalize* field in the layer's entry. Like the *quantize* field it takes either two entries or a single entry: either *scale* and *bias*, or a *standard* field, with the difference that bias may be applied on a per channel basis.

*scale*

The *scale* field is a numeric value that specifies the scaling factor to apply to incoming pixel data.

*bias*

The *bias* field is a dictionary value that specifies the bias to apply to incoming pixel data, on a *per channel* basis, and itself includes three entries, *r*, *g*, and *b*.

Together, a *scale* and *bias* entry might look like:

```json
"normalize": {
  "scale": 0.004,
  "bias": {
    "r": -0.485,
    "g": -0.457,
    "b": -0.408
  }
}
```

And together, Tensor/IO applies the following equation to any pixel data sent to this layer:

```
normalized_red_value   = scale * red_value   + red_bias
normalized_green_value = scale * green_value + green_bias
normalized_blue_value  = scale * blue_value  + blue_bias
``` 

*standard*

The *standard* field is a string value corresponding to one of a number of commonly used normalizations. Its presence overrides the *scale* and *bias* fields.

Tensor/IO currently supports two standard normalizations. The ranges tell Tensor/IO *what values you are normalizing to*:

```json
"normalize": {
  "standard": "[0,1]"
}

"normalize": {
  "standard": "[-1,1]"
}
```

<a name="pixel-normalization"></a>
#### Pixel Denormalization

Tensor/IO can also read pixel data from output tensors and reconstruct pixel buffers from them. When reading pixel data from an unquantized model it will usually be necessary to convert the values from a normalized floating point representation back to `uint8_t` values in the range of 0 to 255. This process is called *denormalization*, and once again Tensor/IO has built in support for it.

To denormalize pixel data add a *denormalize* field to an output layer's description. Like the *normalize* field this field takes either *scale* and *bias* entries or a *standard* entry. The fields work as they do for normalization but as their inverses.

For bias and scale, the following equation will be applied:

```
red_value   = (normalized_red_value   + red_bias)   * scale
green_value = (normalized_green_value + green_bias) * scale
blue_value  = (normalized_blue_value  + blue_bias)  * scale
```

Similarly, Tensor/IO supports two standard denormalizations. The ranges tell Tensor/IO *what values you are denormalizing from*:

```json
"denormalize": {
  "standard": "[0,1]"
}

"denormalize": {
  "standard": "[-1,1]"
}
```

<a name="pixel-buffer-complete-example"></a>
#### A Complete Example

Let's look at a complete example. This is the unquantized MobileNetV2 image classification model provided by TensorFlow. It takes a single input, image data of size 224x224x3 in RGB format, and produces a single output, a vector of 1000 softmax probabilities identifying the object in the image. It expects image data to be normalized to a range from -1 to 1, and we would like to label the output data.

The model bundle folder might look something like:

```
mobilenet-model.tiobundle
  - model.json
  - model.tflite
  - assets
    - labels.txt
```

The *model.json* file might look like:

```json
{
  "name": "MobileNet V2 1.0 224",
  "details": "MobileNet V2 with a width multiplier of 1.0 and an input resolution of 224x224. \n\nMobileNets are based on a streamlined architecture that have depth-wise separable convolutions to build light weight deep neural networks. Trained on ImageNet with categories such as trees, animals, food, vehicles, person etc. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.",
  "id": "mobilenet-v2-100-224-unquantized",
  "version": "1",
  "author": "Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam",
  "license": "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0",
  "model": {
    "file": "model.tflite",
    "backend": "tflite",
    "quantized": false,
    "modes": ["predict"]
  },
  "inputs": [
    {
      "name": "image",
      "type": "image",
      "shape": [224,224,3],
      "format": "RGB",
      "normalize": {
        "standard": "[-1,1]"
      }
    },
  ],
  "outputs": [
    {
      "name": "classification",
      "type": "array",
      "shape": [1,1000],
      "labels": "labels.txt"
    },
  ]
}
```

And we can use this model as follows:

```objc
UIImage *image = [UIImage imageNamed:@"example-image"];
TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];

NSDictionary *inference = (NSDictionary *)[model runOn:buffer];
NSDictionary<NSString*,NSNumber*> *classification = inference[@"classification"];
```

Find more examples of image models at [tensorio/examples](https://github.com/doc-ai/tensorio/tree/master/examples).

<< [Quantization](Quantization.md) || [TOC](TOC.md) || [Training](Training.md) >>