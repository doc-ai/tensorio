### Introduction

TensorIO is a lightweight, cross-platfor framework for on-device machine learning, bringing the power of TensorFlow Lite to iOS, Android, and React Native applications. 

TensorIO provides a declarative input-output interface to an underlying machine learning model while also providing a structure for packaging models, metadata about them, and related assets into a single, deliverable unit. TensorIO makes it easy for for resarchers and application developers to use their models on mobile phones and it simplifies the process of deploying models to mobile devices.

TensorIO implements no machine learning itself. Rather it relies on the power of an underlying machine learning library while taking care of the boilerplate code that is needed to get data into and out of a model. Such preparation includes quantization of inputs, image croping and scaling, normalization, byte order operations, and other transformations. Library adopters describe a model's expectations for inputs and outputs using json and then simply send native data types into a model and receive native data types back from it. TensorIO ensures the data is correctly pre- and postprocessed and that bytes are moved into and out of tensor buffers correctly.

TensorIO is currently available for iOS and React Native with Android support expected by the end of Feb 2019. TensorIO supports TensorFlow Lite models and is extensible to other machine learning libraries.

### License

All TensorIO, Net Runner, and related code is open source under an Apache 2 license. Copyright [doc.ai](https://doc.ai), 2018-present.

### Core Contributors

- [Philip Dow](https://github.com/phildow)
- [Neeraj Kashyap](https://github.com/nkashy1)
- [Sam Leroux](https://github.com/SamLeroux)
- [Aria Vaghef](https://github.com/aria-doc-ai)

### Example Usage

Given a TensorFlow Lite MobileNet ImageNet classification model that has been packaged into a TensorIO bundle, the model.json file will look like:

```json
{
  "name": "MobileNet V2 1.0 224",
  "details": "MobileNet V2 with a width multiplier of 1.0 and an input resolution of 224x224. \n\nMobileNets are based on a streamlined architecture that have depth-wise separable convolutions to build light weight deep neural networks. Trained on ImageNet with categories such as trees, animals, food, vehicles, person etc. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.",
  "id": "mobilenet-v2-100-224-unquantized",
  "version": "1",
  "author": "Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam",
  "license": "Apache License. Version 2.0 http://www.apache.org/licenses/LICENSE-2.0",
  "model": {
    "file": "mobilenet_v2_1.4_224.tflite",
    "quantized": false,
    "type": "image.classification.imagenet"
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
    }
  ],
  "outputs": [
    {
      "name": "classification",
      "type": "array",
      "shape": [1,1000],
      "labels": "labels.txt"
    }
  ]
}

```

This model may be used in iOS as follows:

```objc
UIImage *image = [UIImage imageNamed:@"example-image"];
TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];

TIOTFLiteModel *model = [TIOTFLiteModel modelWithBundleAtPath:path];

NSDictionary *inference = (NSDictionary*)[model runOn:buffer];
NSDictionary *classification = [inference[@"classification"] topN:5 threshold:0.1];
```

And in React Native:

```js
RNTensorIO.load('image-classification');

RNTensorIO.run({
  'image': {
    [RNTensorIO.imageKeyData]: '/path/to/image.jpeg',
    [RNTensorIO.imageKeyFormat]: RNTensorIO.imageTypeFile
  }
}, (error, results) =>  {
  classifications = results['classification'];
  
  RNTensorIO.topN(5, 0.1, classifications, (error, top5) => {
    console.log("TOP 5", top5);
  });
});
```

### iOS

[TensorIO for iOS](https://github.com/doc-ai/tensorio-ios)

Our Objective-C++ implementation of TensorIO. Requires iOS 9.3+ and has been tested on devices as old as a 5th generation iPod touch (2012).

[Net Runner for iOS](https://github.com/doc-ai/net-runner-ios)

Net Runner is our iOS application enviornment for running and evaluating computer vision machine learning models packaged for TensorIO. Models may be run on live camera input or bulk evaluated against album photos. New models may be downloaded directly into the application. Net Runner is available for download in the [iOS App Store](https://itunes.apple.com/us/app/net-runner-by-doc-ai/id1435828634?mt=8).

### Android

TensorIO for Android is in active development and is expected to be made publicly available by the end of Feb 2019. It will provide the same declarative interface for Android devices with support for TensorFlow Lite models. A version of Net Runner for Android will be released either simultaneously or shortly thereafter.

<!--
[TensorIO for Android](https://github.com/doc-ai/tensorio-android)

[Net Runner for Android](https://github.com/doc-ai/net-runner-android)
-->

### React Native

[TensorIO for React Native](https://github.com/doc-ai/react-native-tensorio)

Our React Native bindings for TensorIO, with full support for the iOS version. React Native bindings for Android are expected by the end of March 2019.

[TensorIO Demo App for React Native](https://github.com/doc-ai/react-native-tensorio-example)

An example application demonstrating how to use the TensorIO module in a React Native application.

### Tools

[TensorIO Bundler](https://github.com/doc-ai/tensorio-bundler)

Our bundling utility for packing models into the TensorIO format.