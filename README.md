### Introduction

Tensor/IO is a lightweight, cross-platform library for on-device machine learning, bringing the power of TensorFlow and TensorFlow Lite to iOS, Android, and React Native applications. Tensor/IO does not implement any machine learning itself but works with an underlying library such as TensorFlow to simplify the process of deploying and using models on mobile phones.

#### Tensor/IO is Declarative

Tensor/IO is above all a declarative interface to your model. This means that you describe the input and output layers to your model using a plain-text language and Tensor/IO takes care of the transformations needed to prepare inputs for the model and to read outputs back out of it, allowing you to focus on what you know instead of a low-level C++ interface.

#### On-Device

Tensor/IO runs on iOS and Android mobile phones, with bridging for React Native, and it runs the same underlying model on every OS without needing to convert models to CoreML or MLKit. You'll choose a specific backend for your use case, such as TensorFlow or TensorFlow Lite, and the library takes care of interacting with it in the language of your choice: Objective-C, Swift, Java, Kotlin, or JavaScript.

#### Inference

Prediction with Tensor/IO can often be done with as little as five lines of code. The TensorFlow Lite backend supports deep networks and a range of convolutional models, and the full TensorFlow backend supports almost any network you can build in python. Performance is impressive. MobileNet models execute inference on the iPhone X in ~30ms and can be run in real-time.

#### Training

With support for the full TensorFlow backend you can train models on device and then export their updated weights, which can be immediately used in a prediction model. All without ever leaving the phone. Use the same declarative interface to specify your training inputs and outputs, evaluation metric, and training operation, and to inject placeholder values into your model for on-device hyperparameter tuning.

#### Deployment

Our packaging makes it easy to ship models with your application, but the plain text model interfaces and client-server code also allow you to retrieve and update models in deployed applications. No need to write new code or recompile the app. Our Tensor/IO-Models repo listed below includes the serverside application to support in-production deployments.

#### Federated Learning

Building on support for training and in-production deployment, we have also implemented a federated learning system. Clients check for federated training tasks at a central location and then download and train models locally. Local updates are returned to backend infrastructure and aggregated to create a new global model, which is sent back to local devices for prediction and evaluation. The federated approach helps us to fulfill one of our core missions: to preserve the privacy of data.

### Example Usage

Given a TensorFlow Lite MobileNet ImageNet classification model that has been packaged into a Tensor/IO bundle ([bundled here](https://github.com/doc-ai/tensorio/tree/master/models/image-classification.tiobundle)), the model.json looks like:

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
    "backend": "tflite",
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

Use the model on iOS:

```objc
UIImage *image = [UIImage imageNamed:@"example-image"];
TIOPixelBuffer *buffer = [[TIOPixelBuffer alloc] initWithPixelBuffer:image.pixelBuffer orientation:kCGImagePropertyOrientationUp];

TIOTFLiteModel *model = [TIOTFLiteModel modelWithBundleAtPath:path];

NSDictionary *inference = (NSDictionary*)[model runOn:buffer];
NSDictionary *classification = [inference[@"classification"] topN:5 threshold:0.1];
```

Or with Swift:

```swift
let image = UIImage(named: "example-image")!
let pixels = image.pixelBuffer()!
let value = pixels.takeUnretainedValue() as CVPixelBuffer
let buffer = TIOPixelBuffer(pixelBuffer:value, orientation: .up)

let model = TIOTFLiteModel.withBundleAtPath(path)!

let inference = model.run(on: buffer)
let classification = ((inference as! NSDictionary)["classification"] as! NSDictionary).topN(5, threshold: 0.1)
```

In Java on Android:

```java
InputStream bitmap = getAssets().open("picture2.jpg");
Bitmap bMap = BitmapFactory.decodeStream(bitmap);

TIOModelBundleManager manager = new TIOModelBundleManager(getApplicationContext(), path);
TIOModelBundle bundle = manager.bundleWithId(bundleId);
TIOModel model = bundle.newModel();
model.load();

float[] result =  (float[]) model.runOn(bMap);
String[] labels = ((TIOVectorLayerDescription)model.descriptionOfOutputAtIndex(0)).getLabels();
```

Or Kotlin:

```kotlin
val bitmap = assets.open("picture2.jpg")
val bMap = BitmapFactory.decodeStream(bitmap)

val manager = TIOModelBundleManager(applicationContext, path)
val bundle = manager.bundleWithId(bundleId)
val model = bundle.newModel()
model.load()

val result = model.runOn(bMap) as FloatArray
```

And in React Native:

```js
RNTensor/IO.load('image-classification');

RNTensor/IO.run({
  'image': {
    [RNTensor/IO.imageKeyData]: '/path/to/image.jpeg',
    [RNTensor/IO.imageKeyFormat]: RNTensor/IO.imageTypeFile
  }
}, (error, results) =>  {
  classifications = results['classification'];
  
  RNTensor/IO.topN(5, 0.1, classifications, (error, top5) => {
    console.log("TOP 5", top5);
  });
});
```

### License and Open Source

All Tensor/IO, Net Runner, and related code is open source under an Apache 2 license. Copyright [doc.ai](https://doc.ai), 2018-present.

#### iOS

[Tensor/IO for iOS](https://github.com/doc-ai/tensorio-ios)

Our Objective-C++ implementation of Tensor/IO, with support for Swift. Requires iOS 9.3+ and has been tested on devices as old as a 5th generation iPod touch (2012).

[Net Runner for iOS](https://github.com/doc-ai/net-runner-ios)

Net Runner is our iOS application environment for running and evaluating computer vision machine learning models packaged for Tensor/IO. Models may be run on live camera input or bulk evaluated against album photos. New models may be downloaded directly into the application. Net Runner is available for download in the [iOS App Store](https://itunes.apple.com/us/app/net-runner-by-doc-ai/id1435828634?mt=8).

#### Android

[Tensor/IO for Android](https://github.com/doc-ai/tensorio-android)

Our Java implementation of Tensor/IO for Android. Requires Java 8.

[Net Runner for Android](https://github.com/doc-ai/net-runner-android)

Net Runner is our Android application environment for running computer vision models on device, including support for GPU acceleration.

#### React Native

[Tensor/IO for React Native](https://github.com/doc-ai/react-native-Tensor/IO)

Our React Native bindings for Tensor/IO, with full support for the iOS version. React Native bindings for Android are forthcoming.

[Tensor/IO Demo App for React Native](https://github.com/doc-ai/react-native-Tensor/IO-example)

An example application demonstrating how to use the Tensor/IO module in a React Native application, with a MobileNet ImageNet classification model.

#### Deployment and Federated Learning

[Tensor/IO Models](https://github.com/doc-ai/tensorio-models)

Our backend service for deploying models to device. Client APIs in Tensor/IO communicate with a Tensor/IO Models server to browse, retrieve, and update models.

[Tensor/IO Flea](https://github.com/doc-ai/tensorio-models)

Flea (Federated LEArning) is our backend service for conducting federated training rounds. Client APIs in Tensor/IO communicate with a Tensor/IO Flea server to retrieve federated learning tasks and return updated model weights for later aggregation into a new prediction model.

#### Tools

[Tensor/IO Bundler](https://github.com/doc-ai/tensorio-bundler)

Our bundling utility for packaging models into the Tensor/IO format. Includes a Slack bot, Bundlebot, that helps convert checkpointed models to the Tensor/IO format and deploy them to device in under a minute.

#### Additional Repositories

[TensorFlow @ doc.ai](https://github.com/doc-ai/tensorflow)

Our TensorFlow fork with fixes and additional ops enabled to support both training and inference on iOS. See specifically the [v1.13.0-rc2-ios-fixes](https://github.com/doc-ai/tensorflow/tree/v1.13.0-rc2-ios-fixes) branch and our [build script](https://github.com/doc-ai/tensorflow/blob/v1.13.0-rc2-ios-fixes/tensorflow/contrib/makefile/create_full_ios_frameworks.sh) for composing the framework.

[Tensorflow.framework](https://github.com/doc-ai/tensorflow-ios-framework)

Unofficial build of the full tensorflow.framework (not TFLite) for iOS that we will be using in the Tensor/IO/TensorFlow subspec.

[TensorFlow Cocoapod](https://github.com/doc-ai/tensorio-tensorflow-ios)

Unofficial full build of TensorFlow in a self-contained CocoaPod that we will be using in the Tensor/IO/TensorFlow subspec. Vends the tensorflow, protobuf, and nysnc static libraries and all headers.

### Core Contributors

- [Philip Dow](https://github.com/phildow)
- [Neeraj Kashyap](https://github.com/nkashy1)
- [Sam Leroux](https://github.com/SamLeroux)
- [Aria Vaghef](https://github.com/aria-doc-ai)