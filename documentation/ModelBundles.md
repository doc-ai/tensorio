<a name="model-bundles"></a>
## Model Bundles

Tensor/IO currently includes support for TensorFlow Lite (TF Lite) models. Although the library is built with support for other machine learning frameworks in mind, we'll focus on TF Lite models here.

A TF Lite model is contained in a single *.tflite* file. All the operations and weights required to perform inference with a model are included in this file.

However, a model may have other assets that are required to interpret the resulting inference. For example, an ImageNet image classification model will output 1000 values corresponding to the softmax probability that a particular object has been recognized in an image. The model doesn't match probabilities to their labels, for example "rocking chair" or "lakeside", it only outputs numeric values. It is left to us to associate the numeric values with their labels.

Rather than requiring a developer to do this in application space and consequently store the lables in a text file or in some code somewhere in the application, Tensor/IO wraps models in a bundle and allows model builders to include additional assets in that bundle.

A Tensor/IO bundle is just a folder with an extension that identifies it as such: *.tiobundle*. Assets may be included in this bundle and then referenced from model specific code. 

*When you use your own models with Tensor/IO, make sure to put them in a folder with the .tiobundle extension.*

A Tensor/IO TF Lite bundle has the following directory structure:

```
mymodel.tiobundle
  - model.tflite
  - model.json
  - assets
    - file.txt
    - ...
```

The *model.json* file is required. It describes the interface to your model and includes other metadata about it. More on that below.

The *model.tflite* file is required but may have another name. The bundle must include some *.tflite* file, but its actual name is specified in *model.json*.

The *assets* directory is optional and contains any additional assets required by your specific use case. Those assets may be referenced from *model.json*.

Because image classification is such a common task, Tensor/IO includes built-in support for it, and no additional code is required. You'll simply need to specify a labels file in the model's JSON description, which we'll look at in a moment.

### Using Model Bundles

Tensor/IO encapsulate information about a model in `TIOModelBundle` . This class parses the metadata for a model from the *model.json* file and manage access to files in the *assets* directory.

You may load a bundle from a known path:

```objc
NSString *path = @"...";
TIOModelBundle *bundle = [[TIOModelBundle alloc] initWithPath:path];
```

Model bundles are also used to instantiate model instances with the `newModel` method, effectively functioning as model factories. Each call to this method produces a new model instance:

```objc
id<TIOModel> model = [bundle newModel];
```

Classes that conform to the `TIOModel` protocol also implement a convenience method for instantiating models directly from a model bundle path:

```objc
NSString *path = @"...";
TIOTFLiteModel *model = [TIOTFLiteModel modelWithBundleAtPath:path];
```

### Packaging Model Bundles

#### TF Lite

For more information about packaging TF Lite models into model bundles, see additional instructions for [TF Lite Model Packaging](https://github.com/doc-ai/tensorio-ios/wiki/TensorFlow-Lite-Backend).

#### TensorFlow

For more information about packaging TensorFlow models into model bundles, see additional instructions for [TensorFlow Model Packaging](https://github.com/doc-ai/tensorio-ios/wiki/TensorFlow-Backend).

And as always you can refer to our [example models](https://github.com/doc-ai/tensorio/tree/master/examples).

<< [Basic Usage](BasicUsage.md) || [TOC](TOC.md) || [Model JSON](ModelJSON.md) >>