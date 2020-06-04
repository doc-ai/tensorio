//
//  TF_2_2_KerasMNISTFashionTests.swift
//  TensorIOExamplesTests
//
//  Created by Phil Dow on 6/1/20.
//  Copyright © 2020 doc.ai. All rights reserved.
//

import XCTest
import TensorIO

class TF_2_2_KerasMNISTFashionTests: XCTestCase {

    let modelsURL = Bundle(for: TF_2_0_KerasMNISTFashionTests.self).url(forResource: "models", withExtension: nil)!
    
    func bundle(named name: String, backend: String) -> TIOModelBundle? {
        let path = self.modelsURL.appendingPathComponent(backend).appendingPathComponent(name).path
        return TIOModelBundle(path: path)
    }
    
    func model(with bundle: TIOModelBundle) -> TIOModel? {
        guard let model = bundle.newModel() else {
            print("There was a problem instantiating the model from the bundle")
            return nil
        }
        
        guard let _ = try? model.load() else {
            print("There was a problem loading the model")
            return nil
        }
        
        return model
    }
    
    func image(named name: String) -> UIImage {
        let filename = URL(string: name)!.deletingPathExtension().path
        let ext = URL(string: name)!.pathExtension
        
        let bundle = Bundle(for: type(of: self))
        let path = bundle.path(forResource: filename, ofType: ext)!
        let image = UIImage(contentsOfFile: path)!
        
        return image
    }

    // MARK: -

    override func setUp() { }

    override func tearDown() {}

    // MARK: -

    func testModel_SaveModel() {
        
        // Prepare the image
        
        let buffer = (0..<784).map { $0 } as NSArray
        
        // Load the model
        
        let bundle = self.bundle(named: "keras-mnist-fashion-save-model.tiobundle", backend: "tf_2_2")!
        let model = self.model(with: bundle)!
        
        // Predict
        
        var error: NSError?
        let output = model.run(on: buffer, error: &error)
        
        XCTAssertNotNil(output)
        XCTAssertNil(error)
        
        let logits = (output as! NSDictionary)["StatefulPartitionedCall"] as? [Float]
        
        XCTAssertNotNil(logits)
    }
    
    func testModel_ModelBuilder() {
        
        // Prepare the image
        
        let buffer = (0..<784).map { $0 } as NSArray
        
        // Load the model
        
        let bundle = self.bundle(named: "keras-mnist-fashion-model-builder.tiobundle", backend: "tf_2_2")!
        let model = self.model(with: bundle)!
        
        // Predict
        
        var error: NSError?
        let output = model.run(on: buffer, error: &error)
        
        XCTAssertNotNil(output)
        XCTAssertNil(error)
        
        let logits = (output as! NSDictionary)["dense_1/BiasAdd"] as? [Float]
        
        XCTAssertNotNil(logits)
    }
    
    func testModel_EstimatorPredict() {
        
        // Prepare the image
        
        let buffer = (0..<784).map { $0 } as NSArray
        
        // Load the model
        
        let bundle = self.bundle(named: "keras-mnist-fashion-estimator-predict.tiobundle", backend: "tf_2_2")!
        let model = self.model(with: bundle)!
        
        // Predict
        
        var error: NSError?
        let output = model.run(on: buffer, error: &error)
        
        XCTAssertNotNil(output)
        XCTAssertNil(error)
        
        let logits = (output as! NSDictionary)["sequential/dense_1/BiasAdd"] as? [Float]
        
        XCTAssertNotNil(logits)
    }
    
    func testModel_EstimatorTrain() {
        
        // Prepare the image and label
        
        let buffer = (0..<784).map { $0 } as NSArray
        let label = 0 as NSNumber
        
        // Prepare the training batch
        
        let batch = TIOBatch(keys: ["image", "label"])
        
        batch.addItem([
            "image": buffer,
            "label": label
        ])
        
        // Load the model
        
        let bundle = self.bundle(named: "keras-mnist-fashion-estimator-train.tiobundle", backend: "tf_2_2")!
        let model = self.model(with: bundle) as! TIOTrainableModel
        
        // Train
        
        for /*epoch*/ _ in 0..<4 {
            var error: NSError?
            let output = model.train(batch, error: &error)
            
            XCTAssertNotNil(output)
            XCTAssertNil(error)
            
            let loss = (output as! NSDictionary)["sparse_categorical_crossentropy/weighted_loss/value"] as? Float
            
            XCTAssertNotNil(loss)
        }
    }

}
