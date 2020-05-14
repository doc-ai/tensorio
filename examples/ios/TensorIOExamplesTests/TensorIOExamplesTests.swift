//
//  TensorIOExamplesTests.swift
//  TensorIOExamplesTests
//
//  Created by Phil Dow on 5/11/20.
//  Copyright © 2020 doc.ai. All rights reserved.
//

//  Note that we must set the Host Application to None in the test target, otherwise
//  we see the following error:

//  [libprotobuf ERROR google/protobuf/descriptor_database.cc:58] File already exists in database:
//  tensorflow/contrib/boosted_trees/proto/learner.proto
//  [libprotobuf FATAL google/protobuf/descriptor.cc:1358]
//  CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):

//  I believe this is a problem with coocapods linking the library twice, one into
//  the host application and the second time into the test target.

//  But then we must also comment out inherit! :search_paths in the Podfile for the test target
//  and run $ pod install again or we get undefined symbols errors.

//  Cocoapods is something else

import XCTest
import TensorIO

class TensorIOExamplesTests: XCTestCase {

}
