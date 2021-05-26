import os
import glob
import unittest
import json
import jsonschema

class TestSchema(unittest.TestCase):
  TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'tests'
  )
  VALID_TESTS_DIR = os.path.join(
    TESTS_DIR,
    'valid'
  )
  VALID_TEST_FILES = glob.glob(os.path.join(VALID_TESTS_DIR, '*.json'))

  def setUp(self):
    pass
  
  def tearDown(self):
    pass

  def test_schema_with_valid_json(self):

    with open('schema.json') as f:
      schema = json.load(f)
    
    for test_file in self.VALID_TEST_FILES:
      # print("Testing {}".format(os.path.join('valid',os.path.basename(test_file))))
      with open(test_file) as f:
        json_test = json.load(f)
      try:
        jsonschema.validate(instance=json_test,schema=schema)
      except Exception: 
        self.fail("The file {} failed validation".format(os.path.basename(test_file)))

  def test_schema_with_invalid_json(self):
    # TODO: invalid json tests
    pass
