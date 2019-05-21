import os
import glob
import subprocess
import unittest

# Run all tests for all schemas and backends with python -m unittest test.py

class TestSchema(unittest.TestCase):
  FILE_DIR = os.path.dirname(os.path.abspath(__file__))
  VERSION_DIRS = glob.glob('v*')
  BACKENDS = [
    "tiotask",
    "tensorflow",
    "tflite"
  ]

  def setUp(self):
    pass
  
  def tearDown(self):
    pass

  def test_schema_versions(self):
    for version_dir in self.VERSION_DIRS:
      for backend in self.BACKENDS:
        backend_dir = os.path.join(version_dir, backend)
        if not os.path.exists(backend_dir):
          print("No tests with backend and version at", backend_dir)
          continue

        os.chdir(backend_dir)
        result = subprocess.run(["python", "-m", "unittest", 'test.py'])
        os.chdir(self.FILE_DIR)
        
        if result.returncode is not 0:
          self.fail("The tests for {} failed".format(backend_dir))
