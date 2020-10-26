package ai.doc.tensorio.androidexample

import ai.doc.tensorio.core.data.Batch
import ai.doc.tensorio.core.model.Model
import ai.doc.tensorio.core.modelbundle.ModelBundle
import ai.doc.tensorio.core.training.TrainableModel
import ai.doc.tensorio.core.utilities.AndroidAssets
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.IOException
import java.nio.file.FileSystemException

@RunWith(AndroidJUnit4::class)
class TF20KerasCIFAR10MobileNetTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context
    private val epsilon = 0.01f

    /** Set up a models directory to copy assets to for testing  */

    @Before
    @Throws(Exception::class)
    fun setUp() {
        val f = File(appContext.filesDir, "models")
        if (!f.mkdirs()) {
            throw FileSystemException("on create: " + f.path)
        }
    }

    /** Tear down the models directory  */

    @After
    @Throws(Exception::class)
    fun tearDown() {
        val f = File(appContext.filesDir, "models")
        deleteRecursive(f)
    }

    /** Create a model bundle from a file, copying the asset to models  */

    @Throws(IOException::class, ModelBundle.ModelBundleException::class)
    private fun bundleForFile(filename: String, backend: String): ModelBundle? {
        val assetpath = "models/$backend/$filename"
        val dir = File(appContext.filesDir, "models")
        val file = File(dir, filename)

        AndroidAssets.copyAsset(appContext, assetpath, file)

        return ModelBundle.bundleWithFile(file)
    }

    /** Delete a directory and all its contents  */

    @Throws(FileSystemException::class)
    private fun deleteRecursive(f: File) {
        if (f.isDirectory) {
            for (child in f.listFiles()) {
                deleteRecursive(child)
            }
        }
        if (!f.delete()) {
            throw FileSystemException("on delete: " + f.path)
        }
    }

    @Test
    fun testSavedModel() {
        try {
            val bundle = bundleForFile("keras-cifar10-mobilenet-save-model.tiobundle", "tf_2_0")
            Assert.assertNotNull(bundle)

            val model = bundle!!.newModel()
            Assert.assertNotNull(model)

            val stream1 = appContext.assets.open("cifar-car.jpg")
            val bitmap1 = BitmapFactory.decodeStream(stream1)

            val output = model.runOn(bitmap1)
            Assert.assertNotNull(output)

            val logits = output.get("StatefulPartitionedCall")
            Assert.assertNotNull(logits)
        } catch (e: ModelBundle.ModelBundleException) {
            Assert.fail();
        } catch (e: Model.ModelException) {
            Assert.fail();
        } catch (e: IOException) {
            Assert.fail();
        }
    }

    @Test
    fun testEstimatorPredict() {
        try {
            val bundle = bundleForFile("keras-cifar10-mobilenet-estimator-predict.tiobundle", "tf_2_0")
            Assert.assertNotNull(bundle)

            val model = bundle!!.newModel()
            Assert.assertNotNull(model)

            val stream1 = appContext.assets.open("cifar-car.jpg")
            val bitmap1 = BitmapFactory.decodeStream(stream1)

            val output = model.runOn(bitmap1)
            Assert.assertNotNull(output)

            val logits = output.get("sequential/dense/BiasAdd")
            Assert.assertNotNull(logits)
        } catch (e: ModelBundle.ModelBundleException) {
            Assert.fail();
        } catch (e: Model.ModelException) {
            Assert.fail();
        } catch (e: IOException) {
            Assert.fail();
        }
    }

    @Test
    fun testEstimatorTrain() {
        try {
            val bundle = bundleForFile("keras-cifar10-mobilenet-estimator-train.tiobundle", "tf_2_0")
            Assert.assertNotNull(bundle)

            val model = bundle!!.newModel() as TrainableModel
            Assert.assertNotNull(model)

            val stream1 = appContext.assets.open("cifar-car.jpg")
            val bitmap1 = BitmapFactory.decodeStream(stream1)
            val label1 = floatArrayOf(0.0f)

            val item = Batch.Item()
            item.put("image", bitmap1)
            item.put("label", label1)

            val batch = Batch(item)

            for (epoch in 0 until 4) {
                val output = model.trainOn(batch)
                Assert.assertNotNull(output)
                val loss = output.get("total_loss")
                Assert.assertNotNull(loss)
            }
        } catch (e: ModelBundle.ModelBundleException) {
            Assert.fail();
        } catch (e: Model.ModelException) {
            Assert.fail();
        } catch (e: IOException) {
            Assert.fail();
        }
    }
}