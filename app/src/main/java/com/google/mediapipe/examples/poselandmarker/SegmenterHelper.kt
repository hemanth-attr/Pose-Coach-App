package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult

class SegmenterHelper(
    val context: Context,
    val segmenterListener: SegmenterListener?
) {
    private var imageSegmenter: ImageSegmenter? = null

    init {
        setupSegmenter()
    }

    fun clearSegmenter() {
        imageSegmenter?.close()
        imageSegmenter = null
    }

    private fun setupSegmenter() {
        val baseOptionsBuilder = BaseOptions.builder()
        baseOptionsBuilder.setModelAssetPath("selfie_segmenter.tflite")

        try {
            val optionsBuilder = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setOutputConfidenceMasks(true)
                .setOutputCategoryMask(false)
                .setResultListener(this::returnSegmentationResult)
                .setErrorListener(this::returnSegmentationError)

            imageSegmenter = ImageSegmenter.createFromOptions(context, optionsBuilder.build())
        } catch (e: Exception) {
            segmenterListener?.onError("Segmenter failed to initialize: ${e.message}")
        }
    }

    fun segmentLiveStreamFrame(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (imageSegmenter == null) return

        try {
            val frameTime = SystemClock.uptimeMillis()

            val plane = imageProxy.planes[0]
            val buffer = plane.buffer
            val width = imageProxy.width
            val height = imageProxy.height
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride

            // FIX 1: The "Row Stride" Padding Fix to prevent diagonal glitches
            val paddedWidth = rowStride / pixelStride
            val safeBitmap = if (paddedWidth == width) {
                val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                buffer.rewind()
                bmp.copyPixelsFromBuffer(buffer)
                bmp
            } else {
                val paddedBmp = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888)
                buffer.rewind()
                paddedBmp.copyPixelsFromBuffer(buffer)
                Bitmap.createBitmap(paddedBmp, 0, 0, width, height) // Crop out the hidden padding!
            }

            // FIX 2: Correct rotation math
            val rotationMatrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            }
            val rotatedBitmap = Bitmap.createBitmap(safeBitmap, 0, 0, width, height, rotationMatrix, false)

            // FIX 3: Prevent the mask from being shifted off-screen during front-camera mirroring
            val finalBitmap = if (isFrontCamera) {
                val flipMatrix = Matrix().apply { 
                    postScale(-1f, 1f, rotatedBitmap.width / 2f, rotatedBitmap.height / 2f) 
                }
                Bitmap.createBitmap(rotatedBitmap, 0, 0, rotatedBitmap.width, rotatedBitmap.height, flipMatrix, false)
            } else {
                rotatedBitmap
            }
            
            val mpImage = BitmapImageBuilder(finalBitmap).build()
            imageSegmenter?.segmentAsync(mpImage, frameTime)
            
        } catch (e: Exception) {
            Log.e("SegmenterHelper", "Frame skipped due to buffer issue")
        } finally {
            // CRITICAL FIX: Always close the image proxy to keep the 60FPS pipeline flowing
            imageProxy.close()
        }
    }

    private fun returnSegmentationResult(result: ImageSegmenterResult, image: MPImage) {
        segmenterListener?.onSegmentationResults(ResultBundle(result, image.width, image.height))
    }

    private fun returnSegmentationError(error: RuntimeException) {
        segmenterListener?.onError(error.message ?: "An unknown error has occurred")
    }

    data class ResultBundle(val result: ImageSegmenterResult, val inputImageWidth: Int, val inputImageHeight: Int)
    interface SegmenterListener {
        fun onError(error: String)
        fun onSegmentationResults(resultBundle: ResultBundle)
    }
}