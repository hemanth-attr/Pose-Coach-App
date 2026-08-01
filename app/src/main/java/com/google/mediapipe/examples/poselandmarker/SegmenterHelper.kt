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
        // Loading the model you just downloaded!
        baseOptionsBuilder.setModelAssetPath("selfie_segmenter.tflite")

        try {
            val optionsBuilder = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener(this::returnSegmentationResult)
                .setErrorListener(this::returnSegmentationError)

            val options = optionsBuilder.build()
            imageSegmenter = ImageSegmenter.createFromOptions(context, options)
        } catch (e: Exception) {
            segmenterListener?.onError("Segmenter failed to initialize: ${e.message}")
            Log.e("SegmenterHelper", "Segmenter failed to load model with error: ${e.message}")
        }
    }

   fun segmentLiveStreamFrame(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (imageSegmenter == null) return

        val frameTime = SystemClock.uptimeMillis()
        val bitmapBuffer = Bitmap.createBitmap(
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )

        // --- THE CRASH FIX ---
        // We carefully copy the pixels without destroying the original camera frame
        try {
            imageProxy.planes[0].buffer.rewind() // Ensure we start at the beginning
            bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            imageProxy.planes[0].buffer.rewind() // Rewind it again so the Pose AI can read it next!
        } catch (e: Exception) {
            return // If the frame is corrupted, skip it instead of crashing
        }

        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (isFrontCamera) {
                postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
            }
        }
        
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
        )
        
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()
        
        imageSegmenter?.segmentAsync(mpImage, frameTime)
    }
    private fun returnSegmentationResult(
        result: ImageSegmenterResult,
        image: MPImage
    ) {
        val finishTimeMs = SystemClock.uptimeMillis()
        segmenterListener?.onSegmentationResults(
            ResultBundle(result, image.width, image.height)
        )
    }

    private fun returnSegmentationError(error: RuntimeException) {
        segmenterListener?.onError(error.message ?: "An unknown error has occurred")
    }

    data class ResultBundle(
        val result: ImageSegmenterResult,
        val inputImageWidth: Int,
        val inputImageHeight: Int
    )

    interface SegmenterListener {
        fun onError(error: String)
        fun onSegmentationResults(resultBundle: ResultBundle)
    }
}