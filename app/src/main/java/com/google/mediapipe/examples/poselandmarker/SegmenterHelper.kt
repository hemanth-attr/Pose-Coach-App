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
    
    // MEMORY FIX: Recycle these instead of creating them 30 times a second!
    private var bitmapBuffer: Bitmap? = null
    private var rotatedBitmap: Bitmap? = null

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
                .setOutputConfidenceMasks(true) // CRUCIAL FIX: Tell AI to output the mask!
                .setOutputCategoryMask(false)
                .setResultListener(this::returnSegmentationResult)
                .setErrorListener(this::returnSegmentationError)

            imageSegmenter = ImageSegmenter.createFromOptions(context, optionsBuilder.build())
        } catch (e: Exception) {
            segmenterListener?.onError("Segmenter failed to initialize: ${e.message}")
        }
    }

    fun segmentLiveStreamFrame(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (imageSegmenter == null) {
            imageProxy.close()
            return
        }

        try {
            val frameTime = SystemClock.uptimeMillis()
            
            // Only create the bitmap once!
            if (bitmapBuffer == null || bitmapBuffer!!.width != imageProxy.width || bitmapBuffer!!.height != imageProxy.height) {
                bitmapBuffer = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            }

            imageProxy.planes[0].buffer.rewind()
            bitmapBuffer!!.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }
            
            rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer!!, 0, 0, bitmapBuffer!!.width, bitmapBuffer!!.height, matrix, true
            )
            
            val mpImage = BitmapImageBuilder(rotatedBitmap!!).build()
            imageSegmenter?.segmentAsync(mpImage, frameTime)
            
        } catch (e: Exception) {
            Log.e("SegmenterHelper", "Frame skipped due to buffer issue")
        } finally {
            // CRUCIAL FIX: Safely close the image so the camera can capture the next one!
            imageProxy.close()
        }
    }

    private fun returnSegmentationResult(result: ImageSegmenterResult, image: MPImage) {
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