package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.nio.FloatBuffer

object SegmentationToVector {

    /**
     * Converts the segmentation mask from a PoseLandmarkerResult into a vector contour path.
     */
    fun extractContour(
        result: PoseLandmarkerResult,
        inputWidth: Int,
        inputHeight: Int,
        viewWidth: Int,
        viewHeight: Int,
        contourExtractor: SilhouetteContourExtractor
    ): List<PointF> {
        val masks = result.segmentationMasks()
        if (!masks.isPresent || masks.get().isEmpty()) {
            return emptyList()
        }

        val maskImage = masks.get()[0]
        val maskWidth = maskImage.width
        val maskHeight = maskImage.height
        
        // Extract the raw float buffer from the MPImage (confidence values 0.0 to 1.0)
        val floatBuffer: FloatBuffer = ByteBufferExtractor.extract(maskImage).asFloatBuffer()
        floatBuffer.rewind()

        // Create a bitmap to hold the thresholded mask
        val bitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(maskWidth * maskHeight)
        
        // Convert floats to binary pixels (White for foreground, Black for background)
        for (i in 0 until (maskWidth * maskHeight)) {
            val confidence = floatBuffer.get()
            if (confidence > 0.5f) {
                pixels[i] = Color.WHITE
            } else {
                pixels[i] = Color.BLACK
            }
        }
        bitmap.setPixels(pixels, 0, maskWidth, 0, 0, maskWidth, maskHeight)

        // Convert the Bitmap to an OpenCV contour using our existing extractor
        // We can create a temporary path representing a rectangle for every white pixel,
        // OR we can add a new method to SilhouetteContourExtractor that takes a Bitmap directly!
        
        // Let's add extractFromBitmap to SilhouetteContourExtractor
        return contourExtractor.extractFromBitmap(bitmap, viewWidth, viewHeight)
    }
}
