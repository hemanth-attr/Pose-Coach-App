package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Converts a filled body Path into a clean vector contour by:
 *   1. Rasterizing the filled body onto a small offscreen bitmap
 *   2. Applying morphological operations to weld micro-gaps
 *   3. Extracting the outer contour via OpenCV findContours
 *   4. Simplifying the contour with Douglas-Peucker
 *   5. Mapping coordinates back to view space
 *
 * This is the critical step that converts overlapping capsule geometry
 * into ONE continuous outline with no gaps, overlaps, or internal edges.
 *
 * Performance: By rasterizing at a reduced resolution (e.g., 256×456),
 * the OpenCV operations complete in ~2ms on mid-range devices.
 */
class SilhouetteContourExtractor {

    companion object {
        // Offscreen bitmap resolution — lower = faster, but less detail
        // 256×456 provides good balance between quality and performance
        private const val RASTER_WIDTH = 256
        private const val RASTER_HEIGHT = 456

        // Douglas-Peucker simplification factor (relative to arc length)
        // Lower = more points, higher fidelity. Higher = fewer points, faster.
        private const val SIMPLIFICATION_EPSILON = 0.008

        // Morphological close kernel size — welds small gaps between body parts
        private const val MORPH_CLOSE_SIZE = 9.0
    }

    // ── Reusable offscreen rendering resources (no per-frame allocation) ──
    private var offscreenBitmap: Bitmap = Bitmap.createBitmap(RASTER_WIDTH, RASTER_HEIGHT, Bitmap.Config.ARGB_8888)
    private var offscreenCanvas: Canvas = Canvas(offscreenBitmap)
    private val fillPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        isAntiAlias = false // No AA needed for binary rasterization
    }

    /**
     * Extract the outer contour of the body silhouette.
     *
     * @param bodyPath   Filled Path representing the merged body silhouette (in view coordinates)
     * @param viewWidth  Width of the output view (for coordinate mapping)
     * @param viewHeight Height of the output view (for coordinate mapping)
     * @return Ordered list of contour points in view coordinates, or empty list if extraction fails
     */
    fun extract(bodyPath: Path, viewWidth: Int, viewHeight: Int): List<PointF> {
        if (viewWidth <= 0 || viewHeight <= 0) return emptyList()

        // ── Step 1: Rasterize the body Path onto the offscreen bitmap ──
        offscreenCanvas.drawColor(Color.BLACK) // Clear to black

        // Scale the body path from view space → offscreen bitmap space
        val scaleX = RASTER_WIDTH.toFloat() / viewWidth.toFloat()
        val scaleY = RASTER_HEIGHT.toFloat() / viewHeight.toFloat()

        offscreenCanvas.save()
        offscreenCanvas.scale(scaleX, scaleY)
        offscreenCanvas.drawPath(bodyPath, fillPaint)
        offscreenCanvas.restore()

        return extractFromBitmap(offscreenBitmap, viewWidth, viewHeight)
    }

    /**
     * Extracts a contour from an arbitrary Bitmap mask (e.g. from MediaPipe segmentation).
     */
    fun extractFromBitmap(sourceBitmap: Bitmap, viewWidth: Int, viewHeight: Int, imageWidth: Int, imageHeight: Int): List<PointF> {
        val srcWidth = sourceBitmap.width
        val srcHeight = sourceBitmap.height

        // ── Step 2: Convert bitmap to OpenCV Mat ──
        val pixels = IntArray(srcWidth * srcHeight)
        sourceBitmap.getPixels(pixels, 0, srcWidth, 0, 0, srcWidth, srcHeight)

        // Convert ARGB pixels to grayscale byte array
        val grayBytes = ByteArray(srcWidth * srcHeight)
        for (i in pixels.indices) {
            // Extract red channel (since we assume white on black, R=G=B)
            grayBytes[i] = ((pixels[i] shr 16) and 0xFF).toByte()
        }

        val grayMat = Mat(srcHeight, srcWidth, CvType.CV_8UC1)
        grayMat.put(0, 0, grayBytes)

        // ── Step 3: Threshold to binary ──
        val binaryMat = Mat()
        Imgproc.threshold(grayMat, binaryMat, 127.0, 255.0, Imgproc.THRESH_BINARY)

        // ── Step 4: Morphological CLOSE — weld micro-gaps between body parts ──
        val closeKernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE)
        )
        val closedMat = Mat()
        Imgproc.morphologyEx(binaryMat, closedMat, Imgproc.MORPH_CLOSE, closeKernel)

        // ── Step 5: Find contours — external only ──
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            closedMat, contours, hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        // ── Step 6: Find the largest contour (the human body) ──
        var maxArea = 0.0
        var bestContour: MatOfPoint? = null
        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area > maxArea) {
                maxArea = area
                bestContour = contour
            }
        }

        val result = mutableListOf<PointF>()

        bestContour?.let { contour ->
            // ── Step 7: Douglas-Peucker simplification ──
            val contour2f = MatOfPoint2f(*contour.toArray())
            val approxCurve = MatOfPoint2f()
            val epsilon = SIMPLIFICATION_EPSILON * Imgproc.arcLength(contour2f, true)
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true)

            // ── Step 8: Map coordinates back to view space using CENTER_CROP ──
            val scaleFactor = Math.max(viewWidth * 1f / imageWidth, viewHeight * 1f / imageHeight)
            val postTranslateX = (viewWidth - imageWidth * scaleFactor) / 2f
            val postTranslateY = (viewHeight - imageHeight * scaleFactor) / 2f

            for (point in approxCurve.toArray()) {
                val normX = point.x.toFloat() / srcWidth.toFloat()
                val normY = point.y.toFloat() / srcHeight.toFloat()
                
                val viewX = (normX * imageWidth) * scaleFactor + postTranslateX
                val viewY = (normY * imageHeight) * scaleFactor + postTranslateY
                
                result.add(PointF(viewX, viewY))
            }

            contour2f.release()
            approxCurve.release()
        }

        // ── Cleanup OpenCV Mats ──
        grayMat.release()
        binaryMat.release()
        closeKernel.release()
        closedMat.release()
        hierarchy.release()
        for (c in contours) c.release()

        return result
    }

    /**
     * Release the offscreen bitmap resources.
     * Call when the renderer is being destroyed.
     */
    fun release() {
        if (!offscreenBitmap.isRecycled) {
            offscreenBitmap.recycle()
        }
    }
}
