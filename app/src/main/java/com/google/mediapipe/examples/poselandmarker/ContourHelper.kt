package com.google.mediapipe.examples.poselandmarker

import android.graphics.Path
import android.graphics.PointF
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.abs

object ContourHelper {

    // -------------------------------------------------------
    // TEMPORAL SMOOTHING: Exponential Moving Average on raw mask
    // -------------------------------------------------------
    private var emaMask: FloatArray? = null
    private const val EMA_ALPHA = 0.35f

    // -------------------------------------------------------
    // ONE EURO FILTER: Velocity-aware smoothing on output points
    // Provides stability when still, responsiveness when moving
    // -------------------------------------------------------
    private var prevPoints: List<PointF>? = null
    private var prevTimestamp: Long = 0L
    private const val ONE_EURO_MIN_CUTOFF = 1.5f
    private const val ONE_EURO_BETA = 0.007f

    private fun oneEuroAlpha(cutoff: Float, dt: Float): Float {
        val tau = 1.0f / (2.0f * Math.PI.toFloat() * cutoff)
        return 1.0f / (1.0f + tau / dt)
    }

    private fun applyOneEuroFilter(
        currentPoints: List<PointF>,
        timestamp: Long
    ): List<PointF> {
        val prev = prevPoints
        if (prev == null || prev.size != currentPoints.size || prevTimestamp == 0L) {
            prevPoints = currentPoints.toList()
            prevTimestamp = timestamp
            return currentPoints
        }

        val dt = ((timestamp - prevTimestamp).coerceAtLeast(1L)) / 1000.0f
        prevTimestamp = timestamp

        val filtered = mutableListOf<PointF>()
        val newPrev = mutableListOf<PointF>()

        for (i in currentPoints.indices) {
            val dx = abs(currentPoints[i].x - prev[i].x) / dt
            val dy = abs(currentPoints[i].y - prev[i].y) / dt

            val cutoffX = ONE_EURO_MIN_CUTOFF + ONE_EURO_BETA * dx
            val cutoffY = ONE_EURO_MIN_CUTOFF + ONE_EURO_BETA * dy

            val alphaX = oneEuroAlpha(cutoffX, dt)
            val alphaY = oneEuroAlpha(cutoffY, dt)

            val filteredX = alphaX * currentPoints[i].x + (1f - alphaX) * prev[i].x
            val filteredY = alphaY * currentPoints[i].y + (1f - alphaY) * prev[i].y

            filtered.add(PointF(filteredX, filteredY))
            newPrev.add(PointF(filteredX, filteredY))
        }

        prevPoints = newPrev
        return filtered
    }

    // -------------------------------------------------------
    // MAIN PIPELINE: Mask -> Contour -> Smooth -> Filter
    // -------------------------------------------------------
    fun extractSmoothContour(
        maskBuffer: FloatBuffer,
        maskWidth: Int,
        maskHeight: Int,
        viewWidth: Float,
        viewHeight: Float
    ): List<PointF> {
        val size = maskWidth * maskHeight
        val floatArray = FloatArray(size)
        maskBuffer.rewind()
        maskBuffer.get(floatArray)

        // 1. TEMPORAL EMA on raw mask
        if (emaMask == null || emaMask!!.size != size) {
            emaMask = floatArray.clone()
        } else {
            val ema = emaMask!!
            for (i in 0 until size) {
                ema[i] = (EMA_ALPHA * floatArray[i]) + ((1f - EMA_ALPHA) * ema[i])
            }
        }

        // 2. Convert to OpenCV Mat
        val sourceMat = Mat(maskHeight, maskWidth, CvType.CV_32FC1)
        sourceMat.put(0, 0, emaMask!!)

        // 3. Gaussian blur BEFORE thresholding
        val blurredMat = Mat()
        Imgproc.GaussianBlur(sourceMat, blurredMat, Size(7.0, 7.0), 0.0)

        // 4. Threshold
        val binaryMat = Mat()
        Imgproc.threshold(blurredMat, binaryMat, 0.3, 255.0, Imgproc.THRESH_BINARY)

        val byteMat = Mat()
        binaryMat.convertTo(byteMat, CvType.CV_8UC1)

        // 5. Morphological ERODE — removes thin noise
        val erodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        val erodedMat = Mat()
        Imgproc.erode(byteMat, erodedMat, erodeKernel)

        // 6. Morphological CLOSE — welds overlapping limbs
        val closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(19.0, 19.0))
        val closedMat = Mat()
        Imgproc.morphologyEx(erodedMat, closedMat, Imgproc.MORPH_CLOSE, closeKernel)

        // 7. Find contours
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(closedMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // 8. Find the largest contour (the human body)
        var maxArea = 0.0
        var bestContour: MatOfPoint? = null
        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area > maxArea) {
                maxArea = area
                bestContour = contour
            }
        }

        val rawPoints = mutableListOf<PointF>()

        bestContour?.let { contour ->
            // 9. Douglas-Peucker simplification
            val contour2f = MatOfPoint2f(*contour.toArray())
            val approxCurve = MatOfPoint2f()
            val epsilon = 0.012 * Imgproc.arcLength(contour2f, true)
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true)

            // 10. Scale from mask coordinates to view coordinates
            val scaleX = viewWidth / maskWidth.toFloat()
            val scaleY = viewHeight / maskHeight.toFloat()

            for (point in approxCurve.toArray()) {
                rawPoints.add(PointF(point.x.toFloat() * scaleX, point.y.toFloat() * scaleY))
            }

            contour2f.release()
            approxCurve.release()
        }

        // Release all OpenCV memory
        sourceMat.release()
        blurredMat.release()
        binaryMat.release()
        byteMat.release()
        erodeKernel.release()
        erodedMat.release()
        closeKernel.release()
        closedMat.release()
        hierarchy.release()
        for (c in contours) c.release()

        // 11. Apply One Euro Filter for temporal stability
        return if (rawPoints.size >= 3) {
            applyOneEuroFilter(rawPoints, System.currentTimeMillis())
        } else {
            rawPoints
        }
    }

    // -------------------------------------------------------
    // SPLINE RENDERER: Converts rigid polygon into organic curves
    // -------------------------------------------------------
    fun createSplinePath(points: List<PointF>, tension: Float = 0.25f): Path {
        val path = Path()
        if (points.isEmpty()) return path
        if (points.size < 3) {
            path.moveTo(points[0].x, points[0].y)
            for (i in 1 until points.size) path.lineTo(points[i].x, points[i].y)
            return path
        }

        path.moveTo(points[0].x, points[0].y)

        for (i in 0 until points.size) {
            val p0 = points[if (i > 0) i - 1 else points.size - 1]
            val p1 = points[i]
            val p2 = points[(i + 1) % points.size]
            val p3 = points[(i + 2) % points.size]

            val x1 = p1.x + (p2.x - p0.x) * tension
            val y1 = p1.y + (p2.y - p0.y) * tension
            val x2 = p2.x - (p3.x - p1.x) * tension
            val y2 = p2.y - (p3.y - p1.y) * tension

            path.cubicTo(x1, y1, x2, y2, p2.x, p2.y)
        }

        path.close()
        return path
    }

    /** Reset all temporal state (call when camera stops/starts) */
    fun reset() {
        emaMask = null
        prevPoints = null
        prevTimestamp = 0L
    }
}