package com.google.mediapipe.examples.poselandmarker

import android.graphics.Path
import android.graphics.PointF
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

object ContourHelper {

    private var emaMask: FloatArray? = null
    private const val EMA_ALPHA = 0.4f // Balances responsiveness and smoothness

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

        // 1. TEMPORAL EMA FILTERING (Fixes "Boiling Edges")
        if (emaMask == null || emaMask!!.size != size) {
            emaMask = floatArray.clone()
        } else {
            val ema = emaMask!!
            for (i in 0 until size) {
                ema[i] = (EMA_ALPHA * floatArray[i]) + ((1f - EMA_ALPHA) * ema[i])
            }
        }

        val sourceMat = Mat(maskHeight, maskWidth, CvType.CV_32FC1)
        sourceMat.put(0, 0, emaMask!!)

        val binaryMat = Mat()
        Imgproc.threshold(sourceMat, binaryMat, 0.35, 255.0, Imgproc.THRESH_BINARY)
        
        val byteMat = Mat()
        binaryMat.convertTo(byteMat, CvType.CV_8UC1)

        // 2. MORPHOLOGICAL SMOOTHING (Unifies Overlapping Limbs)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(15.0, 15.0))
        val closedMat = Mat()
        Imgproc.morphologyEx(byteMat, closedMat, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(closedMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var maxArea = 0.0
        var bestContour: MatOfPoint? = null
        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area > maxArea) {
                maxArea = area
                bestContour = contour
            }
        }

        val finalPoints = mutableListOf<PointF>()
        
        // CRASH FIX: Using ?.let guarantees the Kotlin compiler won't throw a Smart Cast error!
        bestContour?.let { contour ->
            val contour2f = MatOfPoint2f(*contour.toArray())
            val approxCurve = MatOfPoint2f()
            val epsilon = 0.015 * Imgproc.arcLength(contour2f, true) // Smooths out the polygon more
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true)

            val scaleX = viewWidth / maskWidth.toFloat()
            val scaleY = viewHeight / maskHeight.toFloat()

            for (point in approxCurve.toArray()) {
                finalPoints.add(PointF(point.x.toFloat() * scaleX, point.y.toFloat() * scaleY))
            }
        }

        sourceMat.release()
        binaryMat.release()
        byteMat.release()
        closedMat.release()
        kernel.release()
        hierarchy.release()

        return finalPoints
    }

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
}