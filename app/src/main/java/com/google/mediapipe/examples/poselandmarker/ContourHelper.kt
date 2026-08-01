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

    // Converts the AI Mask into a simplified mathematical boundary
    fun extractSmoothContour(
        maskBuffer: FloatBuffer,
        maskWidth: Int,
        maskHeight: Int,
        viewWidth: Float,
        viewHeight: Float
    ): List<PointF> {
        // 1. Convert the FloatBuffer (0.0 to 1.0) into an OpenCV Image (Mat)
        val floatArray = FloatArray(maskWidth * maskHeight)
        maskBuffer.rewind()
        maskBuffer.get(floatArray)

        val sourceMat = Mat(maskHeight, maskWidth, CvType.CV_32FC1)
        sourceMat.put(0, 0, floatArray)

        // 2. Threshold the image (Turn confidence > 0.5 into solid white, rest to black)
        val binaryMat = Mat()
        Imgproc.threshold(sourceMat, binaryMat, 0.5, 255.0, Imgproc.THRESH_BINARY)
        
        val byteMat = Mat()
        binaryMat.convertTo(byteMat, CvType.CV_8UC1)

        // 3. Find the Contours (The outer edge of the white pixels)
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(byteMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // 4. Find the Largest Contour (The Human Body)
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
        if (bestContour != null) {
            // 5. Simplify the Contour (Ramer-Douglas-Peucker algorithm)
            // This removes microscopic jagged pixel edges and reduces it to key structural points
            val contour2f = MatOfPoint2f(*bestContour.toArray())
            val approxCurve = MatOfPoint2f()
            val epsilon = 0.008 * Imgproc.arcLength(contour2f, true) // Lower = more detailed, Higher = smoother
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true)

            // 6. Scale points from the 256x256 mask up to your actual phone screen dimensions
            val scaleX = viewWidth / maskWidth.toFloat()
            val scaleY = viewHeight / maskHeight.toFloat()

            for (point in approxCurve.toArray()) {
                finalPoints.add(PointF(point.x.toFloat() * scaleX, point.y.toFloat() * scaleY))
            }
        }

        // Clean up C++ memory to prevent memory leaks
        sourceMat.release()
        binaryMat.release()
        byteMat.release()
        hierarchy.release()

        return finalPoints
    }

    // THE SECRET SAUCE: Converts rigid points into flowing organic curves
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
            val p0 = points[if (i > 0) i - 1 else points.size - 1] // Wrap around for closed loop
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