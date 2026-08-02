package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Path
import android.graphics.PointF

/**
 * Multi-stage spline smoothing pipeline that converts a raw polygon contour
 * into an organic, premium Bézier curve.
 *
 * Pipeline stages:
 *   1. Chaikin Corner Cutting — progressively rounds sharp corners
 *   2. Catmull-Rom → Cubic Bézier — converts smoothed polygon to continuous curves
 *
 * The output is a single closed Path composed entirely of cubic Bézier segments,
 * producing the premium vector look characteristic of the Huawei AI Pose effect.
 */
object SplineSmoother {

    /**
     * Full smoothing pipeline: Chaikin + Catmull-Rom spline.
     *
     * @param points     Raw contour points (closed polygon)
     * @param chaikinIterations Number of Chaikin corner-cutting passes (2–3 recommended)
     * @param tension    Catmull-Rom tension parameter (0.0 = tight curves, 0.5 = loose)
     * @return Smooth closed Path composed of cubic Bézier curves
     */
    fun smooth(
        points: List<PointF>,
        chaikinIterations: Int = 3,
        tension: Float = 0.3f
    ): Path {
        if (points.size < 3) {
            // Not enough points for meaningful smoothing
            val path = Path()
            if (points.isNotEmpty()) {
                path.moveTo(points[0].x, points[0].y)
                for (i in 1 until points.size) {
                    path.lineTo(points[i].x, points[i].y)
                }
            }
            return path
        }

        // ── Stage 1: Chaikin Corner Cutting ──
        var smoothed = points
        for (i in 0 until chaikinIterations) {
            smoothed = chaikinSmooth(smoothed)
        }

        // ── Stage 2: Catmull-Rom → Cubic Bézier ──
        return catmullRomToBezierPath(smoothed, tension)
    }

    /**
     * Chaikin's corner-cutting algorithm for closed polygons.
     *
     * For each edge, inserts two new points at 25% and 75% positions,
     * effectively cutting corners and producing a smoother polygon.
     * Each iteration approximately doubles the point count.
     *
     * Reference: Chaikin, G. "An algorithm for high-speed curve generation" (1974)
     *
     * @param points Closed polygon vertices
     * @return Smoothed polygon with approximately 2× the point count
     */
    private fun chaikinSmooth(points: List<PointF>): List<PointF> {
        if (points.size < 3) return points

        val result = mutableListOf<PointF>()
        val n = points.size

        for (i in 0 until n) {
            val p0 = points[i]
            val p1 = points[(i + 1) % n]

            // Q = 3/4 * P_i + 1/4 * P_{i+1}   (25% from start)
            val qx = 0.75f * p0.x + 0.25f * p1.x
            val qy = 0.75f * p0.y + 0.25f * p1.y

            // R = 1/4 * P_i + 3/4 * P_{i+1}   (75% from start)
            val rx = 0.25f * p0.x + 0.75f * p1.x
            val ry = 0.25f * p0.y + 0.75f * p1.y

            result.add(PointF(qx, qy))
            result.add(PointF(rx, ry))
        }

        return result
    }

    /**
     * Convert a smoothed closed polygon to a cubic Bézier Path using
     * Catmull-Rom spline interpolation.
     *
     * Each segment between consecutive points becomes a cubicTo() call,
     * with control points derived from the Catmull-Rom tangent formula.
     *
     * The tension parameter controls how tightly the curve follows the polygon:
     *   - Lower tension (0.1–0.2) = tighter curves, follows polygon closely
     *   - Higher tension (0.3–0.5) = smoother, more organic curves
     *
     * @param points  Smoothed polygon vertices
     * @param tension Catmull-Rom tension parameter
     * @return Closed Path composed of cubic Bézier curves
     */
    private fun catmullRomToBezierPath(points: List<PointF>, tension: Float): Path {
        val path = Path()
        val n = points.size
        if (n < 3) return path

        path.moveTo(points[0].x, points[0].y)

        for (i in 0 until n) {
            // Four control points for Catmull-Rom: P_{i-1}, P_i, P_{i+1}, P_{i+2}
            val p0 = points[if (i > 0) i - 1 else n - 1]
            val p1 = points[i]
            val p2 = points[(i + 1) % n]
            val p3 = points[(i + 2) % n]

            // Convert Catmull-Rom tangents to cubic Bézier control points:
            // CP1 = P1 + (P2 - P0) * tension
            // CP2 = P2 - (P3 - P1) * tension
            val cp1x = p1.x + (p2.x - p0.x) * tension
            val cp1y = p1.y + (p2.y - p0.y) * tension
            val cp2x = p2.x - (p3.x - p1.x) * tension
            val cp2y = p2.y - (p3.y - p1.y) * tension

            path.cubicTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y)
        }

        path.close()
        return path
    }
}
