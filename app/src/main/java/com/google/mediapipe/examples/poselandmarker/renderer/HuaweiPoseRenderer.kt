package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.os.SystemClock

/**
 * Huawei AI Pose Premium Renderer — the master orchestrator.
 *
 * This is the single entry point for the entire rendering pipeline:
 *   Camera → MediaPipe → [Landmarks] → OneEuroFilter → ProceduralBodyGenerator
 *   → SilhouetteContourExtractor → SplineSmoother → Canvas draw
 *
 * Output: A premium, anti-aliased, white vector outline of a procedurally-generated
 * human body that closely resembles the Huawei AI Pose effect.
 *
 * Design principles:
 *   - No skeletons, no segmentation contours, no rectangles
 *   - One continuous, closed, organic vector outline
 *   - Constant thickness with rounded caps and joins
 *   - Anti-aliased, smooth, no jagged edges
 *   - Temporally stable (no jitter, no shaking)
 *   - Responsive to fast movements
 */
class HuaweiPoseRenderer {

    // ── Pipeline Components ──
    private val landmarkSmoother = LandmarkSmoother(
        landmarkCount = 33,
        minCutoff = 1.7f,   // Smooth when still
        beta = 0.01f         // Responsive when moving
    )
    private val contourExtractor = SilhouetteContourExtractor()
    private val contourSmoother = LandmarkSmoother(
        landmarkCount = 1,   // Will be resized dynamically
        minCutoff = 2.0f,
        beta = 0.005f
    )

    // ── Temporal contour smoothing ──
    // Separate One Euro filters for the final contour points to eliminate frame-to-frame jitter
    private var contourFilters: Array<OneEuroFilter>? = null

    // ── Rendering Paint ──
    private val outlinePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
        color = Color.WHITE
        strokeWidth = 5f
        // Subtle dark shadow for visibility on bright backgrounds
        setShadowLayer(3f, 0f, 0f, Color.argb(120, 0, 0, 0))
    }

    private val highlightPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
        color = Color.parseColor("#80FF3B30") // Semi-transparent bright red
    }

    // ── Cached state ──
    private var lastContour: List<PointF>? = null

    /**
     * Render the Huawei-style pose outline onto the given Canvas.
     *
     * @param canvas          Target canvas to draw on
     * @param landmarks       Transformed target pose landmarks (normalized 0-1)
     * @param incorrectLimbs  Set of incorrect limbs to highlight
     * @param imageWidth      Width of the source camera image
     * @param imageHeight     Height of the source camera image
     * @param scaleFactor     Scale factor from image → view coordinates
     * @param postTranslateX  Horizontal translation for CENTER_CROP alignment
     * @param postTranslateY  Vertical translation for CENTER_CROP alignment
     * @param viewWidth       Width of the overlay view
     * @param viewHeight      Height of the overlay view
     */
    fun render(
        canvas: Canvas,
        landmarks: List<PointF>,
        incorrectLimbs: Set<Limb>,
        imageWidth: Int,
        imageHeight: Int,
        scaleFactor: Float,
        postTranslateX: Float,
        postTranslateY: Float,
        viewWidth: Int,
        viewHeight: Int
    ) {
        if (landmarks.size < 33) return
        if (viewWidth <= 0 || viewHeight <= 0) return

        val timestamp = SystemClock.uptimeMillis()

        // ══════════════════════════════════════════
        // STAGE 1: Convert landmarks to view coordinates
        // ══════════════════════════════════════════
        val rawPoints = Array(landmarks.size.coerceAtMost(33)) { i ->
            val lm = landmarks[i]
            PointF(
                lm.x * imageWidth * scaleFactor + postTranslateX,
                lm.y * imageHeight * scaleFactor + postTranslateY
            )
        }

        // ══════════════════════════════════════════
        // STAGE 2: Temporal smoothing (One Euro Filter)
        // ══════════════════════════════════════════
        val smoothedPoints = landmarkSmoother.smooth(rawPoints, timestamp)

        // ══════════════════════════════════════════
        // STAGE 3: Generate procedural body
        // ══════════════════════════════════════════
        val bodyModel = ProceduralBodyGenerator.generate(smoothedPoints, incorrectLimbs)

        // ══════════════════════════════════════════
        // STAGE 4: Extract contour from silhouette
        // ══════════════════════════════════════════
        var contourPoints = contourExtractor.extract(bodyModel.mainPath, viewWidth, viewHeight)

        if (contourPoints.size < 3) {
            // Contour extraction failed — use cached contour if available
            contourPoints = lastContour ?: return
        }

        // ══════════════════════════════════════════
        // STAGE 5: Temporal smoothing on contour points
        // ══════════════════════════════════════════
        contourPoints = smoothContourPoints(contourPoints, timestamp)
        lastContour = contourPoints

        // ══════════════════════════════════════════
        // STAGE 6: Spline smoothing (Chaikin + Catmull-Rom)
        // ══════════════════════════════════════════
        val smoothOutline = SplineSmoother.smooth(
            points = contourPoints,
            chaikinIterations = 3,
            tension = 0.25f
        )

        // ══════════════════════════════════════════
        // STAGE 7: Render the premium outline & highlights
        // ══════════════════════════════════════════
        canvas.drawPath(smoothOutline, outlinePaint)
        
        // Draw highlights for incorrect limbs over the outline
        for (path in bodyModel.incorrectPaths) {
            canvas.drawPath(path, highlightPaint)
        }
    }

    /**
     * Apply per-point One Euro filtering to the contour for temporal stability.
     *
     * Since the contour point count can vary between frames (Douglas-Peucker
     * may return different numbers of points), we use a fixed-size filter array
     * and interpolate/resample the contour to match.
     */
    private fun smoothContourPoints(points: List<PointF>, timestamp: Long): List<PointF> {
        // Resample to a fixed number of points for consistent filtering
        val targetCount = 64
        val resampled = resampleContour(points, targetCount)

        // Initialize or reinitialize contour filters if size changed
        if (contourFilters == null || contourFilters!!.size != targetCount) {
            contourFilters = Array(targetCount) {
                OneEuroFilter(minCutoff = 2.5f, beta = 0.003f)
            }
        }

        // Apply per-point filtering
        return resampled.mapIndexed { i, pt ->
            contourFilters!![i].filter(pt.x, pt.y, timestamp)
        }
    }

    /**
     * Resample a closed contour to a fixed number of evenly-spaced points.
     * This ensures consistent point count for temporal filtering.
     *
     * @param points      Input contour points (variable count)
     * @param targetCount Desired output point count
     * @return Resampled contour with exactly [targetCount] points
     */
    private fun resampleContour(points: List<PointF>, targetCount: Int): List<PointF> {
        if (points.size <= 1) return points
        if (points.size == targetCount) return points

        // Calculate total perimeter
        var totalLength = 0f
        val segmentLengths = FloatArray(points.size)
        for (i in points.indices) {
            val next = (i + 1) % points.size
            val len = Geometry.distance(points[i], points[next])
            segmentLengths[i] = len
            totalLength += len
        }

        if (totalLength < 1f) return points

        // Walk along the perimeter at equal intervals
        val interval = totalLength / targetCount
        val result = mutableListOf<PointF>()
        var currentDist = 0f
        var segIndex = 0
        var segProgress = 0f

        for (i in 0 until targetCount) {
            val targetDist = i * interval

            // Advance along segments until we reach the target distance
            while (currentDist + segmentLengths[segIndex] - segProgress < targetDist && segIndex < points.size - 1) {
                currentDist += segmentLengths[segIndex] - segProgress
                segIndex = (segIndex + 1) % points.size
                segProgress = 0f
            }

            // Interpolate within the current segment
            val remaining = targetDist - currentDist
            val segLen = segmentLengths[segIndex]
            val t = if (segLen > 0f) ((segProgress + remaining) / segLen).coerceIn(0f, 1f) else 0f

            val p1 = points[segIndex]
            val p2 = points[(segIndex + 1) % points.size]
            result.add(PointF(
                p1.x + (p2.x - p1.x) * t,
                p1.y + (p2.y - p1.y) * t
            ))
        }

        return result
    }

    /**
     * Reset all temporal state.
     * Call when camera stops, restarts, or tracking is completely lost.
     */
    fun reset() {
        landmarkSmoother.reset()
        contourFilters?.forEach { it.reset() }
        contourFilters = null
        lastContour = null
    }

    /**
     * Release all resources held by this renderer.
     * Call when the renderer is being permanently destroyed.
     */
    fun release() {
        reset()
        contourExtractor.release()
    }
}
