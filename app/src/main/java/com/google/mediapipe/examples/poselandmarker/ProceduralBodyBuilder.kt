package com.google.mediapipe.examples.poselandmarker

import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.hypot

object ProceduralBodyBuilder {

    // Temporal smoothing state
    private var emaLandmarks: Array<PointF>? = null
    private const val EMA_ALPHA = 0.25f // Lower is smoother, higher is more responsive

    private val capsulePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    /**
     * Converts a line segment into a filled capsule path of the given thickness.
     */
    private fun getCapsulePath(p1: PointF, p2: PointF, thickness: Float): Path {
        val src = Path()
        src.moveTo(p1.x, p1.y)
        src.lineTo(p2.x, p2.y)
        
        val dst = Path()
        capsulePaint.strokeWidth = thickness
        capsulePaint.getFillPath(src, dst)
        return dst
    }

    /**
     * Generates a smooth, procedural anatomical body path from raw MediaPipe landmarks.
     */
    fun buildBodyPath(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int,
        scaleFactor: Float,
        postTranslateX: Float,
        postTranslateY: Float
    ): Path {
        // 1. Extract and Smooth Landmarks
        val currentPoints = Array(landmarks.size) { PointF() }
        for (i in landmarks.indices) {
            val lm = landmarks[i]
            val viewX = lm.x() * imageWidth * scaleFactor + postTranslateX
            val viewY = lm.y() * imageHeight * scaleFactor + postTranslateY
            currentPoints[i] = PointF(viewX, viewY)
        }

        if (emaLandmarks == null || emaLandmarks!!.size != landmarks.size) {
            emaLandmarks = currentPoints.clone()
        } else {
            for (i in currentPoints.indices) {
                emaLandmarks!![i].x = (EMA_ALPHA * currentPoints[i].x) + ((1f - EMA_ALPHA) * emaLandmarks!![i].x)
                emaLandmarks!![i].y = (EMA_ALPHA * currentPoints[i].y) + ((1f - EMA_ALPHA) * emaLandmarks!![i].y)
            }
        }

        val pts = emaLandmarks!!

        // 2. Base Thickness Calculations
        // We use shoulder width to proportionally scale the body thickness
        val shoulderDx = pts[11].x - pts[12].x
        val shoulderDy = pts[11].y - pts[12].y
        val shoulderWidth = hypot(shoulderDx.toDouble(), shoulderDy.toDouble()).toFloat()
        
        // Failsafe thickness if shoulder width is unusually small (e.g. side profile)
        val baseThickness = shoulderWidth.coerceAtLeast(100f)
        
        val torsoThickness = baseThickness * 0.45f
        val limbThickness = baseThickness * 0.35f
        val forearmThickness = baseThickness * 0.25f
        val legThickness = baseThickness * 0.4f
        val calfThickness = baseThickness * 0.3f
        val neckThickness = baseThickness * 0.25f
        
        // 3. Construct Body Parts
        val fullBody = Path()

        // --- NECK ---
        val midShoulder = PointF((pts[11].x + pts[12].x) / 2f, (pts[11].y + pts[12].y) / 2f)
        fullBody.op(getCapsulePath(pts[0], midShoulder, neckThickness), Path.Op.UNION)

        // --- HEAD ---
        // Head size based on ear distance, or fallback to shoulder proportion
        val earDx = pts[7].x - pts[8].x
        val earDy = pts[7].y - pts[8].y
        var headRadius = hypot(earDx.toDouble(), earDy.toDouble()).toFloat() * 0.8f
        if (headRadius < baseThickness * 0.35f) {
            headRadius = baseThickness * 0.45f // Side profile fallback
        }
        val headPath = Path()
        // Center head slightly above nose to look natural
        val headCenterY = pts[0].y - (headRadius * 0.2f)
        headPath.addCircle(pts[0].x, headCenterY, headRadius, Path.Direction.CW)
        fullBody.op(headPath, Path.Op.UNION)

        // --- TORSO ---
        // Left side, Right side, Top side, Bottom side
        fullBody.op(getCapsulePath(pts[11], pts[23], torsoThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[12], pts[24], torsoThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[11], pts[12], torsoThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[23], pts[24], torsoThickness), Path.Op.UNION)
        
        // Fill the center of the torso
        val torsoCore = Path()
        torsoCore.moveTo(pts[11].x, pts[11].y)
        torsoCore.lineTo(pts[12].x, pts[12].y)
        torsoCore.lineTo(pts[24].x, pts[24].y)
        torsoCore.lineTo(pts[23].x, pts[23].y)
        torsoCore.close()
        fullBody.op(torsoCore, Path.Op.UNION)

        // --- ARMS ---
        // Left Arm
        fullBody.op(getCapsulePath(pts[11], pts[13], limbThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[13], pts[15], forearmThickness), Path.Op.UNION)
        // Right Arm
        fullBody.op(getCapsulePath(pts[12], pts[14], limbThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[14], pts[16], forearmThickness), Path.Op.UNION)

        // --- LEGS ---
        // Left Leg
        fullBody.op(getCapsulePath(pts[23], pts[25], legThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[25], pts[27], calfThickness), Path.Op.UNION)
        // Right Leg
        fullBody.op(getCapsulePath(pts[24], pts[26], legThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[26], pts[28], calfThickness), Path.Op.UNION)

        // --- FEET ---
        fullBody.op(getCapsulePath(pts[27], pts[31], forearmThickness), Path.Op.UNION)
        fullBody.op(getCapsulePath(pts[28], pts[32], forearmThickness), Path.Op.UNION)

        return fullBody
    }

    fun reset() {
        emaLandmarks = null
    }
}
