package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import kotlin.math.hypot
import kotlin.math.max

enum class Limb {
    LEFT_ARM, RIGHT_ARM, LEFT_FOREARM, RIGHT_FOREARM,
    LEFT_THIGH, RIGHT_THIGH, LEFT_CALF, RIGHT_CALF,
    TORSO
}

class HuaweiPoseRenderer {

    // Semi-transparent base fill for the silhouette body
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#3BFFFFFF") // Soft white translucent fill
    }

    // Outer glow stroke for the pose guide
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        color = Color.WHITE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        setShadowLayer(15f, 0f, 0f, Color.parseColor("#B3FFFFFF")) // Glow
    }

    // High-visibility stroke if pose is incorrect
    private val incorrectStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        color = Color.parseColor("#FF5252")
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        setShadowLayer(15f, 0f, 0f, Color.parseColor("#99FF5252"))
    }

    fun render(
        canvas: Canvas,
        landmarks: List<PointF>,
        incorrectLimbs: Set<Limb> = emptySet(),
        imageWidth: Int,
        imageHeight: Int,
        scaleFactor: Float,
        postTranslateX: Float,
        postTranslateY: Float,
        viewWidth: Int,
        viewHeight: Int,
        isPixelCoordinates: Boolean
    ) {
        if (landmarks.size < 29) return

        // 1. Convert ALL normalized landmarks to View Pixel Coordinates FIRST
        val screenPoints = landmarks.map { lm ->
            if (isPixelCoordinates) {
                PointF(
                    lm.x * scaleFactor + postTranslateX,
                    lm.y * scaleFactor + postTranslateY
                )
            } else {
                PointF(
                    lm.x * imageWidth * scaleFactor + postTranslateX,
                    lm.y * imageHeight * scaleFactor + postTranslateY
                )
            }
        }

        // Key landmark extractions
        val nose = screenPoints[0]
        val shoulderL = screenPoints[11]
        val shoulderR = screenPoints[12]
        val elbowL = screenPoints[13]
        val elbowR = screenPoints[14]
        val wristL = screenPoints[15]
        val wristR = screenPoints[16]
        val hipL = screenPoints[23]
        val hipR = screenPoints[24]
        val kneeL = screenPoints[25]
        val kneeR = screenPoints[26]
        val ankleL = screenPoints[27]
        val ankleR = screenPoints[28]

        // 2. Compute dynamic thickness with an enforced MINIMUM floor (120px)
        val shoulderWidth = hypot(
            (shoulderR.x - shoulderL.x).toDouble(),
            (shoulderR.y - shoulderL.y).toDouble()
        ).toFloat()

        val minThickness = 120f // Guarantee minimum thickness in pixels
        val limbThickness = max(shoulderWidth * 0.38f, minThickness)
        val torsoThickness = max(shoulderWidth * 0.45f, minThickness * 1.25f)
        val headRadius = max(shoulderWidth * 0.45f, 130f)

        // 3. Construct individual geometric shapes
        val fullBodyPath = Path()

        // --- HEAD ---
        val headCenterY = nose.y - (headRadius * 0.15f)
        val headPath = Path().apply {
            addCircle(nose.x, headCenterY, headRadius, Path.Direction.CW)
        }
        fullBodyPath.op(headPath, Path.Op.UNION)

        // --- TORSO ---
        val torsoQuad = Path().apply {
            moveTo(shoulderL.x, shoulderL.y)
            lineTo(shoulderR.x, shoulderR.y)
            lineTo(hipR.x, hipR.y)
            lineTo(hipL.x, hipL.y)
            close()
        }
        fullBodyPath.op(torsoQuad, Path.Op.UNION)

        // Torso edge padding for seamless limb connection
        val torsoEdges = listOf(
            Pair(shoulderL, shoulderR),
            Pair(shoulderR, hipR),
            Pair(hipR, hipL),
            Pair(hipL, shoulderL)
        )
        for ((p1, p2) in torsoEdges) {
            fullBodyPath.op(createCapsulePath(p1, p2, torsoThickness), Path.Op.UNION)
        }

        // --- ARMS & LEGS (Separate Capsules) ---
        val limbSegments = listOf(
            Pair(shoulderL, elbowL), // Left Upper Arm
            Pair(elbowL, wristL),    // Left Forearm
            Pair(shoulderR, elbowR), // Right Upper Arm
            Pair(elbowR, wristR),    // Right Forearm
            Pair(hipL, kneeL),       // Left Thigh
            Pair(kneeL, ankleL),     // Left Calf
            Pair(hipR, kneeR),       // Right Thigh
            Pair(kneeR, ankleR)      // Right Calf
        )

        for ((p1, p2) in limbSegments) {
            val capsule = createCapsulePath(p1, p2, limbThickness)
            fullBodyPath.op(capsule, Path.Op.UNION)
        }

        // 4. Render merged body silhouette onto Canvas
        canvas.drawPath(fullBodyPath, fillPaint)

        val activeStroke = if (incorrectLimbs.isNotEmpty()) incorrectStrokePaint else strokePaint
        canvas.drawPath(fullBodyPath, activeStroke)
    }

    /**
     * Builds a thick closed capsule path (rectangle with rounded end-caps)
     * connecting two landmark points in pixel space.
     */
    private fun createCapsulePath(p1: PointF, p2: PointF, thickness: Float): Path {
        val path = Path()
        val radius = thickness / 2f

        // Round joints at endpoints
        path.addCircle(p1.x, p1.y, radius, Path.Direction.CW)
        path.addCircle(p2.x, p2.y, radius, Path.Direction.CW)

        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        val len = hypot(dx.toDouble(), dy.toDouble()).toFloat()

        if (len > 0.001f) {
            // Calculate perpendicular normal vector scaled by radius
            val nx = -dy / len * radius
            val ny = dx / len * radius

            val rectPath = Path().apply {
                moveTo(p1.x + nx, p1.y + ny)
                lineTo(p2.x + nx, p2.y + ny)
                lineTo(p2.x - nx, p2.y - ny)
                lineTo(p1.x - nx, p1.y - ny)
                close()
            }
            path.op(rectPath, Path.Op.UNION)
        }

        return path
    }

    fun reset() {}
    fun release() {}
}
