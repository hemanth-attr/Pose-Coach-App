package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Path
import android.graphics.PointF
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Holds the generated paths for rendering the body model.
 * [mainPath] is the unified correct body.
 * [incorrectPaths] are individual paths for limbs that deviate from the target pose.
 */
data class BodyRenderModel(
    val mainPath: Path,
    val incorrectPaths: List<Path>
)

/**
 * MediaPipe Pose Landmark indices.
 * These constants map to the 33-point MediaPipe Pose model.
 */
object LandmarkIndex {
    const val NOSE = 0
    const val LEFT_EYE_INNER = 1
    const val LEFT_EYE = 2
    const val LEFT_EYE_OUTER = 3
    const val RIGHT_EYE_INNER = 4
    const val RIGHT_EYE = 5
    const val RIGHT_EYE_OUTER = 6
    const val LEFT_EAR = 7
    const val RIGHT_EAR = 8
    const val MOUTH_LEFT = 9
    const val MOUTH_RIGHT = 10
    const val LEFT_SHOULDER = 11
    const val RIGHT_SHOULDER = 12
    const val LEFT_ELBOW = 13
    const val RIGHT_ELBOW = 14
    const val LEFT_WRIST = 15
    const val RIGHT_WRIST = 16
    const val LEFT_PINKY = 17
    const val RIGHT_PINKY = 18
    const val LEFT_INDEX = 19
    const val RIGHT_INDEX = 20
    const val LEFT_THUMB = 21
    const val RIGHT_THUMB = 22
    const val LEFT_HIP = 23
    const val RIGHT_HIP = 24
    const val LEFT_KNEE = 25
    const val RIGHT_KNEE = 26
    const val LEFT_ANKLE = 27
    const val RIGHT_ANKLE = 28
    const val LEFT_HEEL = 29
    const val RIGHT_HEEL = 30
    const val LEFT_FOOT_INDEX = 31
    const val RIGHT_FOOT_INDEX = 32
}

/**
 * Anatomical proportions for the procedural body model.
 * All widths are expressed as ratios relative to the measured shoulder width.
 * These values are tuned to approximate a natural human silhouette.
 */
object AnatomyRatios {
    // ── Body Part Width Ratios (relative to shoulder width) ──
    const val HEAD_RADIUS_FROM_EARS = 0.85f      // Head radius from ear distance
    const val HEAD_RADIUS_FALLBACK = 0.48f       // Fallback head radius when side-profile
    const val HEAD_MIN_RATIO = 0.35f             // Minimum head radius relative to base

    const val NECK_WIDTH = 0.22f                 // Neck capsule width
    const val NECK_LENGTH_RATIO = 0.3f           // How far down the neck extends from head center

    const val TORSO_WIDTH = 0.52f                // Main torso capsule width
    const val TORSO_FILL_INSET = 0.15f           // Inset for the torso fill quad

    const val UPPER_ARM_WIDTH = 0.20f            // Shoulder → Elbow width
    const val FOREARM_WIDTH = 0.16f              // Elbow → Wrist width
    const val HAND_RADIUS = 0.10f                // Hand circle radius

    const val THIGH_WIDTH = 0.24f                // Hip → Knee width
    const val CALF_WIDTH = 0.18f                 // Knee → Ankle width
    const val FOOT_WIDTH = 0.14f                 // Foot capsule width

    // ── Minimum shoulder width to prevent degenerate geometry ──
    const val MIN_SHOULDER_WIDTH_PX = 40f
}

/**
 * Utility functions for 2D geometry operations used throughout the body generator.
 */
object Geometry {
    /**
     * Compute the Euclidean distance between two points.
     */
    fun distance(a: PointF, b: PointF): Float {
        return hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble()).toFloat()
    }

    /**
     * Compute the midpoint between two points.
     */
    fun midpoint(a: PointF, b: PointF): PointF {
        return PointF((a.x + b.x) / 2f, (a.y + b.y) / 2f)
    }

    /**
     * Compute the angle (in radians) from point a to point b.
     */
    fun angle(a: PointF, b: PointF): Float {
        return atan2((b.y - a.y).toDouble(), (b.x - a.x).toDouble()).toFloat()
    }

    /**
     * Linearly interpolate between two points.
     * @param t Interpolation factor [0, 1]. 0 = a, 1 = b.
     */
    fun lerp(a: PointF, b: PointF, t: Float): PointF {
        return PointF(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t
        )
    }

    /**
     * Compute a point offset perpendicular to the direction from a to b.
     * @param center  The center point to offset from
     * @param dir     The direction point (defines the line direction)
     * @param offset  Distance to offset (positive = left side, negative = right side)
     */
    fun perpendicularOffset(center: PointF, dir: PointF, offset: Float): PointF {
        val angle = angle(center, dir)
        val perpAngle = angle + (Math.PI / 2.0).toFloat()
        return PointF(
            center.x + offset * kotlin.math.cos(perpAngle.toDouble()).toFloat(),
            center.y + offset * kotlin.math.sin(perpAngle.toDouble()).toFloat()
        )
    }
}
