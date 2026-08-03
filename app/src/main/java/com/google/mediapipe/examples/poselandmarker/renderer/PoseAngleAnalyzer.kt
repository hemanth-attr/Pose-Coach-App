package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.PointF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2

enum class Limb {
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_LEG,
    RIGHT_LEG
}

object PoseAngleAnalyzer {

    // Tolerance in radians (e.g., 25 degrees)
    private const val ANGLE_TOLERANCE = 25.0 * (PI / 180.0)

    /**
     * Compares the live user's limb angles to the transformed target pose.
     * Returns a set of limbs that are significantly incorrect.
     */
    fun analyze(
        liveLandmarks: List<NormalizedLandmark>,
        transformedTarget: List<PointF>
    ): Set<Limb> {
        val incorrectLimbs = mutableSetOf<Limb>()
        
        if (liveLandmarks.size < 33 || transformedTarget.size < 33) return incorrectLimbs

        // Helper to get angle of a segment in the live pose
        fun getLiveAngle(p1: Int, p2: Int): Float {
            return atan2(
                liveLandmarks[p2].y() - liveLandmarks[p1].y(),
                liveLandmarks[p2].x() - liveLandmarks[p1].x()
            )
        }

        // Helper to get angle of a segment in the target pose
        fun getTargetAngle(p1: Int, p2: Int): Float {
            return atan2(
                transformedTarget[p2].y - transformedTarget[p1].y,
                transformedTarget[p2].x - transformedTarget[p1].x
            )
        }

        // Helper to get shortest angular distance
        fun angleDiff(a1: Float, a2: Float): Double {
            var diff = abs(a1 - a2).toDouble()
            if (diff > PI) {
                diff = 2.0 * PI - diff
            }
            return diff
        }

        // Check Left Arm
        val leftUpperArmDiff = angleDiff(getLiveAngle(11, 13), getTargetAngle(11, 13))
        val leftForearmDiff = angleDiff(getLiveAngle(13, 15), getTargetAngle(13, 15))
        if (leftUpperArmDiff > ANGLE_TOLERANCE || leftForearmDiff > ANGLE_TOLERANCE) {
            incorrectLimbs.add(Limb.LEFT_ARM)
        }

        // Check Right Arm
        val rightUpperArmDiff = angleDiff(getLiveAngle(12, 14), getTargetAngle(12, 14))
        val rightForearmDiff = angleDiff(getLiveAngle(14, 16), getTargetAngle(14, 16))
        if (rightUpperArmDiff > ANGLE_TOLERANCE || rightForearmDiff > ANGLE_TOLERANCE) {
            incorrectLimbs.add(Limb.RIGHT_ARM)
        }

        // Check Left Leg
        val leftThighDiff = angleDiff(getLiveAngle(23, 25), getTargetAngle(23, 25))
        val leftCalfDiff = angleDiff(getLiveAngle(25, 27), getTargetAngle(25, 27))
        if (leftThighDiff > ANGLE_TOLERANCE || leftCalfDiff > ANGLE_TOLERANCE) {
            incorrectLimbs.add(Limb.LEFT_LEG)
        }

        // Check Right Leg
        val rightThighDiff = angleDiff(getLiveAngle(24, 26), getTargetAngle(24, 26))
        val rightCalfDiff = angleDiff(getLiveAngle(26, 28), getTargetAngle(26, 28))
        if (rightThighDiff > ANGLE_TOLERANCE || rightCalfDiff > ANGLE_TOLERANCE) {
            incorrectLimbs.add(Limb.RIGHT_LEG)
        }

        return incorrectLimbs
    }
}
