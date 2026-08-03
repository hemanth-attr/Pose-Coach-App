package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.PointF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

object TargetPoseTransformer {

    /**
     * Transforms a target pose to fit the live user's proportions and position.
     * 
     * @param targetPose The target pose from the database.
     * @param liveLandmarks The live user's normalized landmarks.
     * @return A list of transformed target points (normalized coordinates).
     */
    fun transform(
        targetPose: List<PointF>,
        liveLandmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): List<PointF> {
        if (targetPose.size < 33 || liveLandmarks.size < 33) return targetPose
        if (imageWidth == 0 || imageHeight == 0) return targetPose

        val aspect = imageWidth.toFloat() / imageHeight.toFloat()

        // 1. Calculate live torso center and length (isotropic space)
        val liveMidHip = PointF(
            ((liveLandmarks[23].x() + liveLandmarks[24].x()) / 2f) * aspect,
            (liveLandmarks[23].y() + liveLandmarks[24].y()) / 2f
        )
        val liveMidShoulder = PointF(
            ((liveLandmarks[11].x() + liveLandmarks[12].x()) / 2f) * aspect,
            (liveLandmarks[11].y() + liveLandmarks[12].y()) / 2f
        )
        val liveTorsoLength = Geometry.distance(liveMidHip, liveMidShoulder)
        val liveTorsoAngle = atan2(
            liveMidShoulder.y - liveMidHip.y,
            liveMidShoulder.x - liveMidHip.x
        )

        // 2. Calculate target torso center and length
        // targetPose was saved using naive (x, y) distance, so we must assume it is already isotropic 
        // relative to the camera that saved it. However, to rotate it cleanly, we just use its raw coordinates.
        val targetMidHip = PointF(
            (targetPose[23].x + targetPose[24].x) / 2f,
            (targetPose[23].y + targetPose[24].y) / 2f
        )
        val targetMidShoulder = PointF(
            (targetPose[11].x + targetPose[12].x) / 2f,
            (targetPose[11].y + targetPose[12].y) / 2f
        )
        val targetTorsoLength = Geometry.distance(targetMidHip, targetMidShoulder)
        val targetTorsoAngle = atan2(
            targetMidShoulder.y - targetMidHip.y,
            targetMidShoulder.x - targetMidHip.x
        )

        // 3. Compute scale and rotation
        // Avoid division by zero
        val scale = if (targetTorsoLength > 0.001f) liveTorsoLength / targetTorsoLength else 1f
        val rotationAngle = liveTorsoAngle - targetTorsoAngle

        val cosA = cos(rotationAngle)
        val sinA = sin(rotationAngle)

        // 4. Transform all target points
        val transformedPose = mutableListOf<PointF>()
        for (pt in targetPose) {
            // Translate relative to target hip
            val dx = pt.x - targetMidHip.x
            val dy = pt.y - targetMidHip.y

            // Rotate
            val rx = dx * cosA - dy * sinA
            val ry = dx * sinA + dy * cosA

            // Scale and translate to live hip (isotropic)
            val finalX_iso = liveMidHip.x + rx * scale
            val finalY = liveMidHip.y + ry * scale
            
            // Convert back to normalized coordinates
            transformedPose.add(PointF(finalX_iso / aspect, finalY))
        }

        return transformedPose
    }
}
