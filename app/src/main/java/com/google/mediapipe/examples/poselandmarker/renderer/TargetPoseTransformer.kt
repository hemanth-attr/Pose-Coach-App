package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.PointF
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

object TargetPoseTransformer {

    /**
     * Transforms a target pose (from JSON) to be perfectly centered and uniformly scaled
     * on the screen. This creates the "Huawei AI Pose Template" effect where the
     * template is static and the user must step into it.
     *
     * @param targetPose The target pose from the database (normalized coordinates [0,1]).
     * @param viewWidth The width of the view to render into.
     * @param viewHeight The height of the view to render into.
     * @return A list of transformed target points in VIEW PIXEL coordinates.
     */
    fun transformToStaticTemplate(
        targetPose: List<PointF>,
        viewWidth: Int,
        viewHeight: Int
    ): List<PointF> {
        if (targetPose.isEmpty() || viewWidth == 0 || viewHeight == 0) return targetPose

        // 1. Isotropic Space
        // The target pose points from JSON are already in an isotropic space (proportional 1:1).
        // We do NOT need to artificially multiply X by the camera's aspect ratio, as this will squash it.
        val isotropicPoints = targetPose

        // 2. Find the structural center of the body (Torso Center)
        // This creates a much more premium and stable framing than a simple bounding box,
        // because it keeps the body anchored in the center even if an arm is raised high.
        val midShoulderX = (isotropicPoints[11].x + isotropicPoints[12].x) / 2f
        val midShoulderY = (isotropicPoints[11].y + isotropicPoints[12].y) / 2f
        
        val midHipX = (isotropicPoints[23].x + isotropicPoints[24].x) / 2f
        val midHipY = (isotropicPoints[23].y + isotropicPoints[24].y) / 2f

        val structuralCenterX = (midShoulderX + midHipX) / 2f
        val structuralCenterY = (midShoulderY + midHipY) / 2f

        // 3. Find the maximum extent from the structural center to ensure nothing gets cut off
        var maxDistX = 0f
        var maxDistY = 0f

        for (pt in isotropicPoints) {
            val distX = Math.abs(pt.x - structuralCenterX)
            val distY = Math.abs(pt.y - structuralCenterY)
            if (distX > maxDistX) maxDistX = distX
            if (distY > maxDistY) maxDistY = distY
        }

        // 4. Matrix-based Uniform Projection
        val matrix = android.graphics.Matrix()
        
        // Translate structural torso center to origin
        matrix.postTranslate(-structuralCenterX, -structuralCenterY)
        
        // We want the maximum extent of the pose to comfortably fit within 85% of the screen
        val targetScreenHeight = viewHeight * 0.85f
        val targetScreenWidth = viewWidth * 0.85f
        
        // Total required width/height to contain the furthest points on both sides symmetrically
        val requiredNormHeight = maxDistY * 2f
        val requiredNormWidth = maxDistX * 2f
        
        // Use uniform scaling to completely prevent squishing/stretching
        val scaleY = if (requiredNormHeight > 0) targetScreenHeight / requiredNormHeight else 1f
        val scaleX = if (requiredNormWidth > 0) targetScreenWidth / requiredNormWidth else 1f
        val uniformScale = minOf(scaleX, scaleY)
        
        matrix.postScale(uniformScale, uniformScale)
        
        // Translate back to the exact center of the screen
        matrix.postTranslate(viewWidth / 2f, viewHeight / 2f)
        
        // 4. Apply matrix to all points
        val transformedPose = mutableListOf<PointF>()
        val floatArray = FloatArray(2)
        
        for (pt in isotropicPoints) {
            floatArray[0] = pt.x
            floatArray[1] = pt.y
            matrix.mapPoints(floatArray)
            transformedPose.add(PointF(floatArray[0], floatArray[1]))
        }

        return transformedPose
    }
}
