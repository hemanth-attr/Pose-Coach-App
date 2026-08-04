package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.PointF
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Applies 2D skeletal deformation to a list of vector contour points using Linear Blend Skinning.
 * This preserves the exact organic shape of the user's body outline while bending it to new poses.
 */
class VectorSkinningEngine(
    private val restPoints: List<PointF>,
    private val sourceLandmarks: List<PointF>,
    private val viewWidth: Int,
    private val viewHeight: Int,
    private val imageWidth: Int,
    private val imageHeight: Int
) {
    private val numVertices = restPoints.size
    
    // Bone definitions: pair of MediaPipe landmark indices
    private val bones = listOf(
        Pair(11, 12), // Shoulders
        Pair(23, 24), // Hips
        Pair(11, 23), // Left Torso
        Pair(12, 24), // Right Torso
        Pair(11, 13), // Left Upper Arm
        Pair(13, 15), // Left Forearm
        Pair(12, 14), // Right Upper Arm
        Pair(14, 16), // Right Forearm
        Pair(23, 25), // Left Thigh
        Pair(25, 27), // Left Calf
        Pair(24, 26), // Right Thigh
        Pair(26, 28)  // Right Calf
    )

    // vertexWeights[v][b] = weight of bone b on vertex v
    private val vertexWeights = Array(numVertices) { FloatArray(bones.size) }

    private val scaleFactor = Math.max(viewWidth * 1f / imageWidth, viewHeight * 1f / imageHeight)
    private val postTranslateX = (viewWidth - imageWidth * scaleFactor) / 2f
    private val postTranslateY = (viewHeight - imageHeight * scaleFactor) / 2f

    private fun normalizedToView(normPt: PointF): PointF {
        return PointF(
            (normPt.x * imageWidth) * scaleFactor + postTranslateX,
            (normPt.y * imageHeight) * scaleFactor + postTranslateY
        )
    }

    init {
        calculateBoneWeights()
    }

    private fun calculateBoneWeights() {
        // Simple Inverse Distance Weighting (IDW) to bones
        for (v in 0 until numVertices) {
            val vx = restPoints[v].x
            val vy = restPoints[v].y
            
            var totalWeight = 0f
            val weights = FloatArray(bones.size)
            
            for (b in bones.indices) {
                val bone = bones[b]
                val p1 = normalizedToView(sourceLandmarks[bone.first])
                val p2 = normalizedToView(sourceLandmarks[bone.second])
                
                val p1x = p1.x
                val p1y = p1.y
                val p2x = p2.x
                val p2y = p2.y
                
                val dist = distanceToSegment(vx, vy, p1x, p1y, p2x, p2y)
                
                // Weight = 1 / (d^4 + epsilon) for sharp falloff
                val w = 1f / (dist * dist * dist * dist + 1f)
                weights[b] = w
                totalWeight += w
            }
            
            // Normalize weights
            for (b in bones.indices) {
                vertexWeights[v][b] = weights[b] / totalWeight
            }
        }
    }

    /**
     * Deforms the rest points according to the new target landmarks.
     * @param targetLandmarks The new pose landmarks (in normalized coordinates).
     * @return A new list of PointF representing the deformed vector contour (in pixel space).
     */
    fun deform(targetLandmarks: List<PointF>): List<PointF> {
        if (targetLandmarks.size < 33 || restPoints.isEmpty()) return restPoints

        // 1. Calculate transformation for each bone
        val boneTransforms = Array(bones.size) { BoneTransform() }
        
        for (b in bones.indices) {
            val bone = bones[b]
            val src1 = normalizedToView(sourceLandmarks[bone.first])
            val src2 = normalizedToView(sourceLandmarks[bone.second])
            val tgt1 = targetLandmarks[bone.first]
            val tgt2 = targetLandmarks[bone.second]
            
            // Convert to pixel space (source is normalized, target is ALREADY in pixels)
            val s1x = src1.x; val s1y = src1.y
            val s2x = src2.x; val s2y = src2.y
            
            val t1x = tgt1.x; val t1y = tgt1.y
            val t2x = tgt2.x; val t2y = tgt2.y
            
            // Source angle and length
            val srcAngle = atan2(s2y - s1y, s2x - s1x)
            val srcLen = sqrt((s2x - s1x) * (s2x - s1x) + (s2y - s1y) * (s2y - s1y))
            
            // Target angle and length
            val tgtAngle = atan2(t2y - t1y, t2x - t1x)
            val tgtLen = sqrt((t2x - t1x) * (t2x - t1x) + (t2y - t1y) * (t2y - t1y))
            
            // Transform parameters
            val deltaAngle = tgtAngle - srcAngle
            val scale = if (srcLen > 0.001f) tgtLen / srcLen else 1f
            
            boneTransforms[b] = BoneTransform(
                originX = s1x,
                originY = s1y,
                targetX = t1x,
                targetY = t1y,
                angle = deltaAngle,
                scale = scale
            )
        }

        // 2. Apply Linear Blend Skinning to all vertices
        val deformedPoints = mutableListOf<PointF>()
        for (v in 0 until numVertices) {
            val rx = restPoints[v].x
            val ry = restPoints[v].y
            
            var finalX = 0f
            var finalY = 0f
            
            for (b in bones.indices) {
                val weight = vertexWeights[v][b]
                if (weight > 0.001f) {
                    val transform = boneTransforms[b]
                    
                    // Move to bone origin
                    val dx = rx - transform.originX
                    val dy = ry - transform.originY
                    
                    // Rotate and scale
                    val cosA = cos(transform.angle)
                    val sinA = sin(transform.angle)
                    val scaledDx = dx * transform.scale
                    val scaledDy = dy * transform.scale
                    
                    val rotX = scaledDx * cosA - scaledDy * sinA
                    val rotY = scaledDx * sinA + scaledDy * cosA
                    
                    // Move to target bone origin
                    val bx = rotX + transform.targetX
                    val by = rotY + transform.targetY
                    
                    finalX += bx * weight
                    finalY += by * weight
                }
            }
            
            deformedPoints.add(PointF(finalX, finalY))
        }
        
        return deformedPoints
    }

    // Helper: shortest distance from point to line segment
    private fun distanceToSegment(px: Float, py: Float, vx: Float, vy: Float, wx: Float, wy: Float): Float {
        val l2 = (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy)
        if (l2 == 0f) return sqrt((px - vx) * (px - vx) + (py - vy) * (py - vy))
        var t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2
        t = Math.max(0f, Math.min(1f, t))
        val projX = vx + t * (wx - vx)
        val projY = vy + t * (wy - vy)
        return sqrt((px - projX) * (px - projX) + (py - projY) * (py - projY))
    }

    private data class BoneTransform(
        val originX: Float = 0f,
        val originY: Float = 0f,
        val targetX: Float = 0f,
        val targetY: Float = 0f,
        val angle: Float = 0f,
        val scale: Float = 1f
    )
}
