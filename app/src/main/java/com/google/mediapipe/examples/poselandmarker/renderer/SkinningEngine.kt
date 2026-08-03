package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PointF
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class SkinningEngine(
    private val texture: Bitmap,
    private val sourceLandmarks: List<PointF>,
    private val viewWidth: Int,
    private val viewHeight: Int
) {
    // 30x30 mesh = 31x31 vertices
    private val meshWidth = 30
    private val meshHeight = 30
    private val numVertices = (meshWidth + 1) * (meshHeight + 1)
    
    // Original resting vertex positions
    private val restVerts = FloatArray(numVertices * 2)
    // Deformed active vertex positions
    private val activeVerts = FloatArray(numVertices * 2)
    
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

    // For each vertex, store its weight for each bone.
    // vertexWeights[v][b] = weight of bone b on vertex v
    private val vertexWeights = Array(numVertices) { FloatArray(bones.size) }

    init {
        generateMesh()
        calculateBoneWeights()
    }

    private fun generateMesh() {
        val w = texture.width.toFloat()
        val h = texture.height.toFloat()
        var index = 0
        for (y in 0..meshHeight) {
            val fy = h * y / meshHeight
            for (x in 0..meshWidth) {
                val fx = w * x / meshWidth
                restVerts[index * 2] = fx
                restVerts[index * 2 + 1] = fy
                activeVerts[index * 2] = fx
                activeVerts[index * 2 + 1] = fy
                index++
            }
        }
    }

    private fun calculateBoneWeights() {
        // Simple Inverse Distance Weighting (IDW) to bones
        for (v in 0 until numVertices) {
            val vx = restVerts[v * 2]
            val vy = restVerts[v * 2 + 1]
            
            var totalWeight = 0f
            val weights = FloatArray(bones.size)
            
            for (b in bones.indices) {
                val bone = bones[b]
                val p1 = sourceLandmarks[bone.first]
                val p2 = sourceLandmarks[bone.second]
                
                // Convert normalized landmarks to pixel space for distance calculation
                val p1x = p1.x * viewWidth
                val p1y = p1.y * viewHeight
                val p2x = p2.x * viewWidth
                val p2y = p2.y * viewHeight
                
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

    fun deform(targetLandmarks: List<PointF>) {
        if (targetLandmarks.size < 33) return

        // 1. Calculate transformation for each bone
        // Each transform maps a point from rest-space to deformed-space
        val boneTransforms = Array(bones.size) { BoneTransform() }
        
        for (b in bones.indices) {
            val bone = bones[b]
            val src1 = sourceLandmarks[bone.first]
            val src2 = sourceLandmarks[bone.second]
            val tgt1 = targetLandmarks[bone.first]
            val tgt2 = targetLandmarks[bone.second]
            
            // Convert to pixel space (source is normalized, target is ALREADY in pixels)
            val s1x = src1.x * viewWidth; val s1y = src1.y * viewHeight
            val s2x = src2.x * viewWidth; val s2y = src2.y * viewHeight
            
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
        for (v in 0 until numVertices) {
            val rx = restVerts[v * 2]
            val ry = restVerts[v * 2 + 1]
            
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
            
            activeVerts[v * 2] = finalX
            activeVerts[v * 2 + 1] = finalY
        }
    }

    fun draw(canvas: Canvas, paint: Paint) {
        canvas.drawBitmapMesh(texture, meshWidth, meshHeight, activeVerts, 0, null, 0, paint)
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
