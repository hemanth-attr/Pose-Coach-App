package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.examples.poselandmarker.renderer.HuaweiPoseRenderer
import com.google.mediapipe.examples.poselandmarker.renderer.Limb
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min

/**
 * Overlay view that renders the Huawei AI Pose-style premium outline
 * on top of the camera preview.
 *
 * This view is a thin wrapper that delegates all rendering logic to
 * [HuaweiPoseRenderer]. It handles coordinate system conversion
 * (MediaPipe normalized coords → view pixel coords) and lifecycle.
 */
class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    // ── Pose detection results ──
    private var targetResults: List<android.graphics.PointF>? = null
    private var liveResults: List<android.graphics.PointF>? = null
    private var incorrectLimbs: Set<Limb> = emptySet()
    private var scaleFactor: Float = 1f
    private var postTranslateX: Float = 0f
    private var postTranslateY: Float = 0f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // ── Premium Renderer ──
    private val renderer = HuaweiPoseRenderer()

    init {
        // Required for setShadowLayer glow to render properly
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    // ── Custom Pose Silhouette ──
    private var customVectorPath: android.graphics.Path? = null

    /**
     * Clear all results and reset the renderer's temporal state.
     * Called when switching modes or stopping the camera.
     */
    fun clear() {
        targetResults = null
        liveResults = null
        incorrectLimbs = emptySet()
        customVectorPath = null
        renderer.reset()
        invalidate()
    }

    /**
     * Clear live pose results without fully resetting the renderer.
     * Called when tracking is temporarily lost.
     */
    fun clearLivePose() {
        liveResults = null
        incorrectLimbs = emptySet()
        // We DO NOT clear targetResults or customVectorPath here, so the template stays on screen!
        renderer.reset()
        invalidate()
    }

    private var isPixelCoordinates: Boolean = false

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // ══════════════════════════════════════════
        // RENDER THE HUAWEI AI POSE PREMIUM OUTLINE
        // ══════════════════════════════════════════
        
        // If we have a custom captured human silhouette, draw it directly!
        if (customVectorPath != null) {
            val paint = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.STROKE
                strokeWidth = 6f
                isAntiAlias = true
                strokeJoin = Paint.Join.ROUND
                strokeCap = Paint.Cap.ROUND
                setShadowLayer(15f, 0f, 0f, Color.parseColor("#99FFFFFF")) // Premium Glow
            }
            
            var pathToDraw = customVectorPath!!
            
            // Auto-Adapt: If we can see the live user, make the template float and stick to their body size!
            if (liveResults != null && liveResults!!.size > 24) {
                // 1. Find live user's torso center in pixel space
                val l = liveResults!!
                val leftShoulder = l[11]
                val rightShoulder = l[12]
                val leftHip = l[23]
                val rightHip = l[24]
                
                val liveShoulderCY = ((leftShoulder.y + rightShoulder.y) / 2f) * imageHeight * scaleFactor + postTranslateY
                val liveHipCY = ((leftHip.y + rightHip.y) / 2f) * imageHeight * scaleFactor + postTranslateY
                val liveCX = ((leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4f) * imageWidth * scaleFactor + postTranslateX
                val liveCY = (liveShoulderCY + liveHipCY) / 2f
                val liveTorsoHeight = Math.abs(liveHipCY - liveShoulderCY)
                
                // 2. Find template's original bounds
                val pathBounds = android.graphics.RectF()
                customVectorPath!!.computeBounds(pathBounds, true)
                // We assume the torso is roughly 35% of the total body bounding box height
                val pathTorsoHeight = pathBounds.height() * 0.35f 
                val pathCenterX = pathBounds.centerX()
                val pathCenterY = pathBounds.centerY()
                
                // 3. Transform!
                if (pathTorsoHeight > 0 && liveTorsoHeight > 0) {
                    val scale = Math.max(0.2f, liveTorsoHeight / pathTorsoHeight)
                    
                    val matrix = android.graphics.Matrix()
                    matrix.postTranslate(-pathCenterX, -pathCenterY) // center at 0,0
                    matrix.postScale(scale, scale) // scale to match user
                    matrix.postTranslate(liveCX, liveCY) // move to user's live center
                    
                    val transformedPath = android.graphics.Path()
                    customVectorPath!!.transform(matrix, transformedPath)
                    pathToDraw = transformedPath
                }
            }

            canvas.drawPath(pathToDraw, paint)
            return
        }

        // ══════════════════════════════════════════
        // 2D PROCEDURAL ANIMATION (HUAWEI AI POSE)
        // ══════════════════════════════════════════

        // Otherwise, fall back to procedural geometry
        targetResults?.let { targetPose ->
            if (targetPose.isNotEmpty()) {
                try {
                    renderer.render(
                        canvas = canvas,
                        landmarks = targetPose,
                        incorrectLimbs = incorrectLimbs,
                        imageWidth = imageWidth,
                        imageHeight = imageHeight,
                        scaleFactor = scaleFactor,
                        postTranslateX = postTranslateX,
                        postTranslateY = postTranslateY,
                        viewWidth = width,
                        viewHeight = height,
                        isPixelCoordinates = isPixelCoordinates
                    )
                } catch (e: Exception) {
                    // Failsafe — never crash the UI thread
                }
            }
        }
    }

    /**
     * Update results and coordinate mapping from the detection pipeline.
     * Called from CameraFragment.onResults() on the UI thread.
     */
    fun setCoachResults(
        transformedTarget: List<android.graphics.PointF>?,
        liveLandmarks: List<android.graphics.PointF>?,
        incorrect: Set<Limb>,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode,
        isPixelCoordinates: Boolean = false
    ) {
        this.targetResults = transformedTarget
        this.liveResults = liveLandmarks
        this.incorrectLimbs = incorrect
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth
        this.isPixelCoordinates = isPixelCoordinates

        // Calculate CENTER_CROP scaling
        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }

        val scaledWidth = imageWidth * scaleFactor
        val scaledHeight = imageHeight * scaleFactor

        // Center the cropped image inside the view
        postTranslateX = (width - scaledWidth) / 2f
        postTranslateY = (height - scaledHeight) / 2f

        invalidate()
    }

    /**
     * Fallback for GalleryFragment: renders the raw user's body instead of the coach target.
     */
    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode
    ) {
        if (poseLandmarkerResults.landmarks().isNotEmpty()) {
            val liveLandmarks = poseLandmarkerResults.landmarks()[0]
            val pointFList = liveLandmarks.map { lm ->
                android.graphics.PointF(lm.x(), lm.y())
            }
            setCoachResults(pointFList, pointFList, emptySet(), imageHeight, imageWidth, runningMode)
        } else {
            clearLivePose()
        }
    }

    /**
     * Set a custom captured human silhouette path.
     */
    fun setCustomSilhouette(path: android.graphics.Path?) {
        this.customVectorPath = path
        invalidate()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        renderer.release()
    }
}