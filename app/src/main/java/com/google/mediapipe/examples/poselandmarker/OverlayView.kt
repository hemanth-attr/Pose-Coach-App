package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.examples.poselandmarker.renderer.HuaweiPoseRenderer
import com.google.mediapipe.examples.poselandmarker.renderer.Limb
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Overlay view that implements Skeletal Proportional Mapping.
 * It dynamically scales and retargets the saved target pose (angles) 
 * onto the live user's body size (proportions), rendering it using the Premium Huawei Renderer.
 */
class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    // ── Pose detection results ──
    private var targetResults: List<PointF>? = null
    private var liveResults: List<PointF>? = null
    private var incorrectLimbs: Set<Limb> = emptySet()
    
    // ── Coordinate Mapping ──
    private var scaleFactor: Float = 1f
    private var postTranslateX: Float = 0f
    private var postTranslateY: Float = 0f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var isPixelCoordinates: Boolean = false

    // ── Auto-Adapt Smoothing State (Cinematic Camera) ──
    private var smoothedScale: Float = -1f
    private var smoothedCX: Float = -1f
    private var smoothedCY: Float = -1f

    // ── Premium Renderer ──
    private val renderer = HuaweiPoseRenderer()

    init {
        // Required for setShadowLayer glow to render properly
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    /**
     * Clear all results and reset the tracking state.
     */
    fun clear() {
        targetResults = null
        liveResults = null
        incorrectLimbs = emptySet()
        smoothedScale = -1f
        renderer.reset()
        invalidate()
    }

    /**
     * Clear live pose results without fully resetting the renderer.
     */
    fun clearLivePose() {
        liveResults = null
        incorrectLimbs = emptySet()
        smoothedScale = -1f // Reset smoothing so it doesn't snap if they walk back in
        renderer.reset()
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        targetResults?.let { targetPose ->
            if (targetPose.size >= 29) {
                try {
                    // 1. Convert Target Pose to Screen Coordinates
                    val screenTarget = targetPose.map { lm ->
                        val px = if (isPixelCoordinates) lm.x * scaleFactor + postTranslateX else lm.x * imageWidth * scaleFactor + postTranslateX
                        val py = if (isPixelCoordinates) lm.y * scaleFactor + postTranslateY else lm.y * imageHeight * scaleFactor + postTranslateY
                        PointF(px, py)
                    }

                    val finalPointsToRender: List<PointF>

                    if (liveResults != null && liveResults!!.size >= 29) {
                        // ========================================================
                        // SKELETAL RETARGETING: Match Target Pose to Live User!
                        // ========================================================
                        
                        // Convert Live Pose to Screen Coordinates
                        val screenLive = liveResults!!.map { lm ->
                            val px = if (isPixelCoordinates) lm.x * scaleFactor + postTranslateX else lm.x * imageWidth * scaleFactor + postTranslateX
                            val py = if (isPixelCoordinates) lm.y * scaleFactor + postTranslateY else lm.y * imageHeight * scaleFactor + postTranslateY
                            PointF(px, py)
                        }

                        // Calculate LIVE Torso bounds (Center & Height)
                        val liveShoulder = PointF((screenLive[11].x + screenLive[12].x) / 2f, (screenLive[11].y + screenLive[12].y) / 2f)
                        val liveHip = PointF((screenLive[23].x + screenLive[24].x) / 2f, (screenLive[23].y + screenLive[24].y) / 2f)
                        val liveTorsoHeight = Math.max(1f, hypot(liveShoulder.x - liveHip.x, liveShoulder.y - liveHip.y))
                        val liveCenter = PointF((liveShoulder.x + liveHip.x) / 2f, (liveShoulder.y + liveHip.y) / 2f)

                        // Calculate TARGET Torso bounds
                        val targetShoulder = PointF((screenTarget[11].x + screenTarget[12].x) / 2f, (screenTarget[11].y + screenTarget[12].y) / 2f)
                        val targetHip = PointF((screenTarget[23].x + screenTarget[24].x) / 2f, (screenTarget[23].y + screenTarget[24].y) / 2f)
                        val targetTorsoHeight = Math.max(1f, hypot(targetShoulder.x - targetHip.x, targetShoulder.y - targetHip.y))
                        val targetCenter = PointF((targetShoulder.x + targetHip.x) / 2f, (targetShoulder.y + targetHip.y) / 2f)

                        // Calculate exact mathematical scale needed to make the pose fit the user's real body
                        val rawScale = liveTorsoHeight / targetTorsoHeight

                        // APPLY LOW-PASS FILTER (Buttery Smooth Gliding)
                        if (smoothedScale < 0f) {
                            smoothedScale = rawScale
                            smoothedCX = liveCenter.x
                            smoothedCY = liveCenter.y
                        } else {
                            val alpha = 0.15f // Blends motion seamlessly
                            smoothedScale = (smoothedScale * (1f - alpha)) + (rawScale * alpha)
                            smoothedCX = (smoothedCX * (1f - alpha)) + (liveCenter.x * alpha)
                            smoothedCY = (smoothedCY * (1f - alpha)) + (liveCenter.y * alpha)
                        }

                        // Map Target points onto Live body position/scale
                        finalPointsToRender = screenTarget.map { pt ->
                            val dx = pt.x - targetCenter.x
                            val dy = pt.y - targetCenter.y
                            PointF(
                                smoothedCX + dx * smoothedScale,
                                smoothedCY + dy * smoothedScale
                            )
                        }
                        
                        // Keep rendering to allow smooth animation tracking
                        invalidate() 

                    } else {
                        // ========================================================
                        // FALLBACK: User stepped out of frame (Draw static center)
                        // ========================================================
                        smoothedScale = -1f

                        val targetShoulder = PointF((screenTarget[11].x + screenTarget[12].x) / 2f, (screenTarget[11].y + screenTarget[12].y) / 2f)
                        val targetHip = PointF((screenTarget[23].x + screenTarget[24].x) / 2f, (screenTarget[23].y + screenTarget[24].y) / 2f)
                        val targetTorsoHeight = Math.max(1f, hypot(targetShoulder.x - targetHip.x, targetShoulder.y - targetHip.y))
                        val targetCenter = PointF((targetShoulder.x + targetHip.x) / 2f, (targetShoulder.y + targetHip.y) / 2f)

                        // Safely default it to center of screen, scaled to 30% of screen height
                        val staticScale = (height * 0.3f) / targetTorsoHeight
                        val viewCenterX = width / 2f
                        val viewCenterY = height / 2f

                        finalPointsToRender = screenTarget.map { pt ->
                            val dx = pt.x - targetCenter.x
                            val dy = pt.y - targetCenter.y
                            PointF(
                                viewCenterX + dx * staticScale,
                                viewCenterY + dy * staticScale
                            )
                        }
                    }

                    // 2. Feed the proportionally-perfect skeleton to the Huawei Renderer!
                    renderer.render(
                        canvas = canvas,
                        landmarks = finalPointsToRender,
                        incorrectLimbs = incorrectLimbs,
                        imageWidth = width,
                        imageHeight = height,
                        scaleFactor = 1f,        // Scale/translation is already applied!
                        postTranslateX = 0f,     
                        postTranslateY = 0f,
                        viewWidth = width,
                        viewHeight = height,
                        isPixelCoordinates = true // Tell renderer to just draw the exact pixels
                    )

                } catch (e: Exception) {
                    // Failsafe — never crash the UI thread
                }
            }
        }
    }

    /**
     * Update results and coordinate mapping from the detection pipeline.
     */
    fun setCoachResults(
        transformedTarget: List<PointF>?,
        liveLandmarks: List<PointF>?,
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

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> min(width * 1f / imageWidth, height * 1f / imageHeight)
            RunningMode.LIVE_STREAM -> max(width * 1f / imageWidth, height * 1f / imageHeight)
        }

        val scaledWidth = imageWidth * scaleFactor
        val scaledHeight = imageHeight * scaleFactor
        postTranslateX = (width - scaledWidth) / 2f
        postTranslateY = (height - scaledHeight) / 2f
        invalidate()
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode
    ) {
        if (poseLandmarkerResults.landmarks().isNotEmpty()) {
            val liveLandmarks = poseLandmarkerResults.landmarks()[0]
            val pointFList = liveLandmarks.map { lm -> PointF(lm.x(), lm.y()) }
            setCoachResults(pointFList, pointFList, emptySet(), imageHeight, imageWidth, runningMode)
        } else {
            clearLivePose()
        }
    }

    // Keep this empty so CameraFragment doesn't throw compiler errors!
    fun setCustomSilhouette(path: android.graphics.Path?) {
        // Custom 2D paths are obsolete. We use 3D Skeletal Retargeting now.
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        renderer.release()
    }
}