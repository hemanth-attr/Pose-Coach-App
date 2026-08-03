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

    /**
     * Clear all results and reset the renderer's temporal state.
     * Called when switching modes or stopping the camera.
     */
    fun clear() {
        targetResults = null
        incorrectLimbs = emptySet()
        renderer.reset()
        invalidate()
    }

    /**
     * Clear live pose results without fully resetting the renderer.
     * Called when tracking is temporarily lost.
     */
    fun clearLivePose() {
        targetResults = null
        incorrectLimbs = emptySet()
        renderer.reset()
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // ══════════════════════════════════════════
        // RENDER THE HUAWEI AI POSE PREMIUM OUTLINE
        // ══════════════════════════════════════════
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
                        viewHeight = height
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
        incorrect: Set<Limb>,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode
    ) {
        this.targetResults = transformedTarget
        this.incorrectLimbs = incorrect
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

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
            setCoachResults(pointFList, emptySet(), imageHeight, imageWidth, runningMode)
        } else {
            clearLivePose()
        }
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        renderer.release()
    }
}