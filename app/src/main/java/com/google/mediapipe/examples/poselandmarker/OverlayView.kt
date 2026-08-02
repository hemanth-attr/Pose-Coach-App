package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var scaleFactor: Float = 1f
    private var postTranslateX: Float = 0f
    private var postTranslateY: Float = 0f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // --- SEGMENTATION MASK (thread-safe copy) ---
    private var segmentationMask: FloatArray? = null
    private var maskWidth: Int = 0
    private var maskHeight: Int = 0

    // --- PREMIUM PAINT OBJECTS (reuse to avoid GC) ---
    private val outlinePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
        color = Color.WHITE
        strokeWidth = 8f
        // Optional subtle dark shadow so the white line is visible on bright backgrounds
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    init {
        // Required for setShadowLayer glow to render on hardware-accelerated views
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    fun clear() {
        segmentationMask = null
        invalidate()
    }

    fun clearLivePose() {
        invalidate()
    }

    /**
     * Receives mask data and copies it into our own array.
     * This prevents FloatBuffer lifecycle issues where MediaPipe
     * recycles the buffer before we read it.
     */
    fun setSegmentationMask(mask: FloatBuffer, width: Int, height: Int) {
        val size = width * height
        mask.rewind()
        if (segmentationMask == null || segmentationMask!!.size != size) {
            segmentationMask = FloatArray(size)
        }
        mask.get(segmentationMask!!)
        maskWidth = width
        maskHeight = height
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // ==========================================
        // 1. DRAW THE DYNAMIC HUMAN SILHOUETTE OUTLINE
        // ==========================================
        segmentationMask?.let { maskData ->
            try {
                // Wrap our owned FloatArray into a temporary FloatBuffer for ContourHelper
                val tempBuffer = FloatBuffer.wrap(maskData)

                val boundaryPoints = ContourHelper.extractSmoothContour(
                    maskBuffer = tempBuffer,
                    maskWidth = maskWidth,
                    maskHeight = maskHeight,
                    imageWidth = imageWidth,
                    imageHeight = imageHeight,
                    scaleFactor = scaleFactor,
                    postTranslateX = postTranslateX,
                    postTranslateY = postTranslateY
                )

                if (boundaryPoints.size >= 3) {
                    val silhouettePath = ContourHelper.createSplinePath(boundaryPoints, tension = 0.25f)
                    
                    // Single continuous white outline around the person
                    canvas.drawPath(silhouettePath, outlinePaint)
                }
            } catch (e: Exception) {
                // Failsafe
            }
        }
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode
    ) {
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        // Calculate CENTER_CROP scaling properly
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
        
        // This translation perfectly centers the cropped camera image inside the view
        postTranslateX = (width - scaledWidth) / 2f
        postTranslateY = (height - scaledHeight) / 2f
        
        invalidate()
    }
}