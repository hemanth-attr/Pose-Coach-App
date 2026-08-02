package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var scaleFactor: Float = 1f
    private var postTranslateX: Float = 0f
    private var postTranslateY: Float = 0f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

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
        results = null
        ProceduralBodyBuilder.reset()
        invalidate()
    }

    fun clearLivePose() {
        results = null
        ProceduralBodyBuilder.reset()
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // ==========================================
        // DRAW THE PROCEDURAL HUMAN BODY OUTLINE
        // ==========================================
        results?.let { poseLandmarkerResult ->
            if (poseLandmarkerResult.landmarks().isNotEmpty()) {
                val landmarks = poseLandmarkerResult.landmarks()[0]
                
                try {
                    val bodyPath = ProceduralBodyBuilder.buildBodyPath(
                        landmarks = landmarks,
                        imageWidth = imageWidth,
                        imageHeight = imageHeight,
                        scaleFactor = scaleFactor,
                        postTranslateX = postTranslateX,
                        postTranslateY = postTranslateY
                    )
                    
                    // Single continuous white outline around the procedural body
                    canvas.drawPath(bodyPath, outlinePaint)
                } catch (e: Exception) {
                    // Failsafe
                }
            }
        }
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode
    ) {
        this.results = poseLandmarkerResults
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