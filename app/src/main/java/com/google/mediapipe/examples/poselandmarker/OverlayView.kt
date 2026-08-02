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
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var postTranslateX: Float = 0f
    private var postTranslateY: Float = 0f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // --- TARGET POSE VARIABLES ---
    var targetPose: List<PointF>? = null
    var isPoseMatched = false

    // --- SEGMENTATION MASK (thread-safe copy) ---
    private var segmentationMask: FloatArray? = null
    private var maskWidth: Int = 0
    private var maskHeight: Int = 0

    // --- PREMIUM PAINT OBJECTS (reuse to avoid GC) ---
    private val glowPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }
    
    private val corePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    // --- TARGET POSE (NEON WIREFRAME) PAINTS ---
    private val targetGlowPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        strokeWidth = 14f
        isAntiAlias = true
    }
    
    private val targetCorePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        strokeWidth = 4f
        color = Color.WHITE
        isAntiAlias = true
    }
    
    private val targetJointGlowPaint = Paint(targetGlowPaint).apply { style = Paint.Style.FILL }
    private val targetJointCorePaint = Paint(targetCorePaint).apply { style = Paint.Style.FILL }
    
    // Reusable path for target pose to reduce allocations
    private val targetPosePath = Path()

    init {
        initPaints()
        // Required for setShadowLayer glow to render on hardware-accelerated views
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    fun clear() {
        results = null
        segmentationMask = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    fun clearLivePose() {
        results = null
        invalidate()
    }

    private fun initPaints() {
        linePaint.color = ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
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

        val accentColor = if (isPoseMatched) Color.GREEN else Color.WHITE

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

                    // OUTER GLOW STROKE (wider, semi-transparent)
                    glowPaint.color = accentColor
                    glowPaint.strokeWidth = 22f
                    glowPaint.alpha = 60
                    glowPaint.setShadowLayer(20f, 0f, 0f, accentColor)
                    canvas.drawPath(silhouettePath, glowPaint)

                    // CORE STROKE (narrower, fully opaque)
                    corePaint.color = accentColor
                    corePaint.strokeWidth = 6f
                    corePaint.alpha = 255
                    corePaint.setShadowLayer(8f, 0f, 0f, accentColor)
                    canvas.drawPath(silhouettePath, corePaint)
                }
            } catch (e: Exception) {
                // Failsafe
            }
        }

        // ==========================================
        // 2. DRAW THE TARGET POSE GUIDELINE (Neon Wireframe)
        // ==========================================
        targetPose?.let { pose ->
            val centerX = canvas.width / 2f
            val centerY = canvas.height / 2f
            val drawScale = canvas.height / 3.5f

            fun getPoint(idx: Int): PointF {
                return PointF(
                    centerX + (pose[idx].x * drawScale),
                    centerY + (pose[idx].y * drawScale)
                )
            }

            // A nice cyan color when unmatched, glowing green when matched
            val targetColor = if (isPoseMatched) Color.GREEN else Color.parseColor("#00E5FF")
            
            targetGlowPaint.color = targetColor
            targetGlowPaint.alpha = 150
            targetGlowPaint.setShadowLayer(16f, 0f, 0f, targetColor)
            
            targetJointGlowPaint.color = targetColor
            targetJointGlowPaint.alpha = 150
            targetJointGlowPaint.setShadowLayer(12f, 0f, 0f, targetColor)

            targetPosePath.reset()

            // Draw standard MediaPipe skeleton connections
            PoseLandmarker.POSE_LANDMARKS.forEach { connection ->
                val startIdx = connection!!.start()
                val endIdx = connection.end()
                
                // Optional: Skip face landmarks (0-10) for a cleaner "body only" wireframe
                // if (startIdx > 10 && endIdx > 10) {
                val start = getPoint(startIdx)
                val end = getPoint(endIdx)
                targetPosePath.moveTo(start.x, start.y)
                targetPosePath.lineTo(end.x, end.y)
                // }
            }
            
            // Draw the paths (bones)
            canvas.drawPath(targetPosePath, targetGlowPaint)
            canvas.drawPath(targetPosePath, targetCorePaint)
            
            // Draw the joints (nodes)
            for (i in pose.indices) {
                // Optional: Skip face landmarks for a cleaner aesthetic
                // if (i > 10) {
                val p = getPoint(i)
                canvas.drawCircle(p.x, p.y, 8f, targetJointGlowPaint)
                canvas.drawCircle(p.x, p.y, 4f, targetJointCorePaint)
                // }
            }
        }

        // ==========================================
        // 3. FALLBACK: Original MediaPipe skeleton (Gallery mode only)
        // ==========================================
        results?.let { poseLandmarkerResult ->
            if (segmentationMask == null) {
                for (landmark in poseLandmarkerResult.landmarks()) {
                    for (normalizedLandmark in landmark) {
                        val viewX = normalizedLandmark.x() * imageWidth * scaleFactor + postTranslateX
                        val viewY = normalizedLandmark.y() * imageHeight * scaleFactor + postTranslateY
                        canvas.drawPoint(viewX, viewY, pointPaint)
                    }

                    PoseLandmarker.POSE_LANDMARKS.forEach {
                        val startL = poseLandmarkerResult.landmarks().get(0).get(it!!.start())
                        val endL = poseLandmarkerResult.landmarks().get(0).get(it.end())
                        
                        val startX = startL.x() * imageWidth * scaleFactor + postTranslateX
                        val startY = startL.y() * imageHeight * scaleFactor + postTranslateY
                        val endX = endL.x() * imageWidth * scaleFactor + postTranslateX
                        val endY = endL.y() * imageHeight * scaleFactor + postTranslateY
                        
                        canvas.drawLine(startX, startY, endX, endY, linePaint)
                    }
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
        results = poseLandmarkerResults
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        // Fix Bug 1: Calculate CENTER_CROP scaling properly
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

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }
}