package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // --- TARGET POSE VARIABLES ---
    var targetPose: List<PointF>? = null
    var isPoseMatched = false

    // --- SEGMENTATION AI VARIABLES ---
    private var segmentationMask: FloatBuffer? = null
    private var maskWidth: Int = 0
    private var maskHeight: Int = 0

    init {
        initPaints()
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

    fun setSegmentationMask(mask: FloatBuffer, width: Int, height: Int) {
        segmentationMask = mask
        maskWidth = width
        maskHeight = height
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // 1. CONFIGURE THE PREMIUM GLOWING STROKE (Huawei Style)
        val premiumOutlinePaint = Paint().apply {
            color = if (isPoseMatched) Color.GREEN else Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 10f 
            strokeJoin = Paint.Join.ROUND
            strokeCap = Paint.Cap.ROUND
            isAntiAlias = true
            setShadowLayer(15f, 0f, 0f, if (isPoseMatched) Color.GREEN else Color.WHITE) 
        }

        // 2. DRAW THE DYNAMIC HUMAN OUTLINE
        segmentationMask?.let { mask ->
            try {
                val boundaryPoints = ContourHelper.extractSmoothContour(
                    maskBuffer = mask,
                    maskWidth = maskWidth,
                    maskHeight = maskHeight,
                    viewWidth = width.toFloat(),
                    viewHeight = height.toFloat()
                )

                val organicSilhouettePath = ContourHelper.createSplinePath(boundaryPoints, tension = 0.25f)
                canvas.drawPath(organicSilhouettePath, premiumOutlinePaint)
            } catch (e: Exception) {
                // Failsafe
            }
        }
       
        // 3. DRAW THE TARGET POSE (The solid vector silhouette guideline)
        targetPose?.let { pose ->
            val centerX = canvas.width / 2f
            val centerY = canvas.height / 2f
            val drawScale = canvas.height / 3.5f

            fun getPoint(idx: Int): android.graphics.PointF {
                return android.graphics.PointF(
                    centerX + (pose[idx].x * drawScale),
                    centerY + (pose[idx].y * drawScale)
                )
            }

            val alphaPaint = Paint().apply { alpha = if (isPoseMatched) 255 else 160 }
            val layerId = canvas.saveLayer(0f, 0f, canvas.width.toFloat(), canvas.height.toFloat(), alphaPaint)

            val silhouettePaint = Paint().apply {
                color = if (isPoseMatched) Color.GREEN else Color.WHITE
                style = Paint.Style.FILL_AND_STROKE
                strokeJoin = Paint.Join.ROUND
                strokeCap = Paint.Cap.ROUND
                isAntiAlias = true
            }

            if (pose.size >= 33) {
                val headRadius = drawScale * 0.15f
                val torsoRoundness = drawScale * 0.08f
                val bicepThickness = drawScale * 0.10f
                val forearmThickness = drawScale * 0.07f
                val thighThickness = drawScale * 0.13f
                val calfThickness = drawScale * 0.09f
                val neckThickness = drawScale * 0.08f

                val nose = getPoint(0)
                val p11 = getPoint(11) 
                val p12 = getPoint(12) 
                val p23 = getPoint(23) 
                val p24 = getPoint(24) 

                val shoulderCenterX = (p11.x + p12.x) / 2f
                val shoulderCenterY = (p11.y + p12.y) / 2f

                silhouettePaint.strokeWidth = neckThickness
                canvas.drawLine(shoulderCenterX, shoulderCenterY, nose.x, nose.y, silhouettePaint)
                silhouettePaint.strokeWidth = 5f
                canvas.drawCircle(nose.x, nose.y - (headRadius * 0.3f), headRadius, silhouettePaint)

                val torsoPath = android.graphics.Path()
                torsoPath.moveTo(p11.x, p11.y)
                torsoPath.lineTo(p12.x, p12.y)
                torsoPath.lineTo(p24.x, p24.y)
                torsoPath.lineTo(p23.x, p23.y)
                torsoPath.close()

                silhouettePaint.strokeWidth = torsoRoundness
                canvas.drawPath(torsoPath, silhouettePaint)

                fun drawLimb(startIdx: Int, endIdx: Int, thickness: Float) {
                    val start = getPoint(startIdx)
                    val end = getPoint(endIdx)
                    silhouettePaint.strokeWidth = thickness
                    canvas.drawLine(start.x, start.y, end.x, end.y, silhouettePaint)
                }

                drawLimb(11, 13, bicepThickness)
                drawLimb(13, 15, forearmThickness)
                drawLimb(12, 14, bicepThickness)
                drawLimb(14, 16, forearmThickness)
                drawLimb(23, 25, thighThickness)
                drawLimb(25, 27, calfThickness)
                drawLimb(24, 26, thighThickness)
                drawLimb(26, 28, calfThickness)
            }

            canvas.restoreToCount(layerId)
        }

        // 4. ORIGINAL MEDIAPIPE DRAWING (Hidden for live camera, used for Gallery Fragment fallback)
        results?.let { poseLandmarkerResult ->
            if (segmentationMask == null) {
                for (landmark in poseLandmarkerResult.landmarks()) {
                    for (normalizedLandmark in landmark) {
                        canvas.drawPoint(
                            normalizedLandmark.x() * imageWidth * scaleFactor,
                            normalizedLandmark.y() * imageHeight * scaleFactor,
                            pointPaint
                        )
                    }

                    PoseLandmarker.POSE_LANDMARKS.forEach {
                        canvas.drawLine(
                            poseLandmarkerResult.landmarks().get(0).get(it!!.start()).x() * imageWidth * scaleFactor,
                            poseLandmarkerResult.landmarks().get(0).get(it.start()).y() * imageHeight * scaleFactor,
                            poseLandmarkerResult.landmarks().get(0).get(it.end()).x() * imageWidth * scaleFactor,
                            poseLandmarkerResult.landmarks().get(0).get(it.end()).y() * imageHeight * scaleFactor,
                            linePaint
                        )
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

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }
}