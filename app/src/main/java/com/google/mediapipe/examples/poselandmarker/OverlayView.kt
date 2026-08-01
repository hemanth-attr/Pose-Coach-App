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

    // --- OUR AI ENGINE VARIABLES ---
    var targetPose: List<PointF>? = null
    var isPoseMatched = false

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color = ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // 1. DRAW OUR FLESHY SILHOUETTE
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

            val silhouettePaint = Paint().apply {
                color = if (isPoseMatched) Color.GREEN else Color.argb(180, 255, 255, 255)
                style = Paint.Style.FILL_AND_STROKE
                strokeJoin = Paint.Join.ROUND
                strokeCap = Paint.Cap.ROUND
                isAntiAlias = true
            }

            if (pose.size >= 33) {
                val headRadius = drawScale * 0.18f
                val torsoRoundness = drawScale * 0.1f
                val bicepThickness = drawScale * 0.12f
                val forearmThickness = drawScale * 0.09f
                val thighThickness = drawScale * 0.16f
                val calfThickness = drawScale * 0.12f

                val nose = getPoint(0)
                silhouettePaint.strokeWidth = 5f
                canvas.drawCircle(nose.x, nose.y - (headRadius * 0.5f), headRadius, silhouettePaint)

                val p11 = getPoint(11)
                val p12 = getPoint(12)
                val p23 = getPoint(23)
                val p24 = getPoint(24)

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

                drawLimb(15, 19, forearmThickness * 0.8f)
                drawLimb(16, 20, forearmThickness * 0.8f)
                drawLimb(27, 31, calfThickness * 0.8f)
                drawLimb(28, 32, calfThickness * 0.8f)
            }
        }

        // 2. ORIGINAL MEDIAPIPE DRAWING (Needed for GalleryFragment)
        results?.let { poseLandmarkerResult ->
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

    // REQUIRED BY GalleryFragment
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