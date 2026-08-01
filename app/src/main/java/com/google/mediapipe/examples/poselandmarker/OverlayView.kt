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

        // 1. DRAW OUR SEAMLESS HUMAN SILHOUETTE
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

            // --- THE MAGIC TRICK ---
            // We create a temporary layer for transparency so the body parts don't overlap visually!
            val alphaPaint = Paint().apply {
                alpha = if (isPoseMatched) 255 else 160 // 160 = semi-transparent
            }
            val layerId = canvas.saveLayer(0f, 0f, canvas.width.toFloat(), canvas.height.toFloat(), alphaPaint)

            // The paint itself is FULLY OPAQUE (Solid White or Solid Green)
            val silhouettePaint = Paint().apply {
                color = if (isPoseMatched) Color.GREEN else Color.WHITE
                style = Paint.Style.FILL_AND_STROKE
                strokeJoin = Paint.Join.ROUND
                strokeCap = Paint.Cap.ROUND
                isAntiAlias = true
            }

            if (pose.size >= 33) {
                // Refined, more natural human proportions
                val headRadius = drawScale * 0.15f
                val torsoRoundness = drawScale * 0.08f
                val bicepThickness = drawScale * 0.10f
                val forearmThickness = drawScale * 0.07f
                val thighThickness = drawScale * 0.13f
                val calfThickness = drawScale * 0.09f
                val neckThickness = drawScale * 0.08f

                val nose = getPoint(0)
                val p11 = getPoint(11) // L Shoulder
                val p12 = getPoint(12) // R Shoulder
                val p23 = getPoint(23) // L Hip
                val p24 = getPoint(24) // R Hip

                // Find the center of the shoulders to attach the neck
                val shoulderCenterX = (p11.x + p12.x) / 2f
                val shoulderCenterY = (p11.y + p12.y) / 2f

                // Draw the Neck
                silhouettePaint.strokeWidth = neckThickness
                canvas.drawLine(shoulderCenterX, shoulderCenterY, nose.x, nose.y, silhouettePaint)

                // Draw the Head (shifted slightly up so the nose is the center of the face)
                silhouettePaint.strokeWidth = 5f
                canvas.drawCircle(nose.x, nose.y - (headRadius * 0.3f), headRadius, silhouettePaint)

                // Draw the Solid Torso
                val torsoPath = android.graphics.Path()
                torsoPath.moveTo(p11.x, p11.y)
                torsoPath.lineTo(p12.x, p12.y)
                torsoPath.lineTo(p24.x, p24.y)
                torsoPath.lineTo(p23.x, p23.y)
                torsoPath.close()

                silhouettePaint.strokeWidth = torsoRoundness
                canvas.drawPath(torsoPath, silhouettePaint)

                // Draw Limbs
                fun drawLimb(startIdx: Int, endIdx: Int, thickness: Float) {
                    val start = getPoint(startIdx)
                    val end = getPoint(endIdx)
                    silhouettePaint.strokeWidth = thickness
                    canvas.drawLine(start.x, start.y, end.x, end.y, silhouettePaint)
                }

                // Arms
                drawLimb(11, 13, bicepThickness)
                drawLimb(13, 15, forearmThickness)
                drawLimb(12, 14, bicepThickness)
                drawLimb(14, 16, forearmThickness)

                // Legs
                drawLimb(23, 25, thighThickness)
                drawLimb(25, 27, calfThickness)
                drawLimb(24, 26, thighThickness)
                drawLimb(26, 28, calfThickness)
            }

            // Fuse the layer to the screen! This removes all the ugly overlapping joints.
            canvas.restoreToCount(layerId)
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