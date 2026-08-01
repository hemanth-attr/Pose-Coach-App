/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min
import android.graphics.BitmapFactory
import com.google.mediapipe.examples.poselandmarker.R

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {
    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()
var targetPose: List<android.graphics.PointF>? = null
    var isPoseMatched = false

    private var targetPaint = Paint().apply {
        color = android.graphics.Color.WHITE
        strokeWidth = 12f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        alpha = 200 // Semi-transparent like Huawei
    }
    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

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
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

  override fun draw(canvas: Canvas) {
        super.draw(canvas)

        targetPose?.let { pose ->
            val centerX = canvas.width / 2f
            val centerY = canvas.height / 2f
            val drawScale = canvas.height / 3.5f // Scales the human to fit the screen nicely

            // Helper function to get coordinates
            fun getPoint(idx: Int): android.graphics.PointF {
                return android.graphics.PointF(
                    centerX + (pose[idx].x * drawScale),
                    centerY + (pose[idx].y * drawScale)
                )
            }

            // 1. The Paint that makes it look like a solid human silhouette
            val silhouettePaint = Paint().apply {
                // Semi-transparent white (like Huawei), turns Green when matched
                color = if (isPoseMatched) android.graphics.Color.GREEN else android.graphics.Color.argb(180, 255, 255, 255)
                style = Paint.Style.FILL_AND_STROKE
                strokeJoin = Paint.Join.ROUND
                strokeCap = Paint.Cap.ROUND
                isAntiAlias = true
            }

            // Ensure we have all 33 points loaded
            if (pose.size >= 33) {
                // Calculate dynamic body thicknesses based on screen size
                val headRadius = drawScale * 0.18f
                val torsoRoundness = drawScale * 0.1f
                val bicepThickness = drawScale * 0.12f
                val forearmThickness = drawScale * 0.09f
                val thighThickness = drawScale * 0.16f
                val calfThickness = drawScale * 0.12f

                // 2. Draw the Head
                val nose = getPoint(0)
                silhouettePaint.strokeWidth = 5f 
                canvas.drawCircle(nose.x, nose.y - (headRadius * 0.5f), headRadius, silhouettePaint)

                // 3. Draw the Solid Torso
                val p11 = getPoint(11) // L Shoulder
                val p12 = getPoint(12) // R Shoulder
                val p23 = getPoint(23) // L Hip
                val p24 = getPoint(24) // R Hip

                val torsoPath = android.graphics.Path()
                torsoPath.moveTo(p11.x, p11.y)
                torsoPath.lineTo(p12.x, p12.y)
                torsoPath.lineTo(p24.x, p24.y)
                torsoPath.lineTo(p23.x, p23.y)
                torsoPath.close()
                
                silhouettePaint.strokeWidth = torsoRoundness // Gives the torso soft shoulders/hips
                canvas.drawPath(torsoPath, silhouettePaint)

                // 4. Draw Thick "Human" Limbs
                fun drawLimb(startIdx: Int, endIdx: Int, thickness: Float) {
                    val start = getPoint(startIdx)
                    val end = getPoint(endIdx)
                    silhouettePaint.strokeWidth = thickness
                    canvas.drawLine(start.x, start.y, end.x, end.y, silhouettePaint)
                }

                // Arms
                drawLimb(11, 13, bicepThickness) // Left Bicep
                drawLimb(13, 15, forearmThickness) // Left Forearm
                drawLimb(12, 14, bicepThickness) // Right Bicep
                drawLimb(14, 16, forearmThickness) // Right Forearm

                // Legs
                drawLimb(23, 25, thighThickness) // Left Thigh
                drawLimb(25, 27, calfThickness) // Left Calf
                drawLimb(24, 26, thighThickness) // Right Thigh
                drawLimb(26, 28, calfThickness) // Right Calf
                
                // Hands and Feet (Creates rounded stubs at the end of limbs)
                drawLimb(15, 19, forearmThickness * 0.8f) // L Hand
                drawLimb(16, 20, forearmThickness * 0.8f) // R Hand
                drawLimb(27, 31, calfThickness * 0.8f) // L Foot
                drawLimb(28, 32, calfThickness * 0.8f) // R Foot
            }
        }
    }
}