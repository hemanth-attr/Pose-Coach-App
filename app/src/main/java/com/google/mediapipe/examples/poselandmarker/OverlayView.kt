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

        // Change color to green if they match it!
        targetPaint.color = if (isPoseMatched) android.graphics.Color.GREEN else android.graphics.Color.WHITE
        targetPaint.alpha = 200

        targetPose?.let { pose ->
            val centerX = canvas.width / 2f
            val centerY = canvas.height / 2f
            val drawScale = canvas.height / 4f // Scales the normalized coordinates to fit the screen

            // These are the MediaPipe joint pairs (Shoulder to Elbow, Hip to Knee, etc.)
            val connections = listOf(
                Pair(11, 12), Pair(11, 13), Pair(13, 15), Pair(12, 14), Pair(14, 16), // Arms
                Pair(11, 23), Pair(12, 24), Pair(23, 24), // Torso
                Pair(23, 25), Pair(25, 27), Pair(27, 29), Pair(29, 31), Pair(31, 27), // Left Leg
                Pair(24, 26), Pair(26, 28), Pair(28, 30), Pair(30, 32), Pair(32, 28)  // Right Leg
            )

            // Draw the glowing silhouette lines
            for (connection in connections) {
                val startIdx = connection.first
                val endIdx = connection.second
                
                if (startIdx < pose.size && endIdx < pose.size) {
                    val startX = centerX + (pose[startIdx].x * drawScale)
                    val startY = centerY + (pose[startIdx].y * drawScale)
                    val endX = centerX + (pose[endIdx].x * drawScale)
                    val endY = centerY + (pose[endIdx].y * drawScale)
                    
                    canvas.drawLine(startX, startY, endX, endY, targetPaint)
                }
            }
        }
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
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
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }
}