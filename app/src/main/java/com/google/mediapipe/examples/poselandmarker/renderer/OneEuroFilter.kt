package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.PointF
import kotlin.math.abs

/**
 * One Euro Filter — velocity-aware temporal smoothing.
 *
 * When the landmark is stationary, the filter applies heavy smoothing (low cutoff frequency)
 * to eliminate jitter. When the landmark moves quickly, the filter increases the cutoff
 * frequency to maintain responsiveness and avoid visible lag.
 *
 * Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
 * CHI 2012.
 *
 * @param minCutoff  Minimum cutoff frequency (Hz). Lower = smoother when stationary. Default 1.0
 * @param beta       Speed coefficient. Higher = more responsive to fast movement. Default 0.007
 * @param dCutoff    Derivative cutoff frequency (Hz). Smoothing applied to the speed signal. Default 1.0
 */
class OneEuroFilter(
    private val minCutoff: Float = 1.0f,
    private val beta: Float = 0.007f,
    private val dCutoff: Float = 1.0f
) {
    // ── Internal state per filtered point ──
    private var initialized = false
    private var prevX = 0f
    private var prevY = 0f
    private var prevDx = 0f
    private var prevDy = 0f
    private var prevTimestamp = 0L

    /**
     * Compute the smoothing factor alpha from the cutoff frequency and time delta.
     *
     * alpha = 1 / (1 + tau/dt)  where tau = 1/(2*PI*cutoff)
     */
    private fun smoothingFactor(cutoff: Float, dt: Float): Float {
        val tau = 1.0f / (2.0f * Math.PI.toFloat() * cutoff)
        return 1.0f / (1.0f + tau / dt)
    }

    /**
     * Apply exponential smoothing: out = alpha * current + (1 - alpha) * previous
     */
    private fun exponentialSmoothing(alpha: Float, current: Float, previous: Float): Float {
        return alpha * current + (1.0f - alpha) * previous
    }

    /**
     * Filter a single 2D point at the given timestamp (milliseconds).
     *
     * @param x         Raw X coordinate
     * @param y         Raw Y coordinate
     * @param timestamp Current frame timestamp in milliseconds
     * @return Filtered (smoothed) point
     */
    fun filter(x: Float, y: Float, timestamp: Long): PointF {
        if (!initialized) {
            // First frame — initialize state, no filtering possible
            prevX = x
            prevY = y
            prevDx = 0f
            prevDy = 0f
            prevTimestamp = timestamp
            initialized = true
            return PointF(x, y)
        }

        // Time delta in seconds, clamped to avoid division by zero or huge jumps
        val dtMs = (timestamp - prevTimestamp).coerceAtLeast(1L)
        val dt = dtMs / 1000.0f
        prevTimestamp = timestamp

        // ── Step 1: Estimate the speed (derivative) ──
        val rawDx = (x - prevX) / dt
        val rawDy = (y - prevY) / dt

        // Smooth the derivative signal
        val alphaD = smoothingFactor(dCutoff, dt)
        val filteredDx = exponentialSmoothing(alphaD, rawDx, prevDx)
        val filteredDy = exponentialSmoothing(alphaD, rawDy, prevDy)
        prevDx = filteredDx
        prevDy = filteredDy

        // ── Step 2: Compute adaptive cutoff based on speed ──
        val speedX = abs(filteredDx)
        val speedY = abs(filteredDy)
        val cutoffX = minCutoff + beta * speedX
        val cutoffY = minCutoff + beta * speedY

        // ── Step 3: Filter the position ──
        val alphaX = smoothingFactor(cutoffX, dt)
        val alphaY = smoothingFactor(cutoffY, dt)
        val filteredX = exponentialSmoothing(alphaX, x, prevX)
        val filteredY = exponentialSmoothing(alphaY, y, prevY)
        prevX = filteredX
        prevY = filteredY

        return PointF(filteredX, filteredY)
    }

    /**
     * Reset all temporal state. Call when camera stops/starts or tracking is lost.
     */
    fun reset() {
        initialized = false
        prevX = 0f
        prevY = 0f
        prevDx = 0f
        prevDy = 0f
        prevTimestamp = 0L
    }
}

/**
 * Manages an array of OneEuroFilters, one per landmark.
 * This is the main entry point for smoothing MediaPipe's 33 pose landmarks.
 *
 * @param landmarkCount Number of landmarks to track (33 for MediaPipe Pose)
 * @param minCutoff     Minimum cutoff frequency for all filters
 * @param beta          Speed coefficient for all filters
 */
class LandmarkSmoother(
    private val landmarkCount: Int = 33,
    minCutoff: Float = 1.7f,
    beta: Float = 0.01f
) {
    private val filters = Array(landmarkCount) { OneEuroFilter(minCutoff, beta) }

    /**
     * Smooth an array of landmark positions.
     *
     * @param points    Raw landmark positions (must have [landmarkCount] elements)
     * @param timestamp Frame timestamp in milliseconds
     * @return Array of smoothed landmark positions
     */
    fun smooth(points: Array<PointF>, timestamp: Long): Array<PointF> {
        require(points.size == landmarkCount) {
            "Expected $landmarkCount landmarks, got ${points.size}"
        }
        return Array(landmarkCount) { i ->
            filters[i].filter(points[i].x, points[i].y, timestamp)
        }
    }

    /**
     * Reset all filters. Call when tracking is lost or camera restarts.
     */
    fun reset() {
        filters.forEach { it.reset() }
    }
}
